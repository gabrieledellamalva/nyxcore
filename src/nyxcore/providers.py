"""Chat providers behind one neutral message format.

Neutral messages are plain dicts: ``{"role", "content"}`` plus, for assistant turns,
``"tool_calls": [ToolCall]`` and, for tool results, ``"tool_call_id"`` and ``"name"``.
Each provider renders them into its own wire format.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ChatResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    tokens_per_s: float | None = None
    done_reason: str | None = None


class Provider(Protocol):
    name: str

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        context_window: int | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 512,
    ) -> ChatResult: ...

    def count_tokens(self, model: str, text: str) -> int | None: ...


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (TypeError, json.JSONDecodeError):
        return {"_raw": str(raw)}


class OllamaProvider:
    """Native Ollama API: the only local route that lets us pin ``num_ctx`` per request."""

    name = "ollama"

    def __init__(self, base_url: str | None = None, timeout: float = 600.0) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        if not self.base_url.startswith("http"):
            self.base_url = "http://" + self.base_url
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._think: dict[str, Any] = {}

    @staticmethod
    def _render(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content", "")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.args}} for c in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def _think_setting(self, model: str) -> Any:
        # Reasoning effort is kept minimal so that speed numbers compare like with like.
        if model not in self._think:
            self._think[model] = "low" if "gpt-oss" in model else False
        return self._think[model]

    def chat(self, model, messages, tools=None, *, context_window=None, temperature=0.0, seed=None, max_tokens=512):
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if context_window:
            options["num_ctx"] = context_window
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {"model": model, "messages": self._render(messages), "stream": False, "options": options}
        if tools:
            payload["tools"] = tools
        think = self._think_setting(model)
        if think is not None:
            payload["think"] = think
        t0 = time.perf_counter()
        resp = self._client.post("/api/chat", json=payload)
        if resp.status_code == 400 and "think" in payload and "think" in resp.text.lower():
            self._think[model] = None  # model has no reasoning switch; never send it again
            payload.pop("think")
            resp = self._client.post("/api/chat", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"ollama {resp.status_code}: {resp.text[:300]}")  # keep the server's own reason
        data = resp.json()
        latency = time.perf_counter() - t0
        msg = data.get("message") or {}
        calls = [
            ToolCall(c.get("id") or f"call_{uuid.uuid4().hex[:8]}", c["function"]["name"], _parse_args(c["function"].get("arguments")))
            for c in msg.get("tool_calls") or []
        ]
        eval_ns = data.get("eval_duration") or 0
        completion = data.get("eval_count")
        return ChatResult(
            text=msg.get("content") or "",
            tool_calls=calls,
            thinking=msg.get("thinking") or "",
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=completion,
            latency_s=latency,
            tokens_per_s=(completion / (eval_ns / 1e9)) if completion and eval_ns else None,
            done_reason=data.get("done_reason"),
        )

    def count_tokens(self, model: str, text: str) -> int | None:
        """Exact token count of ``text`` for this model, measured as a prompt-size difference."""

        def prompt_size(content: str) -> int | None:
            res = self.chat(model, [{"role": "user", "content": content}], context_window=32768, max_tokens=1)
            return res.prompt_tokens

        base, full = prompt_size("x"), prompt_size("x\n" + text)
        return None if base is None or full is None else max(1, full - base)


class OpenAICompatProvider:
    """Any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, LM Studio, Anthropic's compat layer)."""

    name = "openai"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 600.0) -> None:
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    @staticmethod
    def _render(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                        for c in m["tool_calls"]
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, model, messages, tools=None, *, context_window=None, temperature=0.0, seed=None, max_tokens=512):
        payload: dict[str, Any] = {"model": model, "messages": self._render(messages), "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
        if seed is not None:
            payload["seed"] = seed
        t0 = time.perf_counter()
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - t0
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        calls = [
            ToolCall(c.get("id") or f"call_{uuid.uuid4().hex[:8]}", c["function"]["name"], _parse_args(c["function"].get("arguments")))
            for c in msg.get("tool_calls") or []
        ]
        completion = usage.get("completion_tokens")
        return ChatResult(
            text=msg.get("content") or "",
            tool_calls=calls,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=completion,
            latency_s=latency,
            tokens_per_s=(completion / latency) if completion and latency else None,  # wall-clock, includes network
            done_reason=choice.get("finish_reason"),
        )

    def count_tokens(self, model: str, text: str) -> int | None:
        return None


def resolve(spec: str) -> tuple[Provider, str]:
    """``"ollama:qwen3:8b"`` -> (OllamaProvider, "qwen3:8b"). A bare name means Ollama."""
    kind, _, model = spec.partition(":")
    if kind == "ollama" and model:
        return OllamaProvider(), model
    if kind == "openai" and model:
        return OpenAICompatProvider(), model
    return OllamaProvider(), spec
