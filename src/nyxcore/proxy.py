"""Recording proxy: point any agent at this port instead of its model server.

Requests are forwarded untouched and responses are streamed straight back; a copy of
the response is parsed afterwards for token usage and timing. Chat calls on Ollama's
``/api/chat`` and on OpenAI-style ``/v1/chat/completions`` become steps. Calls are
grouped into episodes by the ``x-nyxcore-session`` header (default: one per model).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

from .schema import Capture, Episode, Step, TraceWriter
from .server import LocalOnly

CHAT_PATHS = ("/api/chat", "/v1/chat/completions")
DROP_REQUEST = {"host", "content-length", "accept-encoding", "connection", "x-nyxcore-session"}
DROP_RESPONSE = {"content-length", "transfer-encoding", "connection", "content-encoding"}


def parse_usage(path: str, raw: str) -> dict[str, Any]:
    """Pull text, token counts and generation time out of a (possibly streamed) chat response."""
    text, prompt, completion, eval_s = [], None, None, None
    chunks: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not chunks:
        try:
            chunks = [json.loads(raw)]
        except json.JSONDecodeError:
            return {"output": "", "prompt_tokens": None, "completion_tokens": None, "tokens_per_s": None}
    for chunk in chunks:
        if path.startswith("/api/"):
            text.append((chunk.get("message") or {}).get("content") or "")
            prompt = chunk.get("prompt_eval_count", prompt)
            completion = chunk.get("eval_count", completion)
            if chunk.get("eval_duration"):
                eval_s = chunk["eval_duration"] / 1e9
        else:
            for choice in chunk.get("choices") or []:
                part = choice.get("message") or choice.get("delta") or {}
                text.append(part.get("content") or "")
            usage = chunk.get("usage") or {}
            prompt = usage.get("prompt_tokens", prompt)
            completion = usage.get("completion_tokens", completion)
    return {
        "output": "".join(text), "prompt_tokens": prompt, "completion_tokens": completion,
        "tokens_per_s": (completion / eval_s) if completion and eval_s else None,
    }


def build_proxy(upstream: str, writer: TraceWriter) -> LocalOnly:
    client = httpx.AsyncClient(base_url=upstream.rstrip("/"), timeout=None)
    counters: dict[str, int] = defaultdict(int)

    async def handle(request: Request) -> StreamingResponse:
        body = await request.body()
        path = request.url.path
        headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST}
        started = time.time()
        upstream_request = client.build_request(request.method, path, params=request.query_params, content=body, headers=headers)
        response = await client.send(upstream_request, stream=True)
        is_chat = request.method == "POST" and path in CHAT_PATHS and response.status_code == 200
        captured: list[bytes] = []

        async def relay():
            try:
                async for chunk in response.aiter_bytes():
                    if is_chat:
                        captured.append(chunk)
                    yield chunk
            finally:
                await response.aclose()
                if is_chat:
                    record(request, body, path, b"".join(captured).decode("utf-8", "replace"), started)

        out_headers = {k: v for k, v in response.headers.items() if k.lower() not in DROP_RESPONSE}
        return StreamingResponse(relay(), status_code=response.status_code, headers=out_headers)

    def record(request: Request, body: bytes, path: str, raw: str, started: float) -> None:
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            payload = {}
        model = str(payload.get("model", "unknown"))
        session = request.headers.get("x-nyxcore-session") or f"proxy-{model}"
        if session not in counters:
            writer.episode(Episode(session, model, "proxy", {"kind": "proxy", "path": path}))
        usage = parse_usage(path, raw)
        messages = payload.get("messages") or [{}]
        last = messages[-1].get("content")
        writer.step(Step(
            session, counters[session], started, time.time() - started,
            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
            context_window=(payload.get("options") or {}).get("num_ctx"), tokens_per_s=usage["tokens_per_s"],
            observation=last if isinstance(last, str) else None, output=usage["output"],
        ))
        counters[session] += 1

    app = Starlette(routes=[Route("/{path:path}", handle, methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])])
    return LocalOnly(app)


def serve_proxy(upstream: str, port: int, out: str, capture: Capture) -> None:
    import uvicorn

    print(f"NYXCORE proxy: http://127.0.0.1:{port} -> {upstream}  (capture={capture}, trace={out})")
    uvicorn.run(build_proxy(upstream, TraceWriter(out, capture)), host="127.0.0.1", port=port, log_level="warning")
