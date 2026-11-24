"""Trace schema and JSONL storage.

A run is one append-only JSONL file. Three record types share it:
``episode`` (who ran what), ``step`` (one model call) and ``outcome`` (the grade).
Appending per record means a crash mid-run loses at most the call in flight.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Capture = Literal["full", "metadata"]
_CONTENT_FIELDS = ("observation", "output", "tool_args")


@dataclass(slots=True)
class Step:
    episode_id: str
    index: int
    started_at: float
    latency_s: float
    prompt_tokens: int | None = None  # provider-reported, the only exact number here
    completion_tokens: int | None = None
    sent_tokens_est: int | None = None  # what we asked the model to hold (can exceed the window)
    context_window: int | None = None
    tokens_per_s: float | None = None
    tool: str | None = None
    tool_args: dict[str, Any] | None = None
    observation: str | None = None  # tool result the model saw after this call
    output: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Episode:
    id: str
    model: str
    provider: str
    task: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)


class TraceWriter:
    """Append-only writer. ``capture="metadata"`` drops every text field before it hits disk."""

    def __init__(self, path: str | Path, capture: Capture = "full") -> None:
        self.path = Path(path)
        self.capture = capture
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def episode(self, ep: Episode) -> None:
        self._append({"type": "episode", "id": ep.id, "model": ep.model, "provider": ep.provider, "task": ep.task})

    def step(self, step: Step) -> None:
        record = {"type": "step", **asdict(step)}
        if self.capture == "metadata":
            for key in _CONTENT_FIELDS:
                record[key] = None if key != "output" else ""
        self._append(record)

    def outcome(self, episode_id: str, outcome: dict[str, Any]) -> None:
        self._append({"type": "outcome", "episode_id": episode_id, "outcome": outcome})


def load_runs(path: str | Path) -> list[Episode]:
    """Load every ``*.jsonl`` under ``path`` (or the single file) into episodes, in file order."""
    root = Path(path)
    files = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
    episodes: dict[str, Episode] = {}
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.pop("type", None)
            if kind == "episode":
                episodes[rec["id"]] = Episode(rec["id"], rec["model"], rec["provider"], rec.get("task") or {})
            elif kind == "step" and rec["episode_id"] in episodes:
                episodes[rec["episode_id"]].steps.append(Step(**rec))
            elif kind == "outcome" and rec["episode_id"] in episodes:
                episodes[rec["episode_id"]].outcome = rec["outcome"]
    for ep in episodes.values():
        ep.steps.sort(key=lambda s: s.index)
    return [ep for ep in episodes.values() if ep.steps]
