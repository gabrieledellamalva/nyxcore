"""Run the probe matrix: model x context demand x canary depth x seed."""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

from ..providers import resolve
from ..schema import TraceWriter, load_runs
from .agent import run_episode
from .worlds import TOPICS, build_runbook

FIXED_OVERHEAD_TOKENS = 420  # system prompt, tool schemas, question, per-call framing


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-")


def calibrate(provider, model: str) -> float:
    """Characters per token of the synthetic text for this model's tokenizer."""
    sample = "\n".join(build_runbook(seed=0, topic="storage", pages=3, chars_per_page=2400, depth=0.5).pages)
    tokens = provider.count_tokens(model, sample)
    return len(sample) / tokens if tokens else 3.6


def run_matrix(
    models: list[str],
    out_dir: str | Path,
    *,
    context_window: int = 8192,
    fills: tuple[float, ...] = (0.25, 0.5, 0.75, 0.95, 1.25),
    depths: tuple[float, ...] = (0.0, 0.5, 1.0),
    seeds: int = 2,
    pages: int = 6,
    log=lambda line: print(line, file=sys.stderr, flush=True),
) -> None:
    out_dir = Path(out_dir)
    topics = list(TOPICS)
    for spec in models:
        provider, model = resolve(spec)
        path = out_dir / f"{_slug(spec)}.jsonl"
        done = {ep.id for ep in load_runs(path) if ep.outcome} if path.exists() else set()
        writer = TraceWriter(path)
        ratio = calibrate(provider, model)
        log(f"{spec}: {ratio:.2f} chars/token, window {context_window}")
        cells = list(itertools.product(fills, depths, range(seeds)))
        for n, (fill, depth, seed) in enumerate(cells):
            episode_id = f"{_slug(spec)}-f{fill:g}-d{depth:g}-s{seed}"
            if episode_id in done:
                continue
            topic = topics[n % len(topics)]
            budget = max(600, fill * context_window - FIXED_OVERHEAD_TOKENS)
            book = build_runbook(seed=seed, topic=topic, pages=pages, chars_per_page=int(budget / pages * ratio), depth=depth)
            task = {
                "kind": "runbook-canary", "topic": topic, "service": book.service, "fill_target": fill, "depth": depth,
                "seed": seed, "pages": pages, "canary_page": book.canary_page, "trap_page": book.trap_page,
                "canary_code": book.canary_code, "context_window": context_window,
            }
            try:
                ep = run_episode(provider, model, book, episode_id=episode_id, task=task, context_window=context_window,
                                 chars_per_token=ratio, writer=writer, seed=seed)
            except Exception as exc:  # one broken call must not sink a long run
                log(f"  {episode_id}: ERROR {type(exc).__name__}: {exc}")
                continue
            o = ep.outcome
            log(
                f"  [{n + 1}/{len(cells)}] fill {fill:<4g} depth {depth:<3g} {topic:<10} "
                f"{'recalled' if o['recalled'] else 'LOST    '} {'TRAP ' if o['trap_obeyed'] else '     '}"
                f"{'trunc ' if o['truncated'] else '      '}{len(ep.steps)} steps {o['wall_s']}s"
            )
