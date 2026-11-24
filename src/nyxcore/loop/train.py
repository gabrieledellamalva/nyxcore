"""Train and evaluate the loop-vs-standard grid."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import RootFinder
from .task import sample_forests

N_NODES = 48
TREES = 3  # chance level for any non-root node is 1/TREES
TRAIN_DEPTH = 6  # deepest tree seen in training
EVAL_DEPTH = 15  # deepest tree evaluated: everything past TRAIN_DEPTH is extrapolation
LOOP_SWEEP = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


@dataclass(frozen=True)
class Config:
    name: str
    layers: int
    loops: int = 1  # loops used at evaluation by default
    loop_range: tuple[int, int] | None = None  # if set, each training batch draws its loop count from this range
    d: int = 128
    heads: int = 4
    seed: int = 0

    @property
    def kind(self) -> str:
        return "standard" if self.loops == 1 and not self.loop_range else "loop"


GRID = [
    Config("standard-2", layers=2),
    Config("standard-4", layers=4),
    Config("standard-8", layers=8),
    Config("loop-2x2", layers=2, loops=2),
    Config("loop-2x4", layers=2, loops=4),
    Config("loop-4x2", layers=4, loops=2),  # "run the layers twice"
    Config("loop-2xN", layers=2, loops=8, loop_range=(2, 10)),  # trained to tolerate any loop count
]


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def train(cfg: Config, steps: int, device: str, batch: int = 256, lr: float = 1e-3, log=print) -> tuple[RootFinder, dict]:
    torch.manual_seed(cfg.seed)
    data = torch.Generator().manual_seed(cfg.seed + 1)
    model = RootFinder(N_NODES, cfg.d, cfg.heads, cfg.layers, cfg.loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = 200
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    t0, loss_ema = time.perf_counter(), None
    for step in range(steps):
        cap = max(1, min(TRAIN_DEPTH, 1 + int(TRAIN_DEPTH * step / (0.5 * steps))))  # curriculum: shallow trees first
        ids, parents, roots, _ = (t.to(device) for t in sample_forests(batch, N_NODES, (1, cap), data, TREES))
        loops = cfg.loops
        if cfg.loop_range:
            loops = int(torch.randint(cfg.loop_range[0], cfg.loop_range[1] + 1, (1,), generator=data))
        loss = F.cross_entropy(model(ids, parents, loops).flatten(0, 1), roots.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        if (step + 1) % 500 == 0:
            log(f"  {cfg.name} step {step + 1}/{steps} loss {loss_ema:.4f} ({(step + 1) / (time.perf_counter() - t0):.1f} it/s)")
    seconds = time.perf_counter() - t0
    return model, {"train_s": round(seconds, 1), "steps_per_s": round(steps / seconds, 2), "final_loss": round(loss_ema, 5), "steps": steps}


@torch.no_grad()
def accuracy_by_depth(model: RootFinder, device: str, loops: int | None = None, batches: int = 12, batch: int = 512) -> dict[int, float]:
    """Share of nodes whose root is predicted correctly, per node depth, on a fixed evaluation set."""
    model.eval()
    data = torch.Generator().manual_seed(20261124)
    hit = torch.zeros(EVAL_DEPTH + 1)
    seen = torch.zeros(EVAL_DEPTH + 1)
    for _ in range(batches):
        ids, parents, roots, depth = sample_forests(batch, N_NODES, (1, EVAL_DEPTH), data, TREES)
        pred = model(ids.to(device), parents.to(device), loops).argmax(-1).cpu()
        hit += torch.bincount(depth[pred == roots], minlength=EVAL_DEPTH + 1).float()
        seen += torch.bincount(depth.flatten(), minlength=EVAL_DEPTH + 1).float()
    model.train()
    return {d: round((hit[d] / seen[d]).item(), 4) for d in range(EVAL_DEPTH + 1) if seen[d] > 0}


def run_grid(out: str | Path, steps: int = 3000, names: list[str] | None = None, seeds: tuple[int, ...] = (0,),
             log=lambda line: print(line, file=sys.stderr, flush=True)) -> None:
    """Train every (config, seed) not already on disk. One JSON per run, so parallel workers never collide."""
    out = Path(out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    device = pick_device()
    for seed in seeds:
        for base in GRID:
            if names and base.name not in names:
                continue
            cfg = Config(**{**asdict(base), "seed": seed})
            path = out / "runs" / f"{cfg.name}-s{seed}.json"
            if path.exists():
                continue
            log(f"{cfg.name} seed {seed}: {cfg.layers} layers x {cfg.loops} loops on {device}")
            model, info = train(cfg, steps, device, log=log)
            entry = {
                **asdict(cfg), "kind": cfg.kind, "params": model.n_params, "applications": cfg.layers * cfg.loops, **info,
                "acc_by_depth": accuracy_by_depth(model, device),
            }
            if cfg.kind == "loop":  # what happens when the same weights are simply run for longer
                entry["acc_by_loops"] = {r: accuracy_by_depth(model, device, loops=r, batches=6) for r in LOOP_SWEEP}
            torch.save(model.state_dict(), out / "ckpt" / f"{cfg.name}-s{seed}.pt")
            path.write_text(json.dumps(entry, indent=1))
            acc = entry["acc_by_depth"]
            mean = lambda lo, hi: sum(v for d, v in acc.items() if lo <= d <= hi) / max(1, sum(1 for d in acc if lo <= d <= hi))
            log(f"  -> {model.n_params:,} params, depth 1-{TRAIN_DEPTH}: {mean(1, TRAIN_DEPTH):.3f}, depth {TRAIN_DEPTH + 1}-{EVAL_DEPTH}: {mean(TRAIN_DEPTH + 1, EVAL_DEPTH):.3f}, {info['steps_per_s']} it/s")


def collect(out: str | Path) -> dict:
    """Merge per-run files into one summary: per model, the mean over seeds plus every seed's own curve."""
    out = Path(out)
    runs = [json.loads(f.read_text()) for f in sorted((out / "runs").glob("*.json"))]
    avg = lambda curves: {d: round(sum(c[d] for c in curves) / len(curves), 4) for d in curves[0]}
    models = []
    for base in GRID:
        mine = sorted((r for r in runs if r["name"] == base.name), key=lambda r: r["seed"])
        if not mine:
            continue
        entry = {k: mine[0][k] for k in ("name", "kind", "layers", "loops", "loop_range", "params", "applications", "steps")}
        entry["seeds"] = [r["seed"] for r in mine]
        entry["steps_per_s"] = round(sum(r["steps_per_s"] for r in mine) / len(mine), 2)
        entry["acc_by_depth"] = avg([r["acc_by_depth"] for r in mine])
        entry["acc_by_depth_seeds"] = [r["acc_by_depth"] for r in mine]
        if "acc_by_loops" in mine[0]:
            entry["acc_by_loops"] = {r: avg([m["acc_by_loops"][r] for m in mine]) for r in mine[0]["acc_by_loops"]}
        models.append(entry)
    return {
        "task": {"name": "forest-root-finding", "nodes": N_NODES, "trees": TREES, "train_depth": TRAIN_DEPTH, "eval_depth": EVAL_DEPTH},
        "torch": torch.__version__, "models": models,
    }
