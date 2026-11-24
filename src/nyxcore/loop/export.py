"""Export for the viewer and the README: the results table, two analyses and the forest scene.

Everything the viewer draws is measured. A node is "solved" at a point of the
computation when the model's own output layer, read at that point, already names
the right root. Tree shapes are the actual evaluation forests; height is node depth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import RootFinder
from .task import sample_forests
from .train import EVAL_DEPTH, GRID, N_NODES, TRAIN_DEPTH, TREES, collect, pick_device

APPS_PER_TICK = 2  # one tick of the animation = two layer applications = one loop of a 2-layer looped model
TICKS = 8
FORESTS = 30
PAIR = ("standard-8", "loop-2xN")


def _load(runs: Path, name: str, device: str, seed: int = 0) -> RootFinder | None:
    cfg = next((c for c in GRID if c.name == name), None)
    path = runs / "ckpt" / f"{name}-s{seed}.pt"
    if cfg is None or not path.exists():
        return None
    model = RootFinder(N_NODES, cfg.d, cfg.heads, cfg.layers, cfg.loops).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


@torch.no_grad()
def analyse(runs: Path, device: str, name: str = "loop-2xN", max_loops: int = 12, batches: int = 8, batch: int = 512) -> dict:
    """Two questions about a looped model, averaged over its seeds.

    1. Reach: after k loops, how deep a node can it solve for good? (Solved for good =
       right at loop k and at every later loop.)
    2. Halting: if each forest stops as soon as one more loop changes no answer, how
       many loops does it spend, and what does that cost in accuracy?
    """
    solved = np.zeros((max_loops, EVAL_DEPTH + 1))
    seen = np.zeros(EVAL_DEPTH + 1)
    groups = {key: {"forests": 0, "loops": 0.0, "nodes": 0, "adaptive": 0, "fixed8": 0} for key in ("trained", "deeper", "all")}
    seeds = 0
    for seed in range(8):
        model = _load(runs, name, device, seed)
        if model is None:
            continue
        seeds += 1
        data = torch.Generator().manual_seed(4242)
        for _ in range(batches):
            ids, parents, roots, depth = sample_forests(batch, N_NODES, (1, EVAL_DEPTH), data, TREES)
            answers = model.predict_per_loop(ids.to(device), parents.to(device), max_loops).cpu()
            ok = answers == roots
            for_good = torch.flip(torch.cumprod(torch.flip(ok.int(), [0]), 0), [0]).bool()
            for d in range(EVAL_DEPTH + 1):
                mask = depth == d
                solved[:, d] += for_good[:, mask].sum(1).numpy()
                seen[d] += int(mask.sum())
            unchanged = (answers[1:] == answers[:-1]).all(-1)  # loop k+1 agrees with loop k on every node
            used = torch.where(unchanged.any(0), unchanged.int().argmax(0) + 2, torch.full((batch,), max_loops))
            adaptive = ok[used - 1, torch.arange(batch)]
            deeper = depth.max(1).values > TRAIN_DEPTH
            for key, mask in (("trained", ~deeper), ("deeper", deeper), ("all", torch.ones(batch, dtype=torch.bool))):
                g = groups[key]
                g["forests"] += int(mask.sum())
                g["loops"] += float(used[mask].sum())
                g["nodes"] += int(mask.sum()) * N_NODES
                g["adaptive"] += int(adaptive[mask].sum())
                g["fixed8"] += int(ok[7][mask].sum())
    if not seeds:
        return {}
    frac = solved / np.maximum(seen, 1)
    reach = []
    for k in range(max_loops):
        d = 0
        while d + 1 <= EVAL_DEPTH and frac[k, d + 1] >= 0.95:
            d += 1
        reach.append(d)
    return {
        "model": name, "seeds": seeds, "max_loops": max_loops, "layers": next(c.layers for c in GRID if c.name == name),
        "solved_by_loop": np.round(frac, 4).tolist(), "reach": reach,
        "halting": {key: {"forests": g["forests"], "mean_loops": round(g["loops"] / g["forests"], 3),
                          "acc_adaptive": round(g["adaptive"] / g["nodes"], 4), "acc_fixed8": round(g["fixed8"] / g["nodes"], 4)}
                    for key, g in groups.items()},
    }


@torch.no_grad()
def baseline(runs: Path, device: str, name: str = "standard-8", batches: int = 8, batch: int = 512) -> dict:
    """Node accuracy of a standard model on exactly the forests ``analyse`` uses, in the same three groups."""
    groups = {key: [0, 0] for key in ("trained", "deeper", "all")}
    for seed in range(8):
        model = _load(runs, name, device, seed)
        if model is None:
            continue
        data = torch.Generator().manual_seed(4242)
        for _ in range(batches):
            ids, parents, roots, depth = sample_forests(batch, N_NODES, (1, EVAL_DEPTH), data, TREES)
            ok = model(ids.to(device), parents.to(device)).argmax(-1).cpu() == roots
            deeper = depth.max(1).values > TRAIN_DEPTH
            for key, mask in (("trained", ~deeper), ("deeper", deeper), ("all", torch.ones(batch, dtype=torch.bool))):
                groups[key][0] += int(ok[mask].sum())
                groups[key][1] += int(mask.sum()) * N_NODES
    return {"model": name, **{key: round(hit / max(1, total), 4) for key, (hit, total) in groups.items()}}


def _layout(parent: np.ndarray, depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Place every tree of every forest on a field: x/z on the ground, y = depth.

    The largest child continues its parent's trunk with a slight drift; the other
    children branch outwards. Returns (forests * nodes, 3) with x/z centred on 0.
    """
    forests, n = parent.shape
    cols = int(np.ceil(np.sqrt(forests * TREES * 1.25)))
    rows = int(np.ceil(forests * TREES / cols))
    slots = [(c, r) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    spacing = 2.7
    xyz = np.zeros((forests, n, 3))
    slot = 0
    for f in range(forests):
        children: dict[int, list[int]] = {i: [] for i in range(n)}
        for i in range(n):
            if parent[f, i] != i:
                children[int(parent[f, i])].append(i)
        size = np.ones(n)
        for i in np.argsort(-depth[f]):
            if parent[f, i] != i:
                size[parent[f, i]] += size[i]
        for i in np.argsort(depth[f]):
            if parent[f, i] == i:
                c, r = slots[slot]
                slot += 1
                xyz[f, i] = [(c - (cols - 1) / 2) * spacing + rng.normal(0, 0.35), 0.0, (r - (rows - 1) / 2) * spacing + rng.normal(0, 0.35)]
            for rank, child in enumerate(sorted(children[int(i)], key=lambda c: -size[c])):
                if rank == 0:
                    offset = rng.normal(0, 0.16, 2)
                else:
                    angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(0.45, 0.85)
                    offset = radius * np.array([np.cos(angle), np.sin(angle)])
                xyz[f, child] = [xyz[f, i, 0] + offset[0], depth[f, child], xyz[f, i, 2] + offset[1]]
    return xyz.reshape(-1, 3)


@torch.no_grad()
def _solved_per_tick(model: RootFinder, kind: str, ids, parents, roots) -> dict:
    x0 = model.node(ids) + model.parent(parents)
    if kind == "loop":
        per_tick = max(1, APPS_PER_TICK // len(model.blocks))
        _, states = model(ids, parents, loops=TICKS * per_tick, return_states=True)
        states = states[per_tick - 1 :: per_tick]
    else:
        states = model.layer_states(ids, parents)[APPS_PER_TICK - 1 :: APPS_PER_TICK]
    ok = torch.stack([model.decode(h).argmax(-1) == roots for h in [x0, *states]]).flatten(1).cpu().numpy()
    return {"max_tick": len(states), "ok": ["".join("1" if v else "0" for v in tick) for tick in ok],
            "acc": [round(float(tick.mean()), 4) for tick in ok]}


def export_loop(runs: str | Path, out: str | Path, pair: tuple[str, str] = PAIR) -> str:
    runs, out = Path(runs), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = collect(runs)
    device = pick_device()
    ids, parents, roots, depth = (t.to(device) for t in sample_forests(FORESTS, N_NODES, (TRAIN_DEPTH, EVAL_DEPTH), torch.Generator().manual_seed(7), TREES))
    local_parent = (parents[:, :, None] == ids[:, None, :]).float().argmax(-1)
    offsets = torch.arange(FORESTS, device=device)[:, None] * N_NODES
    xyz = _layout(local_parent.cpu().numpy(), depth.cpu().numpy(), np.random.default_rng(11))
    view = {
        "ticks": TICKS, "apps_per_tick": APPS_PER_TICK, "nodes": FORESTS * N_NODES, "train_depth": TRAIN_DEPTH, "eval_depth": EVAL_DEPTH,
        "depth": depth.flatten().tolist(), "parent": (local_parent + offsets).flatten().tolist(),
        "xyz": np.round(xyz, 3).flatten().tolist(), "models": [],
    }
    kinds = {m["name"]: m["kind"] for m in summary["models"]}
    for name in pair:
        model = _load(runs, name, device)
        if model is not None:
            view["models"].append({"name": name, "kind": kinds[name], "params": model.n_params, "layers": len(model.blocks),
                                   **_solved_per_tick(model, kinds[name], ids, parents, roots)})
    analysis = analyse(runs, device)
    if analysis:
        analysis["baseline"] = baseline(runs, device)
    (out / "loop.json").write_text(json.dumps({**summary, "analysis": analysis, "view": view}, separators=(",", ":")))
    return f"loop.json: {len(summary['models'])} models, scene for {[m['name'] for m in view['models']]} -> {out}/"


def _mean(acc: dict, depths) -> float:
    values = [acc[str(d)] if str(d) in acc else acc[d] for d in depths]
    return sum(values) / len(values)


def markdown_table(runs: str | Path) -> str:
    summary = collect(runs)
    task = summary["task"]
    seen, beyond = range(1, task["train_depth"] + 1), range(task["train_depth"] + 1, task["eval_depth"] + 1)
    lines = [f"| Model | Layers x loops | Parameters | Depth 1–{task['train_depth']} (trained) | Depth {task['train_depth'] + 1}–{task['eval_depth']} (never seen) | Seeds |", "|---|---|---|---|---|---|"]
    for m in summary["models"]:
        loops = f"{m['loop_range'][0]}–{m['loop_range'][1]} (eval {m['loops']})" if m["loop_range"] else m["loops"]
        lines.append(f"| `{m['name']}` | {m['layers']} x {loops} | {m['params']:,} | {_mean(m['acc_by_depth'], seen):.1%} | {_mean(m['acc_by_depth'], beyond):.1%} | {len(m['seeds'])} |")
    return "\n".join(lines)


def figure_svg(runs: str | Path, logo_path: str | Path | None = None) -> str:
    """Accuracy by depth as a standalone SVG in the project's own style."""
    summary = collect(runs)
    task = summary["task"]
    models = {m["name"]: m for m in summary["models"]}
    W, H, L, R, T, B = 1200, 675, 96, 920, 150, 560
    depths = list(range(1, task["eval_depth"] + 1))
    X = lambda d: L + (d - 1) / (task["eval_depth"] - 1) * (R - L)
    Y = lambda v: B - v * (B - T)
    lime, white, grey, dim = "#bdfc23", "#eef2f7", "#5b6470", "#8a94a3"
    styles = [("standard-4", grey, 1.5, "standard, 4 layers"), ("standard-8", white, 2.5, "standard, 8 layers"),
              ("loop-2x4", "#6f9416", 1.5, "looped, 2 layers x 4"), ("loop-2xN", lime, 3.5, "looped, 2 layers x N")]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="-apple-system, \'Helvetica Neue\', Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#000"/>',
        f'<text x="{L}" y="64" fill="{white}" font-size="30" font-weight="600">Past its training depth, a standard transformer falls below chance. A looped one keeps going.</text>'
        if False else f'<text x="{L}" y="62" fill="{white}" font-size="28" font-weight="600">Deeper than training: standard transformers collapse, a looped one holds.</text>',
        f'<text x="{L}" y="94" fill="{dim}" font-size="16">Share of tree nodes whose root is named correctly, by how many lookups the node needs. Mean of {len(next(iter(models.values()))["seeds"])} seeds.</text>',
    ]
    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{L}" x2="{R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#1b2028"/><text x="{L - 14}" y="{Y(v) + 5:.1f}" fill="{dim}" font-size="14" text-anchor="end">{v:.0%}</text>')
    for d in depths:
        parts.append(f'<text x="{X(d):.1f}" y="{B + 28}" fill="{dim}" font-size="14" text-anchor="middle">{d}</text>')
    parts.append(f'<text x="{(L + R) / 2:.0f}" y="{B + 62}" fill="{dim}" font-size="15" text-anchor="middle">depth of the node (lookups needed)</text>')
    edge = (X(task["train_depth"]) + X(task["train_depth"] + 1)) / 2
    parts.append(f'<rect x="{edge:.1f}" y="{T}" width="{R - edge:.1f}" height="{B - T}" fill="#ffffff" opacity="0.035"/>')
    parts.append(f'<line x1="{edge:.1f}" x2="{edge:.1f}" y1="{T - 14}" y2="{B}" stroke="{white}" stroke-width="1" opacity="0.6"/>')
    parts.append(f'<text x="{edge - 10:.1f}" y="{T - 20}" fill="{dim}" font-size="14" text-anchor="end">seen in training</text><text x="{edge + 10:.1f}" y="{T - 20}" fill="{white}" font-size="14">never seen</text>')
    chance = 1 / task["trees"]
    parts.append(f'<line x1="{L}" x2="{R}" y1="{Y(chance):.1f}" y2="{Y(chance):.1f}" stroke="{dim}" stroke-dasharray="4 6"/><text x="{L + 8}" y="{Y(chance) - 8:.1f}" fill="{dim}" font-size="13">guessing</text>')
    labels = []
    for name, colour, width, label in styles:
        if name not in models:
            continue
        acc = models[name]["acc_by_depth"]
        pts = " ".join(f"{X(d):.1f},{Y(acc[str(d)] if str(d) in acc else acc[d]):.1f}" for d in depths)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')
        last = acc[str(depths[-1])] if str(depths[-1]) in acc else acc[depths[-1]]
        labels.append([Y(last), colour, f"{label} · {models[name]['params'] / 1e6:.2f}M", width > 2])
    labels.sort(key=lambda item: item[0])
    for i in range(1, len(labels)):  # keep end labels from overlapping
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 22)
    for y, colour, text, bold in labels:
        parts.append(f'<text x="{R + 16}" y="{y + 5:.1f}" fill="{colour}" font-size="14" font-weight="{600 if bold else 400}">{text}</text>')
    if logo_path and Path(logo_path).exists():
        path = Path(logo_path).read_text().split('d="')[1].split('"')[0]
        parts.append(f'<g transform="translate({W - 150},{H - 62}) scale(0.036)"><path fill="{lime}" d="{path}"/></g>'
                     f'<text x="{W - 114}" y="{H - 42}" fill="{white}" font-size="15" font-weight="600" letter-spacing="3">NYXCORE</text>')
    parts.append("</svg>")
    return "\n".join(parts)
