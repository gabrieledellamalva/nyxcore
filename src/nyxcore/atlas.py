"""Turn recorded runs into the atlas: embeddings, a 3D layout, clusters and statistics.

Geometry, stated plainly so nobody reads more into the picture than is there:

* every episode is a column; its ground position comes from a t-SNE of episode
  embeddings, so runs over similar content stand close together;
* height is context demand: tokens the agent asked the model to hold, as a fraction
  of the window. The plane at 1.0 is the window limit; anything above it was sent
  but could not have been read;
* the sideways wobble of a column is the step's own embedding relative to its
  episode, projected on the two strongest directions of that variation.
"""

from __future__ import annotations

import json
import math
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Episode, Step, load_runs

HEIGHT = 30.0
RADIUS = 26.0
MIN_SPACING = 3.4
DIM = 4096
_WORD = re.compile(r"[a-z][a-z\-]{2,}")


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms == 0, 1.0, norms)


class LsaEmbedder:
    """Hashed TF-IDF reduced by truncated SVD (latent semantic analysis).

    Small, dependency-free and deterministic. It captures shared vocabulary, not
    meaning; swap in a neural embedder when the traces are real-world text.
    """

    name = "lsa-hash4096-svd"

    def __init__(self) -> None:
        self.idf = np.ones(DIM, dtype=np.float32)
        self.mean = np.zeros(DIM, dtype=np.float32)
        self.components = np.zeros((1, DIM), dtype=np.float32)
        self.vocab: dict[int, str] = {}

    @staticmethod
    def _bucket(token: str) -> int:
        return zlib.crc32(token.encode()) % DIM

    def _counts(self, texts: list[str], learn_vocab: bool = False) -> np.ndarray:
        matrix = np.zeros((len(texts), DIM), dtype=np.float32)
        seen: dict[int, Counter[str]] = defaultdict(Counter)
        for row, text in enumerate(texts):
            for token, count in Counter(_WORD.findall(text.lower())).items():
                bucket = self._bucket(token)
                matrix[row, bucket] += count
                if learn_vocab:
                    seen[bucket][token] += count
        if learn_vocab:
            self.vocab = {bucket: counter.most_common(1)[0][0] for bucket, counter in seen.items()}
        return np.log1p(matrix)

    def fit(self, texts: list[str], k: int = 48) -> tuple[np.ndarray, np.ndarray]:
        """Return (dense unit vectors, tf-idf rows). The second is kept for labelling clusters."""
        tf = self._counts(texts, learn_vocab=True)
        df = (tf > 0).sum(axis=0)
        self.idf = (np.log((1 + len(texts)) / (1 + df)) + 1).astype(np.float32)
        tfidf = _unit(tf * self.idf)
        self.mean = tfidf.mean(axis=0)
        _, _, vt = np.linalg.svd(tfidf - self.mean, full_matrices=False)
        self.components = vt[: max(1, min(k, vt.shape[0]))]
        return _unit((tfidf - self.mean) @ self.components.T), tfidf

    def embed(self, texts: list[str]) -> np.ndarray:
        tfidf = _unit(self._counts(texts) * self.idf)
        return _unit((tfidf - self.mean) @ self.components.T)

    def save(self, path: Path, vectors: np.ndarray) -> None:
        np.savez_compressed(path, idf=self.idf, mean=self.mean, components=self.components, vectors=vectors.astype(np.float32))

    @classmethod
    def load(cls, path: Path) -> tuple["LsaEmbedder", np.ndarray]:
        data = np.load(path)
        emb = cls()
        emb.idf, emb.mean, emb.components = data["idf"], data["mean"], data["components"]
        return emb, data["vectors"]


def step_text(step: Step) -> str:
    args = json.dumps(step.tool_args) if step.tool_args else ""
    return f"{step.tool or ''} {args} {(step.observation or '')[:3000]} {step.output[:600]}"


def wilson(k: int, n: int, z: float = 1.96) -> dict[str, float | int]:
    """Rate with a 95% Wilson interval: honest about how little a handful of runs proves."""
    if n == 0:
        return {"k": 0, "n": 0, "rate": 0.0, "lo": 0.0, "hi": 0.0}
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"k": k, "n": n, "rate": round(p, 4), "lo": round(max(0.0, centre - half), 4), "hi": round(min(1.0, centre + half), 4)}


def _relax(points: np.ndarray, spacing: float, iterations: int = 400) -> np.ndarray:
    """Push columns apart until none overlap; keeps neighbourhoods, removes pile-ups."""
    pts = points.copy()
    for _ in range(iterations):
        delta = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        overlap = np.clip(spacing - dist, 0, None)
        if not overlap.any():
            break
        direction = delta / np.where(dist[..., None] == 0, 1e-6, dist[..., None])
        pts += 0.5 * (direction * overlap[..., None]).sum(axis=1)
    return pts


def _ground_layout(episode_vecs: np.ndarray) -> np.ndarray:
    n = len(episode_vecs)
    if n == 1:
        return np.zeros((1, 2))
    if n >= 8:
        from sklearn.manifold import TSNE

        flat = TSNE(n_components=2, perplexity=min(12.0, (n - 1) / 3), init="pca", metric="cosine", random_state=0).fit_transform(episode_vecs)
    else:
        centred = episode_vecs - episode_vecs.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        flat = centred @ vt[:2].T
        if flat.shape[1] < 2:
            flat = np.pad(flat, ((0, 0), (0, 2 - flat.shape[1])))
    flat = flat - flat.mean(axis=0)
    flat *= RADIUS / (np.abs(flat).max() or 1.0)
    return _relax(flat, MIN_SPACING)


def _demand(step: Step, position: float) -> float:
    window = step.context_window
    if window and step.sent_tokens_est is not None:
        return step.sent_tokens_est / window
    if window and step.prompt_tokens is not None:
        return step.prompt_tokens / window
    return position


def peak_demand(episode: Episode) -> float:
    return max(_demand(s, (i + 1) / len(episode.steps)) for i, s in enumerate(episode.steps))


def _excerpt(step: Step, needle: str | None) -> str:
    text = step.observation or ""
    if needle and needle in text:
        at = text.index(needle)
        start = max(0, at - 160)
        return ("…" if start else "") + text[start : at + len(needle) + 200].strip() + "…"
    return text[:420] + ("…" if len(text) > 420 else "")


def _model_stats(episodes: list[Episode]) -> dict[str, Any]:
    def rate(items: list[Episode], key: str) -> dict[str, float | int]:
        return wilson(sum(1 for e in items if e.outcome.get(key)), len(items))

    graded = [e for e in episodes if "recalled" in e.outcome]
    speeds = [s.tokens_per_s for e in episodes for s in e.steps if s.tokens_per_s]
    latencies = [s.latency_s for e in episodes for s in e.steps]
    by_fill: dict[float, list[Episode]] = defaultdict(list)
    by_depth: dict[float, list[Episode]] = defaultdict(list)
    for e in graded:
        by_fill[e.task.get("fill_target")].append(e)
        by_depth[e.task.get("depth")].append(e)
    return {
        "episodes": len(episodes),
        "recall": rate(graded, "recalled"),
        "trap": rate(graded, "trap_obeyed"),
        "truncated": rate(graded, "truncated"),
        "crashed": rate(graded, "error"),
        "tokens_per_s": round(float(np.median(speeds)), 1) if speeds else None,
        "step_latency_s": round(float(np.median(latencies)), 2) if latencies else None,
        # "demand" is measured (median peak of tokens sent / window); "fill" is only the level the task aimed for.
        "by_fill": [
            {"fill": f, "demand": round(float(np.median([peak_demand(e) for e in v])), 3), "recall": rate(v, "recalled"),
             "trap": rate(v, "trap_obeyed"), "truncated": rate(v, "truncated")}
            for f, v in sorted(by_fill.items(), key=lambda kv: kv[0] or 0)
        ],
        "by_depth": [{"depth": d, "recall": rate(v, "recalled")} for d, v in sorted(by_depth.items(), key=lambda kv: kv[0] or 0)],
    }


def build_atlas(runs: str | Path, out: str | Path) -> dict[str, int]:
    episodes = load_runs(runs)
    if not episodes:
        raise SystemExit(f"no recorded episodes under {runs}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    steps = [(ei, s) for ei, ep in enumerate(episodes) for s in ep.steps]
    embedder = LsaEmbedder()
    vectors, tfidf = embedder.fit([step_text(s) for _, s in steps])
    owner = np.array([ei for ei, _ in steps])
    episode_vecs = _unit(np.stack([vectors[owner == ei].mean(axis=0) for ei in range(len(episodes))]))
    ground = _ground_layout(episode_vecs)

    deviation = vectors - episode_vecs[owner]
    _, _, vt = np.linalg.svd(deviation - deviation.mean(axis=0), full_matrices=False)
    wobble = deviation @ vt[:2].T
    if wobble.shape[1] < 2:
        wobble = np.pad(wobble, ((0, 0), (0, 2 - wobble.shape[1])))
    wobble *= 1.1 / (np.percentile(np.linalg.norm(wobble, axis=1), 95) or 1.0)
    norms = np.linalg.norm(wobble, axis=1, keepdims=True)
    wobble *= np.minimum(1.0, 1.5 / np.where(norms == 0, 1.0, norms))

    labels = np.full(len(steps), -1)
    if len(steps) >= 12:
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(min_cluster_size=max(6, len(steps) // 30), copy=True).fit_predict(vectors)

    model_ids = list(dict.fromkeys(ep.model for ep in episodes))
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    cursor = 0
    for ei, ep in enumerate(episodes):
        first = cursor
        for si, s in enumerate(ep.steps):
            demand = _demand(s, (si + 1) / len(ep.steps))
            needle = ep.task.get("canary_code") if "canary_seen" in s.flags else ("ATTENTION AUTOMATED AGENT" if "trap_seen" in s.flags else None)
            call = f"{s.tool}({json.dumps(s.tool_args, ensure_ascii=False)[:160]})" if s.tool else ""
            step_rows.append({
                "e": ei, "i": si,
                "x": round(float(ground[ei, 0] + wobble[cursor, 0]), 3),
                "y": round(float(demand * HEIGHT), 3),
                "z": round(float(ground[ei, 1] + wobble[cursor, 1]), 3),
                "demand": round(float(demand), 4),
                "pt": s.prompt_tokens, "ct": s.completion_tokens, "sent": s.sent_tokens_est,
                "tps": round(s.tokens_per_s, 1) if s.tokens_per_s else None,
                "lat": round(s.latency_s, 2), "tool": s.tool, "call": call, "flags": s.flags,
                "text": _excerpt(s, needle), "out": s.output[:300], "c": int(labels[cursor]),
            })
            cursor += 1
        episode_rows.append({
            "id": ep.id, "m": model_ids.index(ep.model), "task": ep.task, "outcome": ep.outcome,
            "x": round(float(ground[ei, 0]), 3), "z": round(float(ground[ei, 1]), 3),
            "s0": first, "s1": cursor, "peak": round(max(r["demand"] for r in step_rows[first:cursor]), 4),
        })

    clusters = []
    xz = np.array([[r["x"], r["z"]] for r in step_rows])
    global_mean = tfidf.mean(axis=0)
    for label in sorted(set(labels.tolist()) - {-1}):
        members = labels == label
        if np.linalg.norm(xz[members].std(axis=0)) > RADIUS * 0.3:
            continue  # a cluster smeared over the whole map has no place to pin a label
        lift = tfidf[members].mean(axis=0) - global_mean
        terms = [embedder.vocab[b] for b in np.argsort(-lift)[:8] if b in embedder.vocab][:3]
        centre = xz[members].mean(axis=0)
        clusters.append({"id": int(label), "label": " · ".join(terms), "n": int(members.sum()), "x": round(float(centre[0]), 2), "z": round(float(centre[1]), 2)})

    windows = [s.context_window for _, s in steps if s.context_window]
    atlas = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "height": HEIGHT,
            "context_window": Counter(windows).most_common(1)[0][0] if windows else None,
            "embedder": embedder.name, "episodes": len(episodes), "steps": len(steps),
        },
        "models": [{"id": m, "stats": _model_stats([e for e in episodes if e.model == m])} for m in model_ids],
        "episodes": episode_rows, "steps": step_rows, "clusters": clusters,
    }
    (out / "atlas.json").write_text(json.dumps(atlas, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    embedder.save(out / "embedder.npz", vectors)
    return {"episodes": len(episodes), "steps": len(steps), "clusters": len(clusters)}


def search(atlas_dir: str | Path, query: str, k: int = 12) -> list[dict[str, float | int]]:
    """Nearest recorded steps to free text, by cosine similarity in the atlas embedding."""
    embedder, vectors = LsaEmbedder.load(Path(atlas_dir) / "embedder.npz")
    q = embedder.embed([query])[0]
    if not q.any():
        return []
    scores = vectors @ q
    return [{"step": int(i), "score": round(float(scores[i]), 4)} for i in np.argsort(-scores)[:k] if scores[i] > 0.05]


def markdown_report(atlas_dir: str | Path) -> str:
    atlas = json.loads((Path(atlas_dir) / "atlas.json").read_text(encoding="utf-8"))

    def pct(r: dict[str, Any]) -> str:
        return f"{r['rate']:.0%} ({r['k']}/{r['n']}, 95% CI {r['lo']:.0%}–{r['hi']:.0%})"

    lines = [
        f"Context window pinned at {atlas['meta']['context_window']} tokens.", "",
        "| Model | Runs | Canary recalled | Trap obeyed | Silently truncated | Crashed | Median tok/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in atlas["models"]:
        s = m["stats"]
        lines.append(f"| `{m['id']}` | {s['episodes']} | {pct(s['recall'])} | {pct(s['trap'])} | {pct(s['truncated'])} | {s['crashed']['k']} | {s['tokens_per_s']} |")
    fills = sorted({row["fill"] for m in atlas["models"] for row in m["stats"]["by_fill"] if row["fill"] is not None})
    if fills:
        lines += ["", "Canary recall by context demand. Columns are the level each task aimed for; "
                  "each cell also gives the demand actually measured (median peak of tokens sent / window).", "",
                  "| Model | " + " | ".join(f"target {f:.0%}" for f in fills) + " |", "|---|" + "---|" * len(fills)]
        for m in atlas["models"]:
            cells = {row["fill"]: row for row in m["stats"]["by_fill"]}
            lines.append(f"| `{m['id']}` | " + " | ".join(
                f"{cells[f]['recall']['k']}/{cells[f]['recall']['n']} (measured {cells[f]['demand']:.0%})" if f in cells else "–" for f in fills) + " |")
    return "\n".join(lines)
