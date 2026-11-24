"""Root finding in a random forest: a task whose difficulty is exactly a depth.

Every node names its parent; a root names itself. The model must output, for each
node, the root of its tree. A node at depth d needs d parent lookups, so the work a
model has to do is known per node, and "can it go deeper than it was trained for"
is a clean question. Node ids are random symbols and node order is shuffled, so
nothing but the parent links carries information.
"""

from __future__ import annotations

import math

import torch


def sample_forests(batch: int, n: int, depth_range: tuple[int, int], generator: torch.Generator, trees: int = 3, chain_bias: float = 0.7):
    """Return (ids, parent_ids, root_ids, depth), each (batch, n), on CPU.

    Every forest has ``trees`` trees of equal size, so guessing a root without
    following links is right only 1/trees of the time. Each sample draws its own
    depth cap from ``depth_range``; ``chain_bias`` is the chance a new node extends
    the deepest eligible branch of its tree, which makes deep nodes common enough to measure.
    """
    lo, hi = depth_range
    rand = lambda *shape: torch.rand(*shape, generator=generator)
    cap = torch.randint(lo, hi + 1, (batch,), generator=generator)
    rows = torch.arange(batch)
    tree = torch.arange(n) % trees
    parent = torch.zeros(batch, n, dtype=torch.long)
    depth = torch.zeros(batch, n, dtype=torch.long)
    parent[:, :trees] = torch.arange(trees)
    for i in range(trees, n):
        d = depth[:, :i]
        eligible = (d < cap[:, None]) & (tree[:i] == tree[i])
        deepest = (d == d.masked_fill(~eligible, -1).max(dim=1, keepdim=True).values) & eligible
        score = rand(batch, i) + 2.0 * (deepest & (rand(batch, 1) < chain_bias))
        choice = score.masked_fill(~eligible, -1.0).argmax(dim=1)
        parent[:, i] = choice
        depth[:, i] = depth[rows, choice] + 1
    root = parent.clone()
    for _ in range(math.ceil(math.log2(n)) + 1):  # pointer jumping reaches every root
        root = root.gather(1, root)
    label = rand(batch, n).argsort(dim=1)  # node index -> random symbol
    order = rand(batch, n).argsort(dim=1)  # shuffle positions
    pick = lambda t: t.gather(1, order)
    return pick(label), pick(label.gather(1, parent)), pick(label.gather(1, root)), pick(depth)
