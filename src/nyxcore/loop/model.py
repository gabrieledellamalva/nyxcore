"""A standard transformer and a looped one, sharing every line of code but the loop.

``loops=1`` is an ordinary transformer with ``layers`` distinct blocks. ``loops=R``
runs the same ``layers`` blocks R times: R times the compute, the same parameters.
That is the idea behind looped (Universal-Transformer-style) models.
"""

from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm1(h)
        h = h + self.attn(x, x, x, need_weights=False)[0]
        return h + self.mlp(self.norm2(h))


class RootFinder(nn.Module):
    def __init__(self, n: int, d: int = 128, heads: int = 4, layers: int = 2, loops: int = 1) -> None:
        super().__init__()
        self.loops = loops
        self.node, self.parent = nn.Embedding(n, d), nn.Embedding(n, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(h))

    def forward(self, ids: torch.Tensor, parents: torch.Tensor, loops: int | None = None, return_states: bool = False):
        """Logits (batch, n, n). With ``return_states`` also the residual stream after every loop."""
        x0 = self.node(ids) + self.parent(parents)
        h, states = x0, []
        for r in range(loops or self.loops):
            if r:
                h = h + x0  # input injection: every loop re-reads the problem, so state cannot drift away from it
            for block in self.blocks:
                h = block(h)
            states.append(h)
        logits = self.decode(h)
        return (logits, states) if return_states else logits

    @torch.no_grad()
    def predict_per_loop(self, ids: torch.Tensor, parents: torch.Tensor, loops: int) -> torch.Tensor:
        """The model's answer after every loop: (loops, batch, n). What it would say if stopped there."""
        x0 = self.node(ids) + self.parent(parents)
        h, answers = x0, []
        for r in range(loops):
            if r:
                h = h + x0
            for block in self.blocks:
                h = block(h)
            answers.append(self.decode(h).argmax(-1))
        return torch.stack(answers)

    def layer_states(self, ids: torch.Tensor, parents: torch.Tensor) -> list[torch.Tensor]:
        """Residual stream after each block of a single pass (for a non-looped model)."""
        h, states = self.node(ids) + self.parent(parents), []
        for block in self.blocks:
            h = block(h)
            states.append(h)
        return states

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
