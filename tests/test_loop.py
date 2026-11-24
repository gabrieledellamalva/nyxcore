"""Loop-lab tests. Skipped unless the optional 'loop' extra (PyTorch) is installed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nyxcore.loop.model import RootFinder  # noqa: E402
from nyxcore.loop.task import sample_forests  # noqa: E402


def test_forest_labels_match_a_brute_force_walk():
    ids, parents, roots, depth = sample_forests(32, 24, (1, 7), torch.Generator().manual_seed(0), trees=3)
    for b in range(32):
        parent_of = dict(zip(ids[b].tolist(), parents[b].tolist()))
        assert sorted(ids[b].tolist()) == list(range(24)), "every symbol appears exactly once"
        assert len(set(roots[b].tolist())) == 3, "three trees"
        for node, root, d in zip(ids[b].tolist(), roots[b].tolist(), depth[b].tolist()):
            hops = 0
            while parent_of[node] != node:
                node, hops = parent_of[node], hops + 1
            assert (node, hops) == (root, d)


def test_trees_are_balanced_so_guessing_is_chance():
    _, _, roots, _ = sample_forests(16, 24, (1, 7), torch.Generator().manual_seed(1), trees=3)
    for b in range(16):
        assert torch.bincount(roots[b]).max().item() == 8


def test_depth_cap_is_respected():
    *_, depth = sample_forests(64, 24, (2, 2), torch.Generator().manual_seed(2), trees=3)
    assert depth.max().item() == 2


def test_looping_adds_compute_not_parameters():
    torch.manual_seed(0)
    plain, looped = RootFinder(24, d=32, layers=2, loops=1), RootFinder(24, d=32, layers=2, loops=4)
    assert plain.n_params == looped.n_params
    ids, parents, *_ = sample_forests(4, 24, (1, 5), torch.Generator().manual_seed(3))
    logits, states = looped(ids, parents, return_states=True)
    assert logits.shape == (4, 24, 24) and len(states) == 4
    assert len(looped(ids, parents, loops=9, return_states=True)[1]) == 9
    assert len(plain.layer_states(ids, parents)) == 2


def test_one_loop_equals_the_plain_model_with_the_same_weights():
    torch.manual_seed(0)
    plain, looped = RootFinder(24, d=32, layers=2, loops=1), RootFinder(24, d=32, layers=2, loops=3)
    looped.load_state_dict(plain.state_dict())
    ids, parents, *_ = sample_forests(2, 24, (1, 5), torch.Generator().manual_seed(4))
    assert torch.allclose(plain(ids, parents), looped(ids, parents, loops=1), atol=1e-6)
