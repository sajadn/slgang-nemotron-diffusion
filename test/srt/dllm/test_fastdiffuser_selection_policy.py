import torch
from types import SimpleNamespace

from sglang.srt.dllm.algorithm.fastdiffuser import FastDiffuser


def _make_algorithm(selection_policy):
    algo = object.__new__(FastDiffuser)
    algo.selection_policy = selection_policy
    return algo


def test_fastdiffuser_leftmost_selection_ignores_confidence_order():
    algo = _make_algorithm("leftmost")
    confidence = torch.tensor([0.1, 0.9, float("-inf"), 0.8, 0.7])
    block_mask = torch.tensor([True, True, False, True, True])
    eos_freeze = torch.zeros_like(block_mask)

    selected = algo._select_transfer_indices(
        confidence, block_mask, eos_freeze, k=2
    )

    assert selected.tolist() == [0, 1]


def test_fastdiffuser_leftmost_selection_skips_eos_frozen_positions():
    algo = _make_algorithm("leftmost")
    confidence = torch.tensor([0.9, 0.8, 0.7, 0.6])
    block_mask = torch.tensor([True, True, True, True])
    eos_freeze = torch.tensor([False, True, True, False])

    selected = algo._select_transfer_indices(
        confidence, block_mask, eos_freeze, k=3
    )

    assert selected.tolist() == [0, 3]


def test_fastdiffuser_confidence_selection_preserves_existing_policy():
    algo = _make_algorithm("confidence")
    confidence = torch.tensor([0.1, 0.9, float("-inf"), 0.8])
    block_mask = torch.tensor([True, True, False, True])
    eos_freeze = torch.zeros_like(block_mask)

    selected = algo._select_transfer_indices(
        confidence, block_mask, eos_freeze, k=2
    )

    assert selected.tolist() == [1, 3]


def test_fastdiffuser_records_logprobs_for_committed_tokens():
    algo = _make_algorithm("leftmost")
    logprob_grid = torch.full((1, 4), float("nan"))
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [0.0, 3.0, -2.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 4.0],
        ]
    )
    positions = torch.tensor([0, 3])
    token_ids = torch.tensor([0, 2])

    algo._record_token_logprobs(logprob_grid, 0, positions, logits, token_ids)

    expected = torch.log_softmax(logits[positions], dim=-1)[
        torch.arange(2), token_ids
    ]
    assert torch.allclose(logprob_grid[0, positions], expected)
    assert torch.isnan(logprob_grid[0, 1])
    assert torch.isnan(logprob_grid[0, 2])


def test_fastdiffuser_logprobs_are_returned_only_for_leftmost_policy():
    leftmost = _make_algorithm("leftmost")
    confidence = _make_algorithm("confidence")
    forward_batch = SimpleNamespace(return_logprob=True)

    assert leftmost._should_return_token_logprobs(forward_batch)
    assert not confidence._should_return_token_logprobs(forward_batch)

    forward_batch.return_logprob = False
    assert not leftmost._should_return_token_logprobs(forward_batch)
