"""
TiDAR utility functions for building positions and custom attention masks.

Adapted from /tmp/tidar_ref/tidar_utils.py for use within the DLLM algorithm framework.
"""

from __future__ import annotations

import torch


def build_tidar_positions_and_mask_prefill(
    seq_lens: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    device: str,
):
    """
    Build positions and custom mask for TiDAR initial draft (prefill).

    For each sequence, emits block_size query tokens at positions
    [seq_len, seq_len+1, ..., seq_len+block_size-1].
    The custom mask is fully True: each draft query attends to all prefix KV
    and all other draft tokens (bidirectional within the draft block).

    Returns: (draft_tokens, positions, custom_mask)
    """
    bs = len(seq_lens)
    offsets = torch.arange(block_size, device=device, dtype=torch.long)
    positions = (
        torch.repeat_interleave(
            seq_lens.to(torch.long).to(device), repeats=block_size
        )
        + offsets.repeat(bs)
    )

    draft_tokens = torch.full(
        (bs * block_size,), mask_token_id, dtype=torch.int32, device=device
    )

    # Fully-True mask: each of the block_size queries attends to
    # (seq_len[b] + block_size) KV positions per sequence.
    total_mask_len = int(
        block_size * seq_lens.sum().item() + bs * block_size * block_size
    )
    custom_mask = torch.ones((total_mask_len,), dtype=torch.bool, device=device)
    return draft_tokens.contiguous(), positions.contiguous(), custom_mask.contiguous()


def build_tidar_draft_mask(block_size: int, device: str) -> torch.Tensor:
    """
    Build the static B×B draft-to-draft attention mask for TiDAR quadratic decode.
    This only depends on block_size and can be cached across iterations.

    Returns: [B, B] bool tensor (B = block_size * (block_size + 1))
    """
    B = block_size * (block_size + 1)
    draft_mask = torch.zeros((B, B), dtype=torch.bool, device=device)
    # First block_size rows: lower-triangular (causal AR)
    draft_mask[:block_size, :block_size] = torch.tril(
        torch.ones((block_size, block_size), dtype=torch.bool, device=device)
    )
    # Blocks j=1..block_size: bidirectional within own block + attend to first j AR positions
    for j in range(1, block_size + 1):
        row_start = j * block_size
        row_end = (j + 1) * block_size
        draft_mask[row_start:row_end, row_start:row_end] = True
        draft_mask[row_start:row_end, :j] = True
    return draft_mask


def build_tidar_position_offsets(block_size: int, device: str) -> torch.Tensor:
    """
    Build the static position offsets for TiDAR quadratic decode.
    offsets[j*block_size + k] = j + k for j in [0..block_size], k in [0..block_size-1]

    Returns: [B] long tensor
    """
    return (
        torch.arange(block_size + 1, device=device, dtype=torch.long).unsqueeze(0)
        + torch.arange(block_size + 1, device=device, dtype=torch.long).unsqueeze(1)
    )[:, :block_size].contiguous().view(-1)


def build_tidar_positions_and_mask_decode(
    seq_lens: torch.Tensor,
    block_size: int,
    prev_draft_tokens: torch.Tensor,
    mask_token_id: int,
    device: str,
    cached_draft_mask: torch.Tensor = None,
    cached_offsets: torch.Tensor = None,
    cached_mask_tokens: torch.Tensor = None,
):
    """
    Build positions and custom mask for TiDAR quadratic decode iteration.

    Emits B = block_size * (block_size + 1) query tokens per sequence.
    Layout:
      - First block_size tokens: AR queries (prev_draft_tokens, causal mask)
      - Next block_size blocks of block_size tokens each: diffusion queries
        (mask tokens, bidirectional within block + attending to first j AR positions)

    When cached_draft_mask/cached_offsets are provided, avoids rebuilding them.

    Returns: (draft_tokens, positions, custom_mask)
    """
    bs = len(seq_lens)
    B = block_size * (block_size + 1)

    # Positions
    if cached_offsets is None:
        cached_offsets = build_tidar_position_offsets(block_size, device)
    positions = (
        torch.repeat_interleave(
            seq_lens.to(torch.long).to(device), repeats=B
        )
        + cached_offsets.repeat(bs)
    )

    # Draft tokens: first block_size = prev_draft_tokens, rest = mask_token_id
    draft_token = prev_draft_tokens.view(bs, block_size)
    if cached_mask_tokens is not None and bs == 1:
        draft_token = torch.cat(
            [draft_token, cached_mask_tokens[:1]], dim=-1
        ).view(-1)
    else:
        draft_token = torch.cat(
            [
                draft_token,
                torch.full(
                    (bs, block_size * block_size),
                    mask_token_id,
                    dtype=torch.int32,
                    device=device,
                ),
            ],
            dim=-1,
        ).view(-1)

    # Build custom mask using cached draft_mask (avoids Python loop + tril rebuild)
    if cached_draft_mask is None:
        cached_draft_mask = build_tidar_draft_mask(block_size, device)

    masks = []
    for i in range(bs):
        prefix_len = int(seq_lens[i].item())
        # [B, prefix_len] all-ones + [B, B] cached draft mask → [B, prefix_len+B] → flat
        prefix_mask = torch.ones(
            (B, prefix_len), dtype=torch.bool, device=device
        )
        masks.append(
            torch.cat([prefix_mask, cached_draft_mask], dim=-1).view(-1)
        )

    custom_mask = torch.cat(masks)
    return draft_token.contiguous(), positions.contiguous(), custom_mask.contiguous()
