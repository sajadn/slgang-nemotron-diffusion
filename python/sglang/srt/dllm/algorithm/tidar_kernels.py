"""
Triton kernels for TiDAR self-speculation algorithm.

Contains:
- stochastic_verify_triton: fused verification kernel (replaces per-request Python loop)
- build_tidar_decode_inputs_pooled: PyTorch-based input builder using pre-allocated buffers
"""

import torch
import triton
import triton.language as tl


def build_tidar_decode_inputs_pooled(
    seq_lens_cpu: torch.Tensor,
    cur_prefix_gpu: torch.Tensor,
    block_size: int,
    B: int,
    mask_id: int,
    prev_draft_tokens: torch.Tensor,
    cached_offsets: torch.Tensor,
    cached_draft_mask: torch.Tensor,
    positions_out: torch.Tensor,
    draft_tokens_out: torch.Tensor,
    custom_mask_out: torch.Tensor,
):
    """
    Build positions, draft tokens, and custom_mask using pre-allocated buffers.

    Avoids allocations (torch.ones, torch.cat) and CPU-GPU syncs (.item())
    by writing directly to pre-allocated output tensors.

    Args:
        seq_lens_cpu: [bs] int64 on CPU - current prefix lengths
        cur_prefix_gpu: [bs] int32/int64 on GPU - current prefix lengths
        block_size: int
        B: block_size * (block_size + 1)
        mask_id: mask token id
        prev_draft_tokens: [bs * block_size] int32 on GPU
        cached_offsets: [B] int64 on GPU - static position offsets
        cached_draft_mask: [B, B] bool on GPU - static draft attention mask
        positions_out: [max_bs * B] int64 on GPU - pre-allocated
        draft_tokens_out: [max_bs * B] int32 on GPU - pre-allocated
        custom_mask_out: large bool buffer on GPU - pre-allocated

    Returns:
        (draft_tokens, positions, custom_mask) - slices of pre-allocated buffers
    """
    bs = len(seq_lens_cpu)

    # --- Positions ---
    pos = positions_out[: bs * B].view(bs, B)
    for b in range(bs):
        torch.add(
            cached_offsets, cur_prefix_gpu[b].to(torch.int64), out=pos[b]
        )
    positions = positions_out[: bs * B]

    # --- Draft tokens: [prev_draft | mask_id padding] ---
    dt = draft_tokens_out[: bs * B].view(bs, B)
    dt[:, :block_size] = prev_draft_tokens.view(bs, block_size)
    dt[:, block_size:] = mask_id
    draft_tokens = draft_tokens_out[: bs * B]

    # --- Custom mask: [B rows × (prefix_len + B) cols] per request ---
    offset = 0
    for b in range(bs):
        prefix_len = int(seq_lens_cpu[b])
        total_kv = prefix_len + B
        m = custom_mask_out[offset : offset + B * total_kv].view(B, total_kv)
        m[:, :prefix_len] = True
        m[:, prefix_len : prefix_len + B] = cached_draft_mask
        offset += B * total_kv
    custom_mask = custom_mask_out[:offset]

    return draft_tokens.contiguous(), positions.contiguous(), custom_mask.contiguous()


@triton.jit
def _stochastic_verify_kernel(
    # Inputs
    mixed_probs_ptr,  # [bs, block_size, vocab_size] float32
    prev_draft_ptr,  # [bs, block_size] int32
    prev_probs_ptr,  # [bs, block_size] float32
    rand_vals_ptr,  # [bs, block_size] float32 (pre-generated)
    remaining_ptr,  # [bs] int32
    is_last_iter: tl.constexpr,  # bool - force accept all on last iter
    # Outputs
    accept_cnt_ptr,  # [bs] int32
    # Constants
    block_size: tl.constexpr,
    vocab_size: tl.constexpr,
):
    """One program per request. Computes accept_cnt via stochastic verification."""
    bid = tl.program_id(0)
    remaining = tl.load(remaining_ptr + bid)

    # Skip completed requests
    if remaining <= 0:
        tl.store(accept_cnt_ptr + bid, 0)
        return

    n_verify = tl.minimum(block_size, remaining)

    # Force-accept on last iteration
    if is_last_iter:
        tl.store(accept_cnt_ptr + bid, n_verify)
        return

    # Stochastic verification: check positions 1..n_verify-1
    # Token at position 0 is always accepted (first AR token)
    # Use found flag instead of break (Triton doesn't support break)
    accept_cnt = n_verify
    found_rejection = 0  # 0 = not found, 1 = found

    base_probs = bid * block_size * vocab_size
    base_draft = bid * block_size
    base_prev_probs = bid * block_size
    base_rand = bid * block_size

    for pos in range(1, block_size):
        # Skip if beyond verify range or already rejected
        if pos < n_verify and found_rejection == 0:
            # Load draft token index at this position
            draft_idx = tl.load(prev_draft_ptr + base_draft + pos).to(tl.int64)

            # Load target probability: mixed_probs[bid, pos-1, draft_idx]
            target_prob = tl.load(
                mixed_probs_ptr + base_probs + (pos - 1) * vocab_size + draft_idx
            )

            # Load draft probability
            draft_prob = tl.load(prev_probs_ptr + base_prev_probs + pos)

            # Compute ratio
            ratio = tl.where(draft_prob > 0.0, target_prob / draft_prob, 0.0)

            # Load random value
            rand_val = tl.load(rand_vals_ptr + base_rand + pos)

            # Check rejection
            if ratio < rand_val:
                accept_cnt = pos
                found_rejection = 1

    # Cap at remaining
    accept_cnt = tl.minimum(accept_cnt, remaining)
    tl.store(accept_cnt_ptr + bid, accept_cnt)


def stochastic_verify_triton(
    mixed_probs: torch.Tensor,
    prev_draft: torch.Tensor,
    prev_probs: torch.Tensor,
    rand_vals: torch.Tensor,
    remaining: torch.Tensor,
    is_last_iter: bool,
    accept_cnt_out: torch.Tensor,
    block_size: int,
):
    """
    Triton wrapper: batched stochastic verification.

    Fuses softmax-gather, ratio computation, and first-rejection search into
    a single kernel launch per batch. Eliminates per-request CPU-GPU syncs.

    Args:
        mixed_probs: [bs, block_size, vocab_size] float32
        prev_draft: [bs, block_size] int32
        prev_probs: [bs, block_size] float32
        rand_vals: [bs, block_size] float32 - pre-generated random values
        remaining: [bs] int32 - remaining tokens needed per request
        is_last_iter: bool - force accept all on last iteration
        accept_cnt_out: [bs] int32 - pre-allocated output
        block_size: int
    """
    bs = mixed_probs.shape[0]
    vocab_size = mixed_probs.shape[2]

    _stochastic_verify_kernel[(bs,)](
        mixed_probs,
        prev_draft,
        prev_probs,
        rand_vals,
        remaining,
        is_last_iter,
        accept_cnt_out,
        block_size=block_size,
        vocab_size=vocab_size,
    )
    return accept_cnt_out
