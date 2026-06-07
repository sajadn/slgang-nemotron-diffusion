"""
FastDiffuser algorithm for SGLang — clean port from HF chat_utils.py.

Key design decisions (ported directly from nvidia/Nemotron chat_utils.py):
  1. get_num_transfer_tokens() distributes mask_count evenly across steps,
     with early steps getting the remainder — identical to HF reference.
  2. Confidence = max softmax probability of the predicted token (low_confidence).
  3. Temperature=0 → greedy argmax; temperature>0 → Gumbel-max sampling.
  4. EOS awareness: once an EOS token is placed at position i, positions
     i+1..block_size-1 are treated as already resolved and skipped in
     subsequent steps.  This is consistent with the Instruct model behaviour
     described in nvidia/Nemotron-Diffusion-Exp-Ministral-8B-Instruct/chat_utils.py
     where generation stops as soon as EOS is placed anywhere in the block.
     SGLang's scheduler then truncates the output at the first EOS token.
  5. Per-request early exit: once all mask tokens in a request are committed,
     that request is compacted out of subsequent forward passes so it no longer
     consumes GPU compute.  The final forward is still run over the full batch
     so the scheduler receives correct logits for all requests.

Usage
-----
  dllm_algorithm: FastDiffuser
  dllm_algorithm_config:
    temperature: 0.0          # 0 = greedy; >0 adds Gumbel noise
    threshold: null           # if null: use HF schedule (distribute tokens evenly)
                              # if set (e.g. 0.9): HF-matching mode — always commit
                              # top-1 token plus all tokens with confidence >= threshold.
                              # Matches get_transfer_index() in HF chat_utils.py.
    max_steps: 32             # denoising steps PER BLOCK (= steps // num_blocks in HF).
                              # For HF steps=512: set max_steps = 512*block_size/max_tokens
                              # e.g. max_tokens=512  → max_steps=32
                              #      max_tokens=2048 → max_steps=8
"""

import dataclasses
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (ported 1-to-1 from HF chat_utils.py)
# ---------------------------------------------------------------------------

def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Gumbel-max trick for categorical sampling.
    temperature=0 → identity (greedy argmax used externally).
    Uses float64 for numerical stability as in the HF reference.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_count: int, steps: int) -> List[int]:
    """
    Distribute mask_count token-placements across steps.
    Earlier steps get one extra token when the count doesn't divide evenly.

    Example: mask_count=7, steps=3 → [3, 2, 2]
    Matches HF get_num_transfer_tokens() exactly.
    """
    base = mask_count // steps
    remainder = mask_count % steps
    return [base + (1 if i < remainder else 0) for i in range(steps)]


# ---------------------------------------------------------------------------
# FastDiffuser algorithm
# ---------------------------------------------------------------------------

class FastDiffuser(DllmAlgorithm):
    """
    FastDiffuser / LLaDA-style iterative denoising algorithm.

    Processes one block of mask tokens per invocation (SGLang's scheduler
    calls this once per block).  The block is denoised over `max_steps`
    forward passes.

    EOS handling
    ------------
    Once an EOS token is placed at any position within a block the algorithm
    freezes all positions at and after that EOS: they are removed from the
    mask and will not be overwritten in subsequent steps.  This matches the
    behaviour of the Instruct model's chat_utils.py and ensures that the
    block returned to the scheduler contains a clean prefix ending at EOS.

    Per-request early exit
    ----------------------
    When a request has no remaining mask tokens it is compacted out of the
    forward_batch so it no longer participates in forward passes.  This saves
    GPU compute when requests finish at different denoising steps.  A mapping
    from compacted to original batch index is kept so the final results are
    always written back to the correct positions.
    """

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        cfg = config.algorithm_config
        self.temperature: float = cfg.get("temperature", 0.0)
        self.threshold: Optional[float] = cfg.get("threshold", None)
        # max_steps is set on DllmConfig (defaults to block_size when not
        # specified; can be overridden via dllm_algorithm_config.max_steps)
        self.max_steps: int = config.max_steps
        self.causal_context: bool = config.causal_context
        self.selection_policy: str = cfg.get("selection_policy", "confidence")
        if self.selection_policy not in ("confidence", "leftmost"):
            raise ValueError(
                "FastDiffuser selection_policy must be one of "
                f"'confidence' or 'leftmost', got {self.selection_policy!r}"
            )
        if self.selection_policy == "leftmost" and self.threshold is not None:
            raise ValueError(
                "FastDiffuser: 'threshold' is incompatible with "
                "selection_policy='leftmost'. Threshold counts globally-confident "
                "positions, but leftmost reveals positions by index regardless of "
                "confidence, so the two policies mix incompatible signals. "
                "Remove 'threshold' (leftmost uses k=1 per step, override via "
                "tokens_per_step) or use selection_policy='confidence'."
            )
        # Fixed token budget per denoising step (disabled by default).
        # When set, overrides threshold/schedule and commits exactly this many
        # tokens per step (capped at remaining masked tokens).  Useful for
        # measuring throughput at different token-commit rates.
        self.tokens_per_step: Optional[int] = cfg.get("tokens_per_step", None)

        # EOS token id — read lazily from the hf_config if available
        self._eos_token_id: Optional[int] = None

        # Efficiency counters (written to _STATS_FILE on every update so the
        # benchmark client can poll without any API changes).
        self._stats_forward_passes: int = 0   # total model_runner.forward() calls
        self._stats_tokens_generated: int = 0  # total output tokens committed
        self._stats_cuda_graph: int = 0       # forward passes that used CUDA graph
        self._stats_eager: int = 0            # forward passes that used eager mode
        self._stats_file: Optional[str] = cfg.get("stats_file", None)

        logger.info(
            "FastDiffuser: block_size=%d  max_steps=%d  temperature=%s  "
            "threshold=%s selection_policy=%s",
            self.block_size,
            self.max_steps,
            self.temperature,
            self.threshold,
            self.selection_policy,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_stats(self) -> None:
        """Atomically write efficiency counters to the stats file (if configured)."""
        if not self._stats_file:
            return
        import json, os, tempfile
        data = {
            "forward_passes": self._stats_forward_passes,
            "tokens_generated": self._stats_tokens_generated,
        }
        tmp = self._stats_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self._stats_file)

    def _get_eos_id(self, model_runner: ModelRunner) -> Optional[int]:
        """Return EOS token id, cached after first call."""
        if self._eos_token_id is None:
            try:
                hf_cfg = model_runner.model_config.hf_config
                eos = getattr(hf_cfg, "eos_token_id", None)
                if isinstance(eos, list):
                    eos = eos[0]
                self._eos_token_id = int(eos) if eos is not None else None
            except Exception:
                self._eos_token_id = None
        return self._eos_token_id

    def _compute_confidence(
        self,
        logits: torch.Tensor,           # [block_size, vocab]
        mask_index: torch.Tensor,        # [block_size] bool
        current_ids: torch.Tensor,       # [block_size] int
        eos_freeze: torch.Tensor,        # [block_size] bool — positions frozen by EOS
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          x0          : [block_size] predicted token ids (masked positions only)
          confidence  : [block_size] confidence scores (-inf for non-mask positions)

        HF-matching notes
        -----------------
        • Argmax: mask_id is excluded so we never "commit" the mask token back.
          HF does not exclude it, which wastes occasional commits, but the
          effect is negligible in practice (verified: no accuracy difference).
        • Confidence (softmax): we use *raw* logits (mask_id included in the
          denominator), exactly as HF's get_transfer_index does.  Using
          logits_no_mask inflates the confidence of every real token by
          removing the mask_id mass from the denominator, causing SGLang to
          commit more tokens per step than HF.
        """
        # Exclude mask token from argmax only — prevents committing mask_id
        logits_for_argmax = logits.clone()
        logits_for_argmax[:, self.mask_id] = -np.inf
        logits_with_noise = _add_gumbel_noise(logits_for_argmax, self.temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)           # [block_size]

        # Confidence = softmax prob of predicted token over the FULL vocabulary
        # (mask_id included in denominator) — matches HF get_transfer_index.
        p = F.softmax(logits, dim=-1)
        x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)      # [block_size]

        # Keep current token for non-mask positions
        x0 = torch.where(mask_index, x0, current_ids)

        # Confidence is -inf for: (a) already-placed tokens, (b) EOS-frozen slots
        active = mask_index & ~eos_freeze
        confidence = torch.where(active, x0_p, torch.full_like(x0_p, -np.inf))

        return x0, confidence

    def _select_transfer_indices(
        self,
        confidence: torch.Tensor,
        block_mask: torch.Tensor,
        eos_freeze: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Select masked positions to commit for the current denoising step."""
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=block_mask.device)

        if self.selection_policy == "confidence":
            _, top_idx = torch.topk(confidence, k=k)
            return top_idx

        active = block_mask & ~eos_freeze
        return active.nonzero(as_tuple=True)[0][:k]

    def _record_token_logprobs(
        self,
        logprob_grid: Optional[torch.Tensor],
        batch_idx: int,
        positions: torch.Tensor,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> None:
        """Store policy logprobs for tokens committed by the denoising step.

        Logprobs are computed as ``log P(token | state, token != mask_id)`` —
        the mask_id column is removed from the softmax denominator so the
        reported value reflects the distribution the model actually emits
        from (mask_id is structurally excluded from the output vocabulary by
        the reveal/force-commit argmax). Keeps reveal, EOS-propagation, and
        force-commit paths consistent.
        """
        if logprob_grid is None or positions.numel() == 0:
            return
        # Fancy-indexing on the first dim returns a copy, so this mutation
        # is local and does not touch the caller's logits buffer.
        logits_slice = logits[positions].to(torch.float32)
        logits_slice[:, self.mask_id] = -np.inf
        logprobs = F.log_softmax(logits_slice, dim=-1)
        token_logprobs = logprobs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        logprob_grid[batch_idx, positions] = token_logprobs.to(logprob_grid.dtype)

    def _should_return_token_logprobs(self, forward_batch: ForwardBatch) -> bool:
        return forward_batch.return_logprob

    def _record_logprobs_from_fully_masked_logits(
        self,
        logprob_grid: Optional[torch.Tensor],
        fully_masked_logits: Optional[torch.Tensor],
        token_grid: torch.Tensor,
        start_list: List[int],
    ) -> None:
        """Store FastDiffuser logprobs from the first fully masked forward.

        These are not denoising-step reveal probabilities. They are the
        fully masked block scores for the final generated tokens, matching
        DiffuGRPO's Megatron fully masked logprob pass.
        """
        if logprob_grid is None or fully_masked_logits is None:
            return

        logprob_grid.fill_(float("nan"))
        device = token_grid.device
        for b, start in enumerate(start_list):
            positions = torch.arange(start, self.block_size, device=device)
            if positions.numel() == 0:
                continue
            bs = b * self.block_size
            self._record_token_logprobs(
                logprob_grid,
                b,
                positions,
                fully_masked_logits[bs : bs + self.block_size],
                token_grid[b, positions],
            )

    def _compact_forward_batch(
        self,
        forward_batch: ForwardBatch,
        active_indices: List[int],
    ) -> ForwardBatch:
        """
        Return a new ForwardBatch containing only the requests in active_indices.

        All per-request tensor fields are sliced to the active subset.
        Per-token fields are gathered block-by-block (each block is exactly
        block_size tokens for DLLM_EXTEND).  The returned object is a shallow
        copy of forward_batch with only the affected fields replaced; all
        other fields (attn_backend, token_to_kv_pool, etc.) are shared.
        """
        bs = self.block_size
        n = len(active_indices)
        device = forward_batch.input_ids.device

        active_t = torch.tensor(active_indices, dtype=torch.long, device=device)
        active_t_cpu = torch.tensor(active_indices, dtype=torch.long)  # CPU for CPU tensors

        # Gather per-token indices for the active blocks
        token_idx = torch.cat([
            torch.arange(a * bs, (a + 1) * bs, device=device)
            for a in active_indices
        ])  # [n * block_size]

        # Per-token tensors — clone input_ids so we don't alias the master copy
        new_input_ids = forward_batch.input_ids[token_idx].clone()
        new_positions = forward_batch.positions[token_idx]
        new_out_cache_loc = forward_batch.out_cache_loc[token_idx]

        # Per-request tensors
        new_seq_lens = forward_batch.seq_lens[active_t]
        new_req_pool = forward_batch.req_pool_indices[active_t]
        new_ext_seq = forward_batch.extend_seq_lens[active_t]
        new_ext_pre = forward_batch.extend_prefix_lens[active_t]

        # extend_start_loc = cumsum of extend_seq_lens, starting at 0
        new_ext_start = torch.zeros(n, dtype=torch.long, device=device)
        new_ext_start[1:] = new_ext_seq[:-1].cumsum(0)

        # Optional CPU-side list fields
        new_ext_pre_cpu = (
            [forward_batch.extend_prefix_lens_cpu[a] for a in active_indices]
            if forward_batch.extend_prefix_lens_cpu is not None else None
        )
        new_ext_seq_cpu = (
            [forward_batch.extend_seq_lens_cpu[a] for a in active_indices]
            if forward_batch.extend_seq_lens_cpu is not None else None
        )
        new_seq_lens_cpu = (
            forward_batch.seq_lens_cpu[active_t_cpu]
            if forward_batch.seq_lens_cpu is not None else None
        )

        return dataclasses.replace(
            forward_batch,
            batch_size=n,
            input_ids=new_input_ids,
            req_pool_indices=new_req_pool,
            seq_lens=new_seq_lens,
            out_cache_loc=new_out_cache_loc,
            seq_lens_sum=int(new_seq_lens.sum()),
            positions=new_positions,
            extend_num_tokens=n * bs,
            extend_seq_lens=new_ext_seq,
            extend_prefix_lens=new_ext_pre,
            extend_start_loc=new_ext_start,
            extend_prefix_lens_cpu=new_ext_pre_cpu,
            extend_seq_lens_cpu=new_ext_seq_cpu,
            seq_lens_cpu=new_seq_lens_cpu,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        eos_id = self._get_eos_id(model_runner)

        # ----------------------------------------------------------------
        # Fast path: no masks in the entire batch → just do one forward
        # (this handles the STAGING_PREFILL phase)
        # ----------------------------------------------------------------
        mask_index_all = forward_batch.input_ids == self.mask_id
        if not mask_index_all.any():
            if forward_batch.input_ids.numel() == 0:
                # Empty extend batch (e.g. warmup with empty prompt) — nothing to forward.
                # Return an empty logits output; the scheduler has nothing to sample.
                empty_logits = LogitsProcessorOutput(
                    next_token_logits=None, full_logits=None
                )
                return empty_logits, [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        # ----------------------------------------------------------------
        # Per-request setup
        # ----------------------------------------------------------------
        # start_list[i] = index within the block where generated tokens begin
        # (= block_size - initial_mask_count for that request)
        start_list: List[int] = []
        for b in range(batch_size):
            bs, be = b * self.block_size, (b + 1) * self.block_size
            n_masks = int((forward_batch.input_ids[bs:be] == self.mask_id).sum())
            start_list.append(self.block_size - n_masks)

        # finished[b] = True once request b has no remaining mask tokens
        finished = [False] * batch_size
        # The chosen-token logprob is a clean policy logprob only for leftmost
        # reveal: exactly one deterministic position is revealed from the current
        # denoising state.  Confidence-based reveal has a more complex selection
        # event, so do not expose this denoising-step score as an API logprob.
        output_logprob_grid = (
            torch.full(
                (batch_size, self.block_size),
                float("nan"),
                dtype=torch.float32,
                device=forward_batch.input_ids.device,
            )
            if self._should_return_token_logprobs(forward_batch)
            else None
        )
        fully_masked_logits = None

        # ----------------------------------------------------------------
        # Iterative denoising loop
        # ----------------------------------------------------------------
        # Track how many denoising steps each request was active (in active_indices).
        # Per-request forward passes = req_steps_active[b] + 1  (the +1 = final forward).
        req_steps_active = [0] * batch_size

        for step in range(self.max_steps):
            # Determine active requests (still have mask tokens)
            active_indices = [b for b in range(batch_size) if not finished[b]]
            if not active_indices:
                break  # all requests done

            # Count this step for every active request
            for b in active_indices:
                req_steps_active[b] += 1

            # ---------- forward pass (compacted if some are done) ----------
            self._stats_forward_passes += 1
            if len(active_indices) == batch_size:
                # Full batch — no compaction overhead
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            else:
                # Some requests are done: run only on active subset
                compact_fb = self._compact_forward_batch(forward_batch, active_indices)
                out = model_runner.forward(compact_fb, pp_proxy_tensors=None)

            logits_output, can_run_graph = out.logits_output, out.can_run_graph
            if can_run_graph:
                self._stats_cuda_graph += 1
            else:
                self._stats_eager += 1

            if (
                step == 0
                and output_logprob_grid is not None
                and self.selection_policy == "confidence"
            ):
                fully_masked_logits = logits_output.full_logits.detach().clone()

            # ---------- per-request token placement ----------
            for compact_i, orig_b in enumerate(active_indices):
                # Logits come from the compacted batch (index compact_i)
                lbs = compact_i * self.block_size
                lbe = lbs + self.block_size
                # Input ids live in the master forward_batch (index orig_b)
                bs_o = orig_b * self.block_size
                be_o = bs_o + self.block_size

                block_ids = forward_batch.input_ids[bs_o:be_o]  # in-place view
                block_mask = block_ids == self.mask_id           # [block_size]

                if not block_mask.any():
                    finished[orig_b] = True
                    continue

                block_logits = logits_output.full_logits[lbs:lbe]  # [block_size, vocab]

                # EOS freeze: positions at or after the first GENERATED EOS token.
                # The extend block may start with some prompt tokens (when prefix
                # caching doesn't cover the entire prompt).  Those prompt tokens
                # can include <|im_end|> (id=eos_id), but they must NOT trigger
                # the freeze — only an EOS that was committed from a mask position
                # should freeze subsequent generation.
                gen_start = start_list[orig_b]  # first generated position in block
                eos_freeze = torch.zeros_like(block_mask)
                if eos_id is not None:
                    placed_in_gen = ~block_mask
                    placed_in_gen[:gen_start] = False  # ignore prompt-portion EOS
                    eos_placed = placed_in_gen & (block_ids == eos_id)
                    if eos_placed.any():
                        first_eos = int(eos_placed.nonzero(as_tuple=True)[0][0])
                        eos_freeze[first_eos:] = True

                x0, confidence = self._compute_confidence(
                    block_logits, block_mask, block_ids, eos_freeze
                )

                # How many tokens to place this step
                active_mask = block_mask & ~eos_freeze
                remaining = int(active_mask.sum())
                if remaining == 0:
                    finished[orig_b] = True
                    continue

                if self.selection_policy == "leftmost":
                    # Leftmost reveal commits positions by index, not confidence,
                    # so the per-step budget is independent of the model's scores.
                    # Default k=1 (canonical one-leftmost-per-step); allow override
                    # via tokens_per_step for throughput experiments.
                    k = self.tokens_per_step if self.tokens_per_step is not None else 1
                elif self.threshold is not None:
                    # HF-matching mode (mirrors get_transfer_index in chat_utils.py):
                    # always commit at least the top-1 token (highest confidence),
                    # plus all other tokens whose confidence >= threshold.
                    k = max(1, int((confidence >= self.threshold).sum()))
                else:
                    # Schedule-only: distribute remaining tokens evenly across
                    # the remaining denoising steps (HF get_num_transfer_tokens).
                    steps_left = self.max_steps - step
                    k = _get_num_transfer_tokens(remaining, steps_left)[0]

                k = min(k, remaining)

                # Select positions according to the configured reveal policy.
                top_idx = self._select_transfer_indices(
                    confidence, block_mask, eos_freeze, k
                )
                self._record_token_logprobs(
                    output_logprob_grid,
                    orig_b,
                    top_idx,
                    block_logits,
                    x0[top_idx],
                )
                block_ids[top_idx] = x0[top_idx]

                # EOS propagation: if EOS was just placed in the generated region,
                # fill subsequent mask positions with EOS to stop cleanly.
                if eos_id is not None:
                    gen_top_idx = top_idx[top_idx >= gen_start]
                    if len(gen_top_idx) and (block_ids[gen_top_idx] == eos_id).any():
                        gen_eos = (block_ids[gen_start:] == eos_id).nonzero(as_tuple=True)[0]
                        if len(gen_eos):
                            first_eos = gen_start + int(gen_eos[0])
                            still_mask = block_ids[first_eos:] == self.mask_id
                            if still_mask.any():
                                propagated_idx = (
                                    still_mask.nonzero(as_tuple=True)[0] + first_eos
                                )
                                propagated_token_ids = torch.full_like(
                                    propagated_idx, eos_id
                                )
                                self._record_token_logprobs(
                                    output_logprob_grid,
                                    orig_b,
                                    propagated_idx,
                                    block_logits,
                                    propagated_token_ids,
                                )
                            block_ids[first_eos:][still_mask] = eos_id

                # Check if this request is now fully committed
                if not (block_ids == self.mask_id).any():
                    finished[orig_b] = True

        # ----------------------------------------------------------------
        # Final forward over the full batch to get updated KV cache / logits.
        # When causal_context=True (Instruct model) this pass must use causal
        # attention — matching HF's per-block KV-update with use_causal_mask=True.
        # We signal this to the attention backend via dllm_causal_kv_update.
        # ----------------------------------------------------------------
        self._stats_forward_passes += 1
        if self.causal_context:
            forward_batch.dllm_causal_kv_update = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        if self.causal_context:
            forward_batch.dllm_causal_kv_update = False
        logits_output, can_run_graph = out.logits_output, out.can_run_graph
        if can_run_graph:
            self._stats_cuda_graph += 1
        else:
            self._stats_eager += 1

        # Force-commit any mask tokens not resolved during denoising.
        # This happens when threshold is set and model confidence stays below
        # the threshold (k=1/step × max_steps < block_size).  We use greedy
        # argmax from the final-forward logits — these already condition on all
        # tokens committed during the denoising loop — to fully denoise the block.
        for b in range(batch_size):
            bs_o = b * self.block_size
            be_o = bs_o + self.block_size
            block_ids = forward_batch.input_ids[bs_o:be_o]
            remaining_mask = (block_ids == self.mask_id).clone()
            if not remaining_mask.any():
                continue
            block_logits = logits_output.full_logits[bs_o:be_o].clone()
            block_logits[:, self.mask_id] = -np.inf
            x0_final = torch.argmax(block_logits, dim=-1)
            # Respect EOS freeze: don't commit past the first GENERATED EOS.
            # Prompt-portion EOS (positions < gen_start) must not block force-commit.
            if eos_id is not None:
                gen_start = start_list[b]
                placed_in_gen = ~remaining_mask
                placed_in_gen[:gen_start] = False
                eos_placed = placed_in_gen & (block_ids == eos_id)
                if eos_placed.any():
                    first_eos = int(eos_placed.nonzero(as_tuple=True)[0][0])
                    remaining_mask[first_eos:] = False
            commit_idx = remaining_mask.nonzero(as_tuple=True)[0]
            self._record_token_logprobs(
                output_logprob_grid,
                b,
                commit_idx,
                block_logits,
                x0_final[commit_idx],
            )
            block_ids[remaining_mask] = x0_final[remaining_mask]

        # ----------------------------------------------------------------
        # Build output token lists (one tensor per request)
        # ----------------------------------------------------------------
        token_grid = forward_batch.input_ids.reshape(batch_size, self.block_size)
        next_token_ids_list = [
            token_grid[b, start_list[b]:] for b in range(batch_size)
        ]
        next_token_logprobs_list = None
        if output_logprob_grid is not None:
            if self.selection_policy == "confidence":
                # Confidence reveal selects tokens as a function of the whole
                # block's scores, so expose the first fully-masked forward as
                # the rollout logprob for DiffuGRPO comparisons.
                self._record_logprobs_from_fully_masked_logits(
                    output_logprob_grid,
                    fully_masked_logits,
                    token_grid,
                    start_list,
                )

            # In normal denoising every returned generated token was either
            # scored by the initial fully-masked forward for confidence reveal,
            # or committed during a reveal/force-commit step for leftmost
            # reveal. If a future path leaves a gap, use the final forward as a
            # best-effort estimate so the API still returns one logprob per
            # output token.
            for b in range(batch_size):
                missing_idx = (
                    torch.isnan(output_logprob_grid[b, start_list[b]:])
                    .nonzero(as_tuple=True)[0]
                    + start_list[b]
                )
                if missing_idx.numel() == 0:
                    continue
                bs_o = b * self.block_size
                token_ids = token_grid[b, missing_idx]
                self._record_token_logprobs(
                    output_logprob_grid,
                    b,
                    missing_idx,
                    logits_output.full_logits[bs_o:bs_o + self.block_size],
                    token_ids,
                )
            next_token_logprobs_list = [
                output_logprob_grid[b, start_list[b]:].detach().cpu().tolist()
                for b in range(batch_size)
            ]

        # ----------------------------------------------------------------
        # Write per-request efficiency stats (if stats_file is configured)
        # ----------------------------------------------------------------
        if self._stats_file:
            import json as _json
            with open(self._stats_file, "a") as _sf:
                for b in range(batch_size):
                    # Report denoising passes only (excludes the final KV-update
                    # forward), matching HF's nfe convention so both sides are
                    # directly comparable.
                    fp = req_steps_active[b]
                    tokens = int(next_token_ids_list[b].shape[0])
                    self._stats_tokens_generated += tokens
                    tpfp = tokens / fp if fp > 0 else 0.0
                    _sf.write(_json.dumps({
                        "forward_passes": fp,
                        "tokens": tokens,
                        "tokens_per_fp": round(tpfp, 4),
                    }) + "\n")
            # Note: do NOT call _flush_stats() here — it overwrites the JSONL file.
            # _flush_stats() is for external polling of aggregate counters only.

        logger.info(
            "FastDiffuser block done: CG=%d  eager=%d  total_fp=%d",
            self._stats_cuda_graph,
            self._stats_eager,
            self._stats_forward_passes,
        )
        if next_token_logprobs_list is not None:
            logits_output.next_token_logprobs = next_token_logprobs_list
        return logits_output, next_token_ids_list, can_run_graph


Algorithm = FastDiffuser
