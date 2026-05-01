"""
TiDAR (self-speculation) algorithm for SGLang DLLM framework.

Uses a single "quadratic" forward pass with B = block_size * (block_size + 1)
tokens and a structured attention mask to simultaneously produce AR (causal)
and diffusion (bidirectional) predictions. Tokens are accepted speculatively,
yielding significantly more tokens per forward pass than FastDiffuser.

HF equivalent: model.self_spec_generate(...)

Usage:
    dllm_algorithm: TiDAR
    dllm_algorithm_config:
        ar_mix_weight: 0.0
        causal_context: true
        max_quad_iters: 4
        stats_file: /tmp/tidar_stats.jsonl
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.tidar_input import TiDARInput
from sglang.srt.dllm.algorithm.tidar_kernels import (
    build_tidar_decode_inputs_pooled,
)
from sglang.srt.dllm.algorithm.tidar_utils import (
    build_tidar_draft_mask,
    build_tidar_position_offsets,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

logger = logging.getLogger(__name__)


class TiDAR(DllmAlgorithm):
    """
    TiDAR self-speculation algorithm for diffusion LLMs.

    Each call to run() fills one block of block_size tokens by:
      1. Running a bidirectional DLLM_EXTEND to get an initial draft.
      2. Iterating quadratic decode steps (TARGET_VERIFY with custom_mask)
         until block_size tokens are committed.
    """

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        cfg = config.algorithm_config
        # ar_mix_weight: None = pure AR (HF default), 0.0-1.0 = mix AR+diffusion
        _amw = cfg.get("ar_mix_weight", None)
        self.ar_mix_weight = None if _amw is None else float(_amw)
        self.accept_all: bool = cfg.get("accept_all", False)
        self.max_quad_iters: int = cfg.get("max_quad_iters", self.block_size)
        self.causal_context: bool = config.causal_context
        self._eos_token_id: Optional[int] = None

        # Precomputed masks/offsets (lazily initialized on first run with device)
        self._cached_draft_mask: Optional[torch.Tensor] = None
        self._cached_offsets: Optional[torch.Tensor] = None
        self._cached_mask_tokens: Optional[torch.Tensor] = None

        # Pre-allocated tensor pool (lazily initialized)
        self._pool_positions: Optional[torch.Tensor] = None  # [max_bs * B] int64
        self._pool_draft_tokens: Optional[torch.Tensor] = None  # [max_bs * B] int32
        self._pool_custom_mask: Optional[torch.Tensor] = None  # large bool buffer
        self._pool_accept_cnt: Optional[torch.Tensor] = None  # [max_bs] int32
        self._pool_remaining: Optional[torch.Tensor] = None  # [max_bs] int32
        self._pool_max_bs: int = 0

        # Efficiency counters
        self._stats_forward_passes: int = 0
        self._stats_tokens_generated: int = 0
        self._stats_file: Optional[str] = cfg.get("stats_file", None)
        self._stats_cg_forwards: int = 0
        self._stats_eager_forwards: int = 0

        logger.info(
            "TiDAR: block_size=%d  ar_mix_weight=%s  causal_context=%s",
            self.block_size,
            self.ar_mix_weight,
            self.causal_context,
        )

    def _get_eos_id(self, model_runner: ModelRunner) -> Optional[int]:
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

    def _init_pool(self, max_bs: int, B: int, block_size: int, device) -> None:
        """Initialize or resize pre-allocated tensor pool."""
        # Assume max prefix length for mask buffer sizing (8192 is generous)
        max_prefix = 8192
        max_mask_per_req = B * (max_prefix + B)
        self._pool_positions = torch.empty(max_bs * B, dtype=torch.int64, device=device)
        self._pool_draft_tokens = torch.empty(
            max_bs * B, dtype=torch.int32, device=device
        )
        self._pool_custom_mask = torch.empty(
            max_bs * max_mask_per_req, dtype=torch.bool, device=device
        )
        self._pool_accept_cnt = torch.empty(max_bs, dtype=torch.int32, device=device)
        self._pool_remaining = torch.empty(max_bs, dtype=torch.int32, device=device)
        self._pool_max_bs = max_bs
        logger.info("TiDAR: initialized tensor pool for max_bs=%d", max_bs)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        bs = forward_batch.batch_size
        block_size = self.block_size
        B = block_size * (block_size + 1)
        device = forward_batch.input_ids.device
        eos_id = self._get_eos_id(model_runner)

        # Lazy-init cached tensors on first run (needs device)
        if self._cached_draft_mask is None:
            self._cached_draft_mask = build_tidar_draft_mask(block_size, device)
            self._cached_offsets = build_tidar_position_offsets(block_size, device)
            self._cached_mask_tokens = torch.full(
                (1, block_size * block_size),
                self.mask_id,
                dtype=torch.int32,
                device=device,
            )

        # Ensure pre-allocated pool is large enough for this batch
        if bs > self._pool_max_bs:
            self._init_pool(bs, B, block_size, device)

        # Refs for KV management
        allocator = model_runner.token_to_kv_pool_allocator
        req_to_token = model_runner.req_to_token_pool.req_to_token

        # ----------------------------------------------------------------
        # Fast path: no masks → just do one forward (prompt-caching pass)
        # ----------------------------------------------------------------
        mask_index_all = forward_batch.input_ids == self.mask_id
        if not mask_index_all.any():
            if forward_batch.input_ids.numel() == 0:
                empty_logits = LogitsProcessorOutput(
                    next_token_logits=None, full_logits=None
                )
                return empty_logits, [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        # Save state from the original DLLM_EXTEND batch
        original_out_cache_loc = forward_batch.out_cache_loc.clone()
        prefix_lens_gpu = forward_batch.extend_prefix_lens.clone()
        prefix_lens_cpu = prefix_lens_gpu.cpu()
        # Save fields we'll clobber during TARGET_VERIFY
        saved_global_num_tokens_cpu = forward_batch.global_num_tokens_cpu

        # ================================================================
        # STEP 1: Initial draft with DLLM_EXTEND (bidirectional)
        # ================================================================
        block_forward_passes = 1
        self._stats_forward_passes += 1
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output = out.logits_output

        # Extract draft tokens (exclude mask_id from argmax)
        full_logits = logits_output.full_logits  # [bs * block_size, vocab]
        logits_for_argmax = full_logits.clone()
        logits_for_argmax[:, self.mask_id] = float("-inf")
        draft_probs = F.softmax(full_logits, dim=-1)
        draft_tokens = torch.argmax(logits_for_argmax, dim=-1).to(torch.int32)
        draft_token_probs = draft_probs.gather(
            -1, draft_tokens.long().unsqueeze(-1)
        ).squeeze(-1)

        # Free bidirectional KV (not reusable as causal)
        allocator.free(original_out_cache_loc)

        # Current prefix state (per-request)
        cur_prefix_gpu = prefix_lens_gpu.clone()
        cur_prefix_cpu = prefix_lens_cpu.clone()

        # ================================================================
        # STEP 2: Quadratic decode loop
        # ================================================================
        all_committed: List[List[int]] = [[] for _ in range(bs)]
        final_logits_output = logits_output
        can_run_graph = False
        found_eos = [False] * bs

        for _iteration in range(self.max_quad_iters):
            # Check if all requests have committed enough tokens
            if all(len(c) >= block_size for c in all_committed):
                break

            # Build B tokens, positions, custom_mask using pre-allocated buffers
            draft_token_input, positions, custom_mask = (
                build_tidar_decode_inputs_pooled(
                    seq_lens_cpu=cur_prefix_cpu,
                    cur_prefix_gpu=cur_prefix_gpu,
                    block_size=block_size,
                    B=B,
                    mask_id=self.mask_id,
                    prev_draft_tokens=draft_tokens,
                    cached_offsets=self._cached_offsets,
                    cached_draft_mask=self._cached_draft_mask,
                    positions_out=self._pool_positions,
                    draft_tokens_out=self._pool_draft_tokens,
                    custom_mask_out=self._pool_custom_mask,
                )
            )

            # Allocate B * bs KV slots
            out_cache_loc = allocator.alloc(B * bs)
            if out_cache_loc is None:
                logger.warning("TiDAR: OOM allocating %d KV slots, breaking early", B * bs)
                break

            # Map new KV slots to req_to_token_pool
            assign_req_to_token_pool_func(
                forward_batch.req_pool_indices,
                req_to_token,
                cur_prefix_gpu,
                cur_prefix_gpu + B,
                out_cache_loc,
                bs,
            )

            # Build TiDARInput
            spec_info = TiDARInput(
                draft_token=draft_token_input,
                positions=positions,
                custom_mask=custom_mask,
                num_queries=B,
            )
            spec_info.seq_lens_cpu = cur_prefix_cpu.clone()
            spec_info.seq_lens_sum = int(cur_prefix_cpu.sum().item())

            # Switch forward_batch to TARGET_VERIFY
            forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
            forward_batch.input_ids = draft_token_input
            forward_batch.positions = positions
            forward_batch.out_cache_loc = out_cache_loc
            forward_batch.spec_info = spec_info
            forward_batch.seq_lens = cur_prefix_gpu.clone()
            forward_batch.seq_lens_cpu = cur_prefix_cpu.clone()
            forward_batch.seq_lens_sum = int(cur_prefix_cpu.sum().item())
            # Prevent post_forward_mlp_sync from touching next_token_logits=None
            forward_batch.global_num_tokens_cpu = None

            # Forward
            block_forward_passes += 1
            self._stats_forward_passes += 1
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            if out.can_run_graph:
                self._stats_cg_forwards += 1
            else:
                self._stats_eager_forwards += 1
            final_logits_output = out.logits_output

            # Get logits: [bs * B, vocab] → [bs, block_size+1, block_size, vocab]
            logits = final_logits_output.full_logits[: bs * B]
            logits = logits.view(bs, block_size + 1, block_size, -1)

            # Process each request
            prev_draft = draft_tokens.view(bs, block_size)
            prev_probs = draft_token_probs.view(bs, block_size)
            vocab_size = logits.shape[-1]

            # --- Logits mixing (matching HF self_spec_generate) ---
            # logits shape: [bs, block_size+1, block_size, vocab]
            # [:, 0, :, :] = AR logits (causal), [:, j+1, :, :] = diffusion block j
            ar_logits_all = logits[:, 0]     # [bs, block_size, vocab]
            diff_diag_all = logits[:, 1:, 0] # [bs, block_size, vocab] (position 0 of each diff block)

            if self.ar_mix_weight is None:
                # HF default: use pure AR logits for verification
                mixed_all = ar_logits_all
            else:
                mixed_all = (
                    ar_logits_all * self.ar_mix_weight
                    + diff_diag_all * (1 - self.ar_mix_weight)
                )

            # --- Greedy verification (matching HF exactly) ---
            # HF: useful_token_pred[:, accept_cnt-1, 0] != draft_input_ids[:, accept_cnt]
            # In our layout: target_argmax[b, k] is the greedy prediction at AR position k
            # Accept token 0 unconditionally. For token i (i>=1), check:
            #   argmax(mixed[b, i-1]) == prev_draft[b, i]
            remaining_t = self._pool_remaining[:bs]
            for b in range(bs):
                r = block_size - len(all_committed[b])
                if found_eos[b]:
                    r = 0
                remaining_t[b] = r

            accept_cnt_t = self._pool_accept_cnt[:bs]
            target_argmax = torch.argmax(mixed_all, dim=-1)  # [bs, block_size]

            for b in range(bs):
                r = int(remaining_t[b].item())
                if r <= 0:
                    accept_cnt_t[b] = 0
                    continue
                n_verify = min(block_size, r)
                # Token 0 always accepted; check tokens 1..n_verify-1
                matches = target_argmax[b, :n_verify - 1] == prev_draft[b, 1:n_verify]
                if not matches.all():
                    first_mm = (~matches).nonzero(as_tuple=True)[0][0].item()
                    accept_cnt_t[b] = 1 + first_mm
                else:
                    accept_cnt_t[b] = n_verify

            # --- One CPU sync to read accept counts ---
            accept_cnts = accept_cnt_t.tolist()

            # --- Per-request post-processing (KV management + draft update) ---
            for b in range(bs):
                accept_cnt = accept_cnts[b]
                if accept_cnt <= 0:
                    # Free all B KV slots for this request
                    b_start = b * B
                    allocator.free(out_cache_loc[b_start : b_start + B])
                    continue

                select_draft_idx = min(accept_cnt, block_size)

                # Collect accepted tokens
                accepted = prev_draft[b, :accept_cnt].tolist()

                # Check for EOS in accepted tokens
                if eos_id is not None and eos_id in accepted:
                    eos_pos = accepted.index(eos_id)
                    accepted = accepted[: eos_pos + 1]
                    accept_cnt = len(accepted)
                    found_eos[b] = True

                all_committed[b].extend(accepted)

                # Free rejected + diffusion KV, keep accepted AR KV
                b_start = b * B
                if accept_cnt < B:
                    allocator.free(
                        out_cache_loc[b_start + accept_cnt : b_start + B]
                    )

                # Update prefix for this request
                cur_prefix_gpu[b] += accept_cnt
                cur_prefix_cpu[b] += accept_cnt

                # New draft for next iteration: selective softmax on chosen block only
                if select_draft_idx < block_size + 1:
                    block_logits = logits[b, select_draft_idx]  # [block_size, vocab]
                    block_probs = F.softmax(block_logits, dim=-1)  # [block_size, vocab]
                    new_draft = torch.argmax(block_probs, dim=-1)  # [block_size]
                    new_draft_probs = block_probs.gather(
                        -1, new_draft.long().unsqueeze(-1)
                    ).squeeze(-1)  # [block_size]
                    draft_tokens = draft_tokens.view(bs, block_size)
                    draft_token_probs = draft_token_probs.view(bs, block_size)
                    draft_tokens[b] = new_draft.to(torch.int32)
                    draft_token_probs[b] = new_draft_probs
                    draft_tokens = draft_tokens.view(-1)
                    draft_token_probs = draft_token_probs.view(-1)

        # ================================================================
        # Restore forward_batch state
        # ================================================================
        forward_batch.forward_mode = ForwardMode.DLLM_EXTEND
        forward_batch.spec_info = None
        forward_batch.global_num_tokens_cpu = saved_global_num_tokens_cpu

        # ================================================================
        # Build output token lists (one tensor per request)
        # ================================================================
        next_token_ids_list = []
        for b in range(bs):
            committed = all_committed[b]
            if len(committed) < block_size and not found_eos[b]:
                logger.warning(
                    "TiDAR: request %d has only %d/%d tokens after %d iters "
                    "(increase max_quad_iters)",
                    b, len(committed), block_size, self.max_quad_iters,
                )
            next_token_ids_list.append(
                torch.tensor(committed, dtype=torch.long, device=device)
            )

        # ================================================================
        # Write per-request efficiency stats
        # ================================================================
        if self._stats_file:
            import json as _json

            with open(self._stats_file, "a") as _sf:
                for b in range(bs):
                    tokens = len(all_committed[b])
                    self._stats_tokens_generated += tokens
                    # Forward passes for THIS block only
                    tpfp = (
                        tokens / block_forward_passes
                        if block_forward_passes > 0
                        else 0.0
                    )
                    _sf.write(
                        _json.dumps(
                            {
                                "forward_passes": block_forward_passes,
                                "tokens": tokens,
                                "tokens_per_fp": round(tpfp, 4),
                            }
                        )
                        + "\n"
                    )

        logger.info(
            "TiDAR block done: %d forward passes (total=%d, cg=%d, eager=%d), committed=%s",
            block_forward_passes,
            self._stats_forward_passes,
            self._stats_cg_forwards,
            self._stats_eager_forwards,
            [len(c) for c in all_committed],
        )
        return final_logits_output, next_token_ids_list, can_run_graph


Algorithm = TiDAR
