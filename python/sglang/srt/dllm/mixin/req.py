from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            # Always run a causal/bidirectional EXTEND pass on the prompt first
            # to cache prompt KV before the first DLLM_EXTEND denoising pass.
            self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            return

        # Check the LATEST block (not the first block which may already be
        # denoised).  For block k>0, fill_ids = origin + output_k-1 + masks_k
        # so the last block_size tokens are the masks to denoise.
        latest_block_start = len(self.fill_ids) - self.dllm_config.block_size
        input_block = self.fill_ids[latest_block_start:]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if not self.fill_ids:
            # First block: block offset = prompt length.
            self.dllm_block_offset = len(self.origin_input_ids)
        else:
            # Output-based offset: works for both full-block (FastDiffuser) and
            # partial-block (LinearSpec) acceptance.
            self.dllm_block_offset = len(self.origin_input_ids) + len(self.output_ids)
        self.fill_ids = (
            self.origin_input_ids
            + self.output_ids
            + [self.dllm_config.mask_id] * self.dllm_config.block_size
        )

    def init_prompt_cache_input(self: Req):
        """Prepare for causal EXTEND pass to cache prompt KV.

        Sets fill_ids = origin_input_ids (no masks) and prefix_indices = empty
        so alloc_for_extend allocates fresh KV for the whole prompt.
        """
        self.fill_ids = list(self.origin_input_ids)
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.set_extend_input_len(len(self.fill_ids))

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        # LinearSpec produces partial blocks, so prefix_len won't always be
        # block-aligned.  Only assert for the initial prompt (before any output).
        if not self.output_ids:
            assert (
                prefix_len % self.dllm_config.block_size == 0
            ), f"Unexpected prefix len: {prefix_len}"
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
