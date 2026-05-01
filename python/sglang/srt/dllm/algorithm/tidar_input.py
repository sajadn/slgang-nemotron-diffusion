"""
TiDARInput — SpecInput subclass for TiDAR self-speculation.

Reuses EAGLE_VERIFY SpecInputType so FlashInfer's prefill wrapper accepts the
custom_mask path (generate_attn_arg_prefill → FlashInferIndicesUpdaterPrefill).

Adapted from /tmp/tidar_ref/tidar_info.py for use within the DLLM algorithm framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass
class TiDARInput(SpecInput):
    draft_token: Optional[torch.Tensor]
    positions: Optional[torch.Tensor]
    custom_mask: Optional[torch.Tensor]
    num_queries: int
    draft_token_num: Optional[int] = None
    seq_lens_cpu: Optional[torch.Tensor] = None
    seq_lens_sum: Optional[int] = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        if self.draft_token_num is None:
            self.draft_token_num = self.num_queries
        # TiDAR uses full_logits, not hidden states — no special capture mode needed
        self.capture_hidden_mode = CaptureHiddenMode.NULL

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_queries, self.num_queries

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        # qo_indptr: grouped by request, each with num_queries
        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.num_queries,
            step=self.num_queries,
            dtype=torch.int32,
            device=device,
        )

        # kv_indptr: cumulative KV lengths (prefix + num_queries draft tokens)
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.num_queries
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # Build kv_indices via Triton kernel
        total_kv_len = int(cum_kv_seq_len[-1].item())
        kv_indices = torch.empty(total_kv_len, dtype=torch.int32, device=device)
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.shape[1],
        )
        # Pad custom_mask for CG bs padding (follows EAGLE pattern).
        # When CG runner pads raw_bs to next captured bs, extra requests get
        # zero-mask (no attention) — their outputs are discarded anyway.
        mask_numel = sum(
            self.num_queries * int(paged_kernel_lens[b].item())
            for b in range(bs)
        )
        custom_mask = self.custom_mask
        if custom_mask.numel() < mask_numel:
            custom_mask = torch.cat(
                [
                    custom_mask,
                    torch.zeros(
                        mask_numel - custom_mask.numel(),
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )
        return kv_indices, cum_kv_seq_len, qo_indptr, custom_mask
