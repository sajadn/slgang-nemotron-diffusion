"""
AR algorithm for SGLang DLLM — pure autoregressive mode via the causal verify pass.

Skips the draft pass entirely; runs exactly 1 causal forward pass per token.
Equivalent to standard AR generation using the diffusion model's causal attention path.

Usage:
  dllm_algorithm: AR
  dllm_algorithm_config:
    causal_context: true   # REQUIRED — always set this
    stats_file: null
"""

import json as _json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class AR(DllmAlgorithm):
    """
    Pure AR mode: skip draft, run only the causal verify pass.
    Emits exactly 1 token per forward pass — true autoregressive generation.
    block_size is forced to 1 regardless of config.
    """

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        # Force block_size=1 for true AR (1 token per FP)
        self.block_size = 1
        self.causal_context: bool = config.causal_context
        self._seed_tokens: Dict[str, int] = {}
        self._eos_token_id: Optional[int] = None

        cfg = config.algorithm_config
        self._stats_file: Optional[str] = cfg.get("stats_file", None)
        self._stats_forward_passes: int = 0

        logger.info("AR: causal_context=%s  block_size=1 (forced)", self.causal_context)

    def _get_eos_id(self, model_runner: ModelRunner) -> Optional[int]:
        if self._eos_token_id is None:
            try:
                self._eos_token_id = model_runner.tokenizer.eos_token_id
            except Exception:
                pass
        return self._eos_token_id

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        eos_id = self._get_eos_id(model_runner)

        # ----------------------------------------------------------------
        # Prefill path: no masks → run one forward, extract seed tokens
        # ----------------------------------------------------------------
        mask_index_all = forward_batch.input_ids == self.mask_id
        if not mask_index_all.any():
            if forward_batch.input_ids.numel() == 0:
                return LogitsProcessorOutput(next_token_logits=None, full_logits=None), [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output = out.logits_output
            if logits_output.next_token_logits is not None:
                seed_logits = logits_output.next_token_logits.clone()
                seed_logits[:, self.mask_id] = -np.inf
                seeds = torch.argmax(seed_logits, dim=-1)
                for b_idx, rid in enumerate(forward_batch.rids):
                    self._seed_tokens[rid] = int(seeds[b_idx].item())
            return logits_output, [], out.can_run_graph

        # ----------------------------------------------------------------
        # Decode path: inject seed, run causal verify, output 1 token
        # ----------------------------------------------------------------
        rids = forward_batch.rids[:batch_size]

        # Inject seed into the single mask position (block_size=1)
        for b in range(batch_size):
            rid = rids[b]
            if rid in self._seed_tokens:
                forward_batch.input_ids[b] = self._seed_tokens[rid]

        # DECODE pass — use the optimized single-token decode attention kernel
        # instead of the DLLM_EXTEND prefill kernel (~50% throughput improvement).
        # dllm_ar_mode=True routes to the DECODE CUDA graph captured at startup.
        orig_mode = forward_batch.forward_mode
        forward_batch.forward_mode = ForwardMode.DECODE
        forward_batch.dllm_ar_mode = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.forward_mode = orig_mode
        forward_batch.dllm_ar_mode = False
        self._stats_forward_passes += 1

        # In DECODE mode the logits processor returns next_token_logits [bs, V].
        logits_out = out.logits_output
        verify_logits = (
            logits_out.next_token_logits
            if logits_out.next_token_logits is not None
            else logits_out.full_logits
        )
        verify_logits[:, self.mask_id] = -1e9
        ar_all = torch.argmax(verify_logits, dim=-1)  # [batch_size]

        # ----------------------------------------------------------------
        # Build output: 1 token per request
        # seed = the token we placed in input_ids (current output token)
        # ar_all[b] = causal prediction AFTER seed position (next seed)
        # ----------------------------------------------------------------
        next_token_ids_list: List[torch.Tensor] = []
        for b in range(batch_size):
            rid = rids[b]
            seed_val = int(forward_batch.input_ids[b].item())
            next_seed = int(ar_all[b].item())
            output_tokens = torch.tensor(
                [seed_val], dtype=torch.long, device=forward_batch.input_ids.device
            )

            if eos_id is not None and seed_val == eos_id:
                self._seed_tokens.pop(rid, None)
            else:
                self._seed_tokens[rid] = next_seed

            next_token_ids_list.append(output_tokens)

        # ----------------------------------------------------------------
        # Stats
        # ----------------------------------------------------------------
        if self._stats_file:
            entry = {
                "forward_passes": 1,
                "tokens": batch_size,
                "acceptance_rate": 1.0,
            }
            with open(self._stats_file, "a") as f:
                f.write(_json.dumps(entry) + "\n")

        return out.logits_output, next_token_ids_list, out.can_run_graph


Algorithm = AR
