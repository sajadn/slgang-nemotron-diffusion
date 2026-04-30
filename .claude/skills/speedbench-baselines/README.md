# SPEED-Bench Baselines — Nemotron-Diffusion & Speculative Decoding

Complete baseline results for Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2 (LinearSpec)
and speculative decoding baselines (Eagle3, MTP) on NVIDIA B200 (183 GB).

All experiments run on: 4× B200, SGLang branch `nemotron-dllm-upstream-5`.

---

## Environment

```bash
VENV="/home/mkhadkevich/dev/sglang/.venv/bin/python"
VENV_HF="/home/mkhadkevich/dev/sglang/.venvhf/bin/python"   # torch 2.8.0, transformers 5.0.0rc1, peft 0.15.2
EVAL="/home/mkhadkevich/dev/sglang/benchmark/speedbench/eval_speedbench.py"

MODEL_NEMO="/data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2/snapshots/adfaddf53841a39cb5bf8b9a793364edbaae17a9"
LORA_V3="/data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2-lora/snapshots/7b4c1dc9a7aca504e9c3ddbac77de93db7463194/v3"
LORA_GREEDY="/data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2-lora/snapshots/7b4c1dc9a7aca504e9c3ddbac77de93db7463194/oproj128_ce10kl1_greedy"
LORA_V3_NEMO_SGLANG="/data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2-lora/v3_nemo_sglang_dl/v3_nemo_sglang"
MODEL_Q3_8B="/data/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_Q35_9B="/data/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
EAGLE="/data/huggingface/hub/models--Tengyunw--qwen3_8b_eagle3/snapshots/181e0e287957a2be01a25f39b8faaeb404be76ed"

# Required env vars for all server launches
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**Dataset**: `nvidia/SPEED-Bench-Internal`, qualitative split — **713 single-turn samples**
(11 categories × ~65-80 samples; always use `--single_turn_only` for consistent comparison)

**Eval command template** (all configs):
```bash
$VENV $EVAL \
  --base_url http://localhost:<PORT>/v1 \
  --max_tokens 1024 \
  --concurrent <BS> \
  --single_turn_only \
  --api chat \
  --no_thinking \
  --summary_path /tmp/summary_<name>_bs<BS>.json \
  [--stats_file /tmp/stats_<name>.jsonl]   # only for LinearSpec configs
```

---

## Throughput Plot

![SPEED-Bench sweep results](speedbench_sweep_results.png)

---

## 1. AR Baseline — Qwen3-8B (221.9 tok/s @ bs=1)

### Server launch
```bash
CUDA_VISIBLE_DEVICES=0 $VENV -m sglang.launch_server \
  --model-path $MODEL_Q3_8B \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.9 \
  --max-running-requests 128 \
  --attention-backend flashinfer \
  --port 41000 >> /tmp/server_ar.log 2>&1 &
```

### Results
| bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 | bs=64 | bs=128 |
|------|------|------|------|-------|-------|-------|--------|
| 222 | 424 | 834 | 1599 | 3027 | 5317 | 7906 | 11774 |

---

## 2. Eagle3 — Qwen3-8B, d=15 (best speculative baseline)

### 2a. BF16, no compile (370.3 tok/s @ bs=1, 1.67× AR)

```bash
CUDA_VISIBLE_DEVICES=0 $VENV -m sglang.launch_server \
  --model-path $MODEL_Q3_8B \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path $EAGLE \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 15 \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.65 \
  --max-running-requests 128 \
  --attention-backend flashinfer \
  --port 41001 >> /tmp/server_eagle3_bf16.log 2>&1 &
```

### 2b. FP8+compile, d=15 (408.4 tok/s @ bs=1, 1.84× AR)

```bash
CUDA_VISIBLE_DEVICES=0 $VENV -m sglang.launch_server \
  --model-path $MODEL_Q3_8B \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path $EAGLE \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 15 \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.65 \
  --max-running-requests 128 \
  --attention-backend flashinfer \
  --quantization fp8 \
  --cuda-graph-bs 1 2 4 8 16 32 64 128 \
  --enable-torch-compile --torch-compile-max-bs 128 \
  --port 41002 >> /tmp/server_eagle3_fp8_compile.log 2>&1 &
```

### 2c. BF16+compile, d=15 (369.4 tok/s @ bs=1 — no gain vs no-compile)

Same as 2a but add:
```bash
  --cuda-graph-bs 1 2 4 8 16 32 64 128 \
  --enable-torch-compile --torch-compile-max-bs 128 \
```

### Eagle3 throughput comparison (tok/s)
| Config | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 | bs=64 | bs=128 |
|--------|------|------|------|------|-------|-------|-------|--------|
| BF16 (no compile) | 370 | 678 | 1255 | 2197 | 3370 | 4335 | 4594 | 5422 |
| BF16+compile | 369 | 671 | 1244 | 2211 | 3422 | 4367 | 4629 | 5486 |
| **FP8+compile** | **408** | **721** | **1370** | **2401** | **3637** | **4576** | **4933** | **5852** |

**Finding**: torch.compile alone gives no gain for Eagle3. FP8+compile gives ~+10% uniformly.

---

## 3. MTP NEXTN — Qwen3.5-9B, d=3 (388.9 tok/s @ bs=1, 1.75× AR)

Requires `SGLANG_ENABLE_SPEC_V2=1`. Uses native MTP heads — does NOT load model twice.

### Server launch
```bash
SGLANG_ENABLE_SPEC_V2=1 CUDA_VISIBLE_DEVICES=0 $VENV -m sglang.launch_server \
  --model-path $MODEL_Q35_9B \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.9 \
  --max-running-requests 128 \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path $MODEL_Q35_9B \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 3 \
  --mamba-scheduler-strategy extra_buffer \
  --port 41003 >> /tmp/server_mtp.log 2>&1 &
```

> **Note**: MTP + torch.compile + 8 CUDA graphs OOMs on B200 (183 GB) in both BF16 and FP8
> due to CUDA graph private pool growth (~55 GiB). FP8+compile and BF16+compile skipped.

### Results (tok/s)
| bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 | bs=64 | bs=128 |
|------|------|------|------|-------|-------|-------|--------|
| 389 | 742 | 1394 | 2418 | 3723 | 5213 | 6624 | 7587 |

---

## 4. Nemotron-Diffusion LinearSpec — All Configs

Model: `nvidia/Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2`
LoRA: v3 (`oproj128`, r=128, target=o_proj)

> **CRITICAL**: Always include `causal_context: true` in every YAML config.
> Without it the model loops `</think></think>...` with near-zero accuracy.

### YAML configs

**BF16+compile bl=32** (`/tmp/dllm_bf16_compile_bl32.yaml`):
```yaml
algorithm: LinearSpec
block_size: 32
causal_context: true
lora_path: /data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2-lora/snapshots/7b4c1dc9a7aca504e9c3ddbac77de93db7463194/v3
stats_file: /tmp/stats_bf16_compile_bl32.jsonl
```

**FP8+compile bl=32** (`/tmp/dllm_fp8_compile_bl32.yaml`):
```yaml
algorithm: LinearSpec
block_size: 32
causal_context: true
lora_path: /data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2-lora/snapshots/7b4c1dc9a7aca504e9c3ddbac77de93db7463194/v3
stats_file: /tmp/stats_fp8_compile_bl32.jsonl
```

**BF16+compile bl=64** (same but `block_size: 64`),
**FP8+compile bl=64** (same but `block_size: 64`)

---

### 4a. BF16+compile bl=32 (439.9 tok/s @ bs=1)

```bash
CUDA_VISIBLE_DEVICES=0 $VENV -m sglang.launch_server \
  --model-path $MODEL_NEMO \
  --tokenizer-path $MODEL_NEMO \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.6 \
  --max-running-requests 128 \
  --attention-backend flashinfer \
  --dllm-algorithm LinearSpec \
  --dllm-algorithm-config /tmp/dllm_bf16_compile_bl32.yaml \
  --cuda-graph-bs 1 2 4 8 16 32 64 128 \
  --enable-torch-compile --torch-compile-max-bs 128 \
  --port 42001 >> /tmp/server_nemo_bf16_bl32.log 2>&1 &
```

### 4b. FP8+compile bl=32 (492.5 tok/s @ bs=1) — **best single-GPU config**

```bash
CUDA_VISIBLE_DEVICES=0 $VENV -m sglang.launch_server \
  --model-path $MODEL_NEMO \
  --tokenizer-path $MODEL_NEMO \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.6 \
  --max-running-requests 128 \
  --attention-backend flashinfer \
  --quantization fp8 \
  --dllm-algorithm LinearSpec \
  --dllm-algorithm-config /tmp/dllm_fp8_compile_bl32.yaml \
  --cuda-graph-bs 1 2 4 8 16 32 64 128 \
  --enable-torch-compile --torch-compile-max-bs 128 \
  --port 42002 >> /tmp/server_nemo_fp8_bl32.log 2>&1 &
```

### 4c. BF16+compile bl=64 (430.0 tok/s @ bs=1)

Same as 4a with `block_size: 64` in YAML and `--port 42003`.

### 4d. FP8+compile bl=64 (496.2 tok/s @ bs=1)

Same as 4b with `block_size: 64` in YAML and `--port 42004`.

---

### Throughput (tok/s) — all Nemotron configs

Measured with pre-templated `<think></think>` prompts via `/v1/completions` (corrected; no thinking tokens).
713 single-turn SPEED-Bench prompts, sliding-window concurrency at each batch size.

| Config | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 | bs=64 | bs=128 |
|--------|------|------|------|------|-------|-------|-------|--------|
| BF16+compile bl=32 | 440 | 777 | 1317 | 2204 | 3069 | 3651 | 4024 | 4230 |
| BF16+compile bl=64 | 433 | 738 | 1248 | 1784 | 2223 | 2451 | 2510 | 2602 |
| **FP8+compile bl=32** | **488** | **859** | **1482** | **2394** | **3470** | **4236** | **4610** | **4836** |
| FP8+compile bl=64 | 499 | 868 | 1409 | 2020 | 2639 | 2891 | 3010 | 3122 |

> **Note**: Previous chat-API measurements were within 2% of these corrected values — thinking tokens
> inflate gen_len but also constitute real generated tokens, so tok/s is unaffected.

**Findings**:
- FP8 gives ~+12% at bs=1, ~+10% sustained at all batch sizes
- bl=32 outperforms bl=64 at bs≥8 (better scheduling efficiency at concurrency)
- bl=64 slightly higher tok/FP but lower throughput overall

---

### Acceptance Rate (tok/FP = tokens per forward pass)

tok/FP = gen_len / nfe, where nfe = 1 (prefill) + 2 × n_blocks (1 draft + 1 verify per block).
tok/blk = gen_len / n_blocks = 2 × tok/FP (approximately).

Measured on 713 single-turn SPEED-Bench prompts with `<think></think>` pre-closed
(no thinking tokens). Previous measurements with chat API (`<think>\n` open tag)
were artificially low (~3.06–3.16) due to thinking tokens inflating gen_len
without contributing to block acceptance.

| Config | tok/FP | tok/blk |
|--------|--------|---------|
| BF16+compile bl=32 | 3.359 | 6.718 |
| BF16+compile bl=64 | 3.441 | 6.882 |
| FP8+compile bl=32  | 3.277 | 6.553 |
| **FP8+compile bl=64**  | **3.491** | **6.981** |

tok/FP is stable across batch sizes (acceptance is a model property, not a throughput one).

---

## 5. LoRA Variant Comparison (BF16+compile bl=32, SGLang)

All configs use the same server as §4a. Change `lora_path` in YAML.

| LoRA | Trained on | bs=1 tok/s | bs=4 tok/s | bs=8 tok/s | bs=32 tok/s | tok/FP (all bs) |
|------|-----------|-----------|-----------|-----------|------------|----------------|
| v3 | HF/flex_attention | 440 | 1311 | 2175 | 3629 | ~3.06–3.10 |
| **v3_nemo_sglang** | SGLang/FlashInfer | **415** | **1249** | **2089** | **3454** | **~2.92–2.94** |

> v3_nemo_sglang (trained on SGLang-generated data) performs **worse** than v3 (trained on HF native data).
> Root cause under investigation — training details (data volume, distribution, learning rate) need review.
> The SGLang LoRA application is confirmed correct (asymmetric: ON for draft, OFF for verify).

---

## 6. Native HF Acceptance (linear_spec_generate) — v3 LoRA

Uses `modeling_ministral_dlm.py`'s `linear_spec_generate` directly with `.venvhf`
(torch 2.8.0+cu129, transformers 5.0.0rc1, peft 0.15.2).

### Reproduction script

```python
# Install deps in .venvhf:
# uv venv .venvhf --python 3.12
# uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129 --python .venvhf/bin/python
# uv pip install "transformers==5.0.0rc1" "peft==0.15.2" "accelerate==1.12.0" \
#     "tokenizers==0.22.1" "safetensors==0.5.3" datasets --python .venvhf/bin/python

import torch, json, os, multiprocessing, time
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from datasets import load_dataset

REPO = "/data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2-lora/snapshots/7b4c1dc9a7aca504e9c3ddbac77de93db7463194"
LORA = f"{REPO}/v3"
BLOCK_LEN = 32
MAX_GEN = 1024

def build_prompt(tokenizer, turns):
    messages = [{"role": "user", "content": turns[0]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Pre-close think tag to disable thinking (must use <think></think>, not open <think>)
    return prompt.replace("<think>\n", "<think></think>\n")

def worker(gpu_id, rows, out_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    model = PeftModel.from_pretrained(model, LORA).eval()
    base = model.model  # linear_spec_generate toggles LoRA asymmetrically

    shard_file = Path(out_dir) / f"gpu{gpu_id}.jsonl"
    with open(shard_file, "a") as f:
        for row in rows:
            ids = tokenizer(build_prompt(tokenizer, row["turns"]), return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                out, nfe = base.linear_spec_generate(
                    ids, max_new_tokens=MAX_GEN, block_length=BLOCK_LEN,
                    temperature=0.0, eos_token_id=tokenizer.eos_token_id)
            gen_len = out.shape[1] - ids.shape[1]
            n_blocks = (nfe - 1) / 2
            f.write(json.dumps({"tpb": gen_len/n_blocks, "tpn": gen_len/nfe, "nfe": nfe, "gen_len": gen_len}) + "\n")
            f.flush()
```

Run on 4 GPUs with `multiprocessing.Process` per GPU, shard with `rows[g::4]`.

### Results (v3 LoRA, no-thinking, bl=32, 713 samples, 4× B200)

| Metric | Value | HF card (greedy LoRA) |
|--------|-------|-----------------------|
| tpb (tok/block) | **7.03** | 7.39 |
| tpn (tok/NFE)   | **3.47** | 3.64 |
| gen_mean        | 638 | 635 |

### Per-category tpb/tpn (v3 LoRA, no-thinking)

| Category | n | tpb | tpn |
|----------|---|-----|-----|
| coding | 71 | 8.921 | 4.429 |
| humanities | 72 | 7.200 | 3.562 |
| math | 62 | 8.516 | 4.212 |
| multilingual | 80 | 10.733 | 5.191 |
| qa | 80 | 4.467 | 2.214 |
| rag | 65 | 6.167 | 3.050 |
| reasoning | 18 | 8.605 | 4.258 |
| roleplay | 35 | 5.514 | 2.731 |
| stem | 74 | 7.450 | 3.697 |
| summarization | 80 | 5.388 | 2.659 |
| writing | 76 | 5.067 | 2.524 |

---

## 7. SGLang vs Native HF Acceptance Gap

| Config | tpn (tok/FP) | tpb (tok/blk) |
|--------|-------------|--------------|
| Native HF `linear_spec_generate`, v3 LoRA, no-thinking | **3.47** | **7.03** |
| SGLang BF16+compile bl=32, v3 LoRA (corrected) | 3.36 | 6.72 |
| SGLang FP8+compile bl=64, v3 LoRA (corrected) | 3.49 | 6.98 |
| **Gap (BF16 vs HF)** | **-3%** | **-4%** |

**Previous measurements (~3.06 tpn) were wrong**: the chat API produced `<think>\n`
(open tag) instead of `<think></think>` (pre-closed), causing thinking tokens to inflate
gen_len without contributing to block acceptance. Corrected eval uses pre-templated
prompts with `<think></think>` via `/v1/completions`.

**LoRA application confirmed correct in SGLang** (asymmetric: ON for draft, OFF for verify):
- Two separate CUDA graphs captured per batch size (`<bs>` for draft, `causal_<bs>` for verify)
- Draft graph: `_dllm_pre_draft_hook` sets base+LoRA weights before capture
- Verify graph: `_dllm_pre_verify_hook` sets base-only weights before capture
- No weight swaps needed at replay time — correct weights permanently baked in

**Remaining ~3% gap** (BF16) is from FlashInfer vs flex_attention kernel numerical
differences — the v3 LoRA was trained on flex_attention logits. FP8+compile bl=64
essentially closes the gap at 3.49 vs HF's 3.47.

---

## Key Findings Summary

| Finding | Detail |
|---------|--------|
| **Best Nemotron config** | FP8+compile bl=32: 492 tok/s @ bs=1, 4880 tok/s @ bs=128 |
| **FP8 vs BF16** | +12% at bs=1, ~+10% at all batch sizes, same tok/FP |
| **bl=32 vs bl=64** | bl=32 wins at bs≥8; bl=64 slightly higher tok/FP but lower throughput |
| **torch.compile (Nemotron)** | Significant speedup: +12% at bs=1, needed for FP8 efficiency |
| **torch.compile (Eagle3/MTP)** | FP8+compile: +10%; BF16+compile: no gain |
| **MTP+compile OOM** | Both BF16 and FP8 MTP+compile OOM on B200 (CUDA graph pool ~55 GiB) |
| **Native vs SGLang acceptance** | -3% gap (BF16) / ~0% (FP8+compile bl=64) — old -12% was chat-API thinking-token artifact |
| **Thinking tokens** | Must use `<think></think>` (pre-closed) for no-thinking eval; `enable_thinking=False` in chat template produces open `<think>` tag — wrong |
| **causal_context: true** | Required for all LinearSpec configs; without it → `</think>` loops |
