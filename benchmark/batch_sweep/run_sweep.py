#!/usr/bin/env python3
"""
Batch-size throughput sweep across all baselines and Nemotron-Diffusion configs.

Usage:
  python run_sweep.py [--dry-run] [--configs ar mtp eagle nemotron_bf16_64 ...]

Output:
  /tmp/batch_sweep_results_v2.json   — raw results
  /tmp/batch_sweep_results_v2.png    — throughput plot
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
VENV        = "/home/mkhadkevich/dev/sglang/.venv/bin/python"
EVAL        = "/home/mkhadkevich/dev/sglang/benchmark/gsm8k/eval_sglang.py"
MODEL_NEMO  = "/data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2/snapshots/adfaddf53841a39cb5bf8b9a793364edbaae17a9"
LORA_V3     = "/data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-Instruct-v2-lora/snapshots/7b4c1dc9a7aca504e9c3ddbac77de93db7463194/v3"
MODEL_Q3    = "/data/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_Q35   = "/data/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_EAGLE = "/data/huggingface/hub/models--Tengyunw--qwen3_8b_eagle3/snapshots/181e0e287957a2be01a25f39b8faaeb404be76ed"
RESULTS_JSON = "/tmp/batch_sweep_results_v2.json"
RESULTS_PNG  = "/tmp/batch_sweep_results_v2.png"

BATCH_SIZES  = [1, 2, 4, 8, 16, 32, 64, 128]
NUM_SAMPLES  = 200   # GSM8K subset for speed
MAX_TOKENS   = 1024
WARMUP_TIMEOUT = 720  # seconds to wait for server ready

BASE_ENV = {
    **os.environ,
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

# ─── Config definitions ───────────────────────────────────────────────────────
CONFIGS = {
    "ar_qwen3_8b": {
        "label": "Qwen3-8B AR",
        "gpu": 0, "port": 41000,
        "prompt_style": "default",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_Q3,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--port", "41000",
        ],
        "env_extra": {},
        "is_dllm": False,
        "stats_file": None,
    },
    "mtp_qwen35_9b": {
        "label": "Qwen3.5-9B MTP d=3",
        "gpu": 1, "port": 41001,
        "prompt_style": "default",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_Q35,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--max-running-requests", "128",
            "--speculative-algorithm", "NEXTN",
            "--speculative-draft-model-path", MODEL_Q35,
            "--speculative-eagle-topk", "1",
            "--speculative-num-steps", "3",
            "--speculative-num-draft-tokens", "3",
            "--mamba-scheduler-strategy", "extra_buffer",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--port", "41001",
        ],
        "env_extra": {"SGLANG_ENABLE_SPEC_V2": "1"},
        "is_dllm": False,
        "stats_file": None,
    },
    "eagle3_qwen3_8b": {
        "label": "Qwen3-8B Eagle3 d=15",
        "gpu": 2, "port": 41002,
        "prompt_style": "default",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_Q3,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--max-running-requests", "128",
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", MODEL_EAGLE,
            "--speculative-num-steps", "4",
            "--speculative-eagle-topk", "4",
            "--speculative-num-draft-tokens", "15",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--port", "41002",
        ],
        "env_extra": {},
        "is_dllm": False,
        "stats_file": None,
    },
    "nemo_bf16_bl64": {
        "label": "Nemotron BF16 bl=64 v3LoRA",
        "gpu": 3, "port": 41003,
        "prompt_style": "v2",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/sweep_nemo_bf16_bl64.yaml",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--port", "41003",
        ],
        "env_extra": {},
        "is_dllm": True,
        "stats_file": "/tmp/stats_sweep_nemo_bf16_bl64.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 64, "causal_context": True,
                 "lora_path": LORA_V3, "stats_file": "/tmp/stats_sweep_nemo_bf16_bl64.jsonl"},
        "yaml_path": "/tmp/sweep_nemo_bf16_bl64.yaml",
    },
    "nemo_bf16_bl32": {
        "label": "Nemotron BF16 bl=32 v3LoRA",
        "gpu": 3, "port": 41003,
        "prompt_style": "v2",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/sweep_nemo_bf16_bl32.yaml",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--port", "41003",
        ],
        "env_extra": {},
        "is_dllm": True,
        "stats_file": "/tmp/stats_sweep_nemo_bf16_bl32.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 32, "causal_context": True,
                 "lora_path": LORA_V3, "stats_file": "/tmp/stats_sweep_nemo_bf16_bl32.jsonl"},
        "yaml_path": "/tmp/sweep_nemo_bf16_bl32.yaml",
    },
    "nemo_fp8_bl64": {
        "label": "Nemotron FP8 bl=64 v3LoRA",
        "gpu": 0, "port": 41000,
        "prompt_style": "v2",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.85",  # 0.9 OOMs on bs=128 CUDA graph capture with FP8+LoRA
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--quantization", "fp8",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/sweep_nemo_fp8_bl64.yaml",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--port", "41000",
        ],
        "env_extra": {},
        "is_dllm": True,
        "stats_file": "/tmp/stats_sweep_nemo_fp8_bl64.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 64, "causal_context": True,
                 "lora_path": LORA_V3, "stats_file": "/tmp/stats_sweep_nemo_fp8_bl64.jsonl"},
        "yaml_path": "/tmp/sweep_nemo_fp8_bl64.yaml",
    },
    "nemo_fp8_bl32": {
        "label": "Nemotron FP8 bl=32 v3LoRA",
        "gpu": 1, "port": 41001,
        "prompt_style": "v2",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--quantization", "fp8",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/sweep_nemo_fp8_bl32.yaml",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--port", "41001",
        ],
        "env_extra": {},
        "is_dllm": True,
        "stats_file": "/tmp/stats_sweep_nemo_fp8_bl32.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 32, "causal_context": True,
                 "lora_path": LORA_V3, "stats_file": "/tmp/stats_sweep_nemo_fp8_bl32.jsonl"},
        "yaml_path": "/tmp/sweep_nemo_fp8_bl32.yaml",
    },
    "nemo_fp8_compile_bl64": {
        "label": "Nemotron FP8+compile bl=64 v3LoRA",
        "gpu": 2, "port": 41002,
        "prompt_style": "v2",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.7",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--quantization", "fp8",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/sweep_nemo_fp8_compile_bl64.yaml",
            "--cuda-graph-bs", "1",
            "--enable-torch-compile", "--torch-compile-max-bs", "1",
            "--port", "41002",
        ],
        "env_extra": {},
        "is_dllm": True,
        "compile": True,
        "stats_file": "/tmp/stats_sweep_nemo_fp8_compile_bl64.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 64, "causal_context": True,
                 "lora_path": LORA_V3, "stats_file": "/tmp/stats_sweep_nemo_fp8_compile_bl64.jsonl"},
        "yaml_path": "/tmp/sweep_nemo_fp8_compile_bl64.yaml",
    },
    "nemo_fp8_compile_bl32": {
        "label": "Nemotron FP8+compile bl=32 v3LoRA",
        "gpu": 3, "port": 41003,
        "prompt_style": "v2",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.7",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--quantization", "fp8",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/sweep_nemo_fp8_compile_bl32.yaml",
            "--cuda-graph-bs", "1",
            "--enable-torch-compile", "--torch-compile-max-bs", "1",
            "--port", "41003",
        ],
        "env_extra": {},
        "is_dllm": True,
        "compile": True,
        "stats_file": "/tmp/stats_sweep_nemo_fp8_compile_bl32.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 32, "causal_context": True,
                 "lora_path": LORA_V3, "stats_file": "/tmp/stats_sweep_nemo_fp8_compile_bl32.jsonl"},
        "yaml_path": "/tmp/sweep_nemo_fp8_compile_bl32.yaml",
    },
}

# Parallel rounds: run these config groups simultaneously (one per GPU)
ROUNDS = [
    ["ar_qwen3_8b", "mtp_qwen35_9b", "eagle3_qwen3_8b", "nemo_bf16_bl64"],
    ["nemo_fp8_bl64", "nemo_fp8_bl32", "nemo_fp8_compile_bl64", "nemo_bf16_bl32"],
    ["nemo_fp8_compile_bl32"],
]

# Follow-up round: remap to non-conflicting GPUs/ports for reruns
FOLLOWUP_GPU_MAP = {
    "ar_qwen3_8b":    (0, 41000),
    "nemo_bf16_bl64": (1, 41001),
    "nemo_fp8_bl64":  (2, 41002),
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
import re
import urllib.request


def write_yaml(cfg):
    if "yaml" not in cfg:
        return
    import yaml as _yaml
    with open(cfg["yaml_path"], "w") as f:
        _yaml.dump(cfg["yaml"], f, default_flow_style=False)


def wait_for_server(port, timeout=WARMUP_TIMEOUT):
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(5)
    return False


def parse_eval_output(text):
    """Parse eval_sglang.py stdout for tok/s, acc, avg_len.

    Expected lines:
      Throughput: 123.4 tok/s
      Avg gen length: 200.5 tok/sample
      GSM8K Accuracy: 240/200 = 92.3%
    """
    result = {}
    # "  Throughput: 123.4 tok/s"
    m = re.search(r"Throughput:\s*([\d.]+)\s*tok/s", text)
    if m:
        result["tok_per_s"] = float(m.group(1))
    # "GSM8K Accuracy: 240/200 = 92.3%"
    m = re.search(r"Accuracy:.*?=\s*([\d.]+)%", text)
    if m:
        result["accuracy"] = float(m.group(1))
    # "  Avg gen length: 200.5 tok/sample"
    m = re.search(r"Avg gen length:\s*([\d.]+)", text)
    if m:
        result["avg_len"] = float(m.group(1))
    return result


def parse_stats_file(path):
    """Parse DLLM stats jsonl for tok/FP and acceptance rate."""
    if not path or not Path(path).exists():
        return {}
    data = [json.loads(l) for l in open(path) if l.strip()]
    if not data:
        return {}
    toks = sum(x["tokens"] for x in data)
    fps  = sum(x["forward_passes"] for x in data)
    acc_sum = sum(x.get("acceptance_rate", 0) * x["forward_passes"] for x in data)
    return {
        "tok_per_fp": toks / fps if fps else 0,
        "acceptance_rate": acc_sum / fps if fps else 0,
        "total_tokens": toks,
        "total_fps": fps,
    }


def run_eval_for_bs(port, concurrent, prompt_style, log_prefix, dry_run=False):
    """Run eval_sglang.py at given concurrency, return parsed metrics."""
    log_path = f"{log_prefix}_bs{concurrent}.log"
    cmd = [
        VENV, EVAL,
        "--benchmark", "gsm8k",
        "--base_url", f"http://localhost:{port}/v1",
        "--max_tokens", str(MAX_TOKENS),
        "--num_samples", str(NUM_SAMPLES),
        "--concurrent", str(concurrent),
        "--no_thinking",
        "--prompt_style", prompt_style,
        "--temperature", "0.0",
    ]
    print(f"    [eval] concurrent={concurrent} → {log_path}")
    if dry_run:
        return {"tok_per_s": 0, "accuracy": 0, "avg_len": 0}
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=BASE_ENV, stdout=f, stderr=f, timeout=1800)
    text = Path(log_path).read_text()
    return parse_eval_output(text)


def sweep_config(name, cfg, dry_run=False):
    """Start server, sweep all batch sizes, kill server, return results."""
    print(f"\n{'='*60}")
    print(f"Config: {cfg['label']} (GPU {cfg['gpu']}, port {cfg['port']})")
    print(f"{'='*60}")

    gpu_env = {**BASE_ENV, **cfg.get("env_extra", {}),
               "CUDA_VISIBLE_DEVICES": str(cfg["gpu"])}

    # Write YAML if needed
    write_yaml(cfg)

    # Clear stats file
    if cfg.get("stats_file"):
        Path(cfg["stats_file"]).write_text("")

    log_prefix = f"/tmp/sweep_{name}"

    if not dry_run:
        # Start server
        server_log = f"/tmp/sweep_{name}_server.log"
        print(f"  Starting server → {server_log}")
        server = subprocess.Popen(
            cfg["server_args"], env=gpu_env,
            stdout=open(server_log, "w"), stderr=subprocess.STDOUT
        )

        is_compile = cfg.get("compile", False)
        timeout = WARMUP_TIMEOUT + (300 if is_compile else 0)
        print(f"  Waiting for server (timeout={timeout}s)...")
        if not wait_for_server(cfg["port"], timeout):
            print(f"  ERROR: server timed out!")
            server.kill()
            return None
        print(f"  Server ready.")
    else:
        server = None

    config_results = {"label": cfg["label"], "batch_sizes": {}}

    bs_list = BATCH_SIZES
    # For compile configs: CUDA graph only at bs=1, higher bs runs in eager
    # Still sweep all batch sizes (server can handle them in eager)
    for bs in bs_list:
        # Clear stats before each BS run so we get per-BS stats
        if cfg.get("stats_file"):
            Path(cfg["stats_file"]).write_text("")

        metrics = run_eval_for_bs(cfg["port"], bs, cfg["prompt_style"],
                                  log_prefix, dry_run)
        metrics["tok_per_s_per_req"] = metrics.get("tok_per_s", 0) / bs

        # Read DLLM stats if available
        if cfg.get("stats_file") and not dry_run:
            dllm_stats = parse_stats_file(cfg["stats_file"])
            metrics.update(dllm_stats)

        config_results["batch_sizes"][bs] = metrics
        tok_s = metrics.get("tok_per_s", 0)
        tok_s_req = metrics.get("tok_per_s_per_req", 0)
        acc = metrics.get("accuracy", 0)
        tok_fp = metrics.get("tok_per_fp", "-")
        print(f"    bs={bs:3d}: {tok_s:7.1f} tok/s total | {tok_s_req:6.1f} tok/s/req "
              f"| acc={acc:.1f}% | tok/FP={tok_fp}")

    if server and not dry_run:
        print(f"  Killing server PID {server.pid}...")
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        time.sleep(10)  # let GPU memory free

    return config_results


def run_round(config_names, dry_run=False):
    """Run a group of configs in parallel (each on its own GPU)."""
    import threading

    round_results = {}
    threads = []
    lock = threading.Lock()

    def worker(name):
        result = sweep_config(name, CONFIGS[name], dry_run=dry_run)
        with lock:
            if result:
                round_results[name] = result

    for name in config_names:
        t = threading.Thread(target=worker, args=(name,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return round_results


def generate_plot(all_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]

    for i, (name, result) in enumerate(all_results.items()):
        xs, ys = [], []
        for bs in sorted(result["batch_sizes"].keys()):
            m = result["batch_sizes"][bs]
            x = m.get("tok_per_s_per_req", 0)
            y = m.get("tok_per_s", 0)
            if x > 0 and y > 0:
                xs.append(x)
                ys.append(y)

        if xs:
            ax.plot(xs, ys, marker=markers[i % len(markers)],
                    color=colors[i], label=result["label"],
                    linewidth=2, markersize=8)
            # Annotate batch sizes at first and last point
            for j, bs in enumerate(sorted(result["batch_sizes"].keys())):
                m = result["batch_sizes"][bs]
                x = m.get("tok_per_s_per_req", 0)
                y = m.get("tok_per_s", 0)
                if j == 0 or j == len(result["batch_sizes"]) - 1:
                    ax.annotate(f"bs={bs}", (x, y),
                                textcoords="offset points", xytext=(5, 5),
                                fontsize=7, color=colors[i])

    ax.set_xlabel("Throughput per request (tok/s/req)", fontsize=13)
    ax.set_ylabel("Total throughput (tok/s)", fontsize=13)
    ax.set_title("Throughput vs Per-Request Throughput\n"
                 "GSM8K 200 samples, max_tokens=1024, no_thinking\n"
                 "1× NVIDIA B200, batch sizes 1–128",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(RESULTS_PNG, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {RESULTS_PNG}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--configs", nargs="*", choices=list(CONFIGS.keys()),
                        help="Run only these configs (default: all)")
    parser.add_argument("--rounds", nargs="*", type=int,
                        help="Run only these round indices (0-based)")
    args = parser.parse_args()

    all_results = {}

    # Load existing results if any
    if Path(RESULTS_JSON).exists():
        try:
            all_results = json.loads(Path(RESULTS_JSON).read_text())
            print(f"Loaded {len(all_results)} existing results from {RESULTS_JSON}")
        except Exception:
            pass

    rounds = ROUNDS
    if args.rounds is not None:
        rounds = [ROUNDS[i] for i in args.rounds if i < len(ROUNDS)]
    if args.configs:
        rounds = [args.configs]

    for round_idx, config_names in enumerate(rounds):
        # Filter out already-done configs
        to_run = [n for n in config_names if n not in all_results]
        if not to_run:
            print(f"Round {round_idx}: all configs already done, skipping")
            continue

        print(f"\n{'#'*60}")
        print(f"ROUND {round_idx}: {[CONFIGS[n]['label'] for n in to_run]}")
        print(f"{'#'*60}")

        round_results = run_round(to_run, dry_run=args.dry_run)
        all_results.update(round_results)

        # Save after each round
        Path(RESULTS_JSON).write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved results to {RESULTS_JSON}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Total tok/s by config and batch size")
    print(f"{'='*80}")
    header = f"{'Config':<35}" + "".join(f"bs={bs:4d}" for bs in BATCH_SIZES)
    print(header)
    for name, result in all_results.items():
        row = f"{result['label']:<35}"
        for bs in BATCH_SIZES:
            m = result["batch_sizes"].get(bs, {})
            v = m.get("tok_per_s", 0)
            row += f"{v:8.0f}" if v else "       -"
        print(row)

    generate_plot(all_results)
    print(f"\nDone. Results: {RESULTS_JSON}, Plot: {RESULTS_PNG}")


if __name__ == "__main__":
    main()
