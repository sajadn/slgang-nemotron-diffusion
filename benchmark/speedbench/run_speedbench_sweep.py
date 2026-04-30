"""
SPEED-Bench batch-size sweep: tok/s vs tok/s/req across bs=1–128.
7 configs × 8 batch sizes = 56 eval runs across 2 rounds on 4× B200.

Configs:
  Round 1: ar_qwen3_8b, mtp_qwen35_9b, eagle3_qwen3_8b, nemo_fp8_compile_bl32
  Round 2: nemo_fp8_compile_bl64, nemo_bf16_compile_bl32, nemo_bf16_compile_bl64
"""
import json, os, re, signal, subprocess, sys, threading, time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
VENV  = "/home/mkhadkevich/dev/sglang/.venv/bin/python"
EVAL  = "/home/mkhadkevich/dev/sglang/benchmark/speedbench/eval_speedbench.py"
MODEL_NEMO = (
    "/data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-"
    "Instruct-v2/snapshots/adfaddf53841a39cb5bf8b9a793364edbaae17a9"
)
LORA_V3 = (
    "/data/huggingface/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-8B-"
    "Instruct-v2-lora/snapshots/7b4c1dc9a7aca504e9c3ddbac77de93db7463194/v3"
)
Q3_8B  = "Qwen/Qwen3-8B"
Q35_9B = ("/data/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
          "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
EAGLE  = "Tengyunw/qwen3_8b_eagle3"

RESULTS_JSON   = "/tmp/speedbench_sweep_results.json"
BATCH_SIZES    = [1, 2, 4, 8, 16, 32, 64, 128]
MAX_TOKENS     = 1024
WARMUP_TIMEOUT = 720   # seconds; +1200 for compile across all batch sizes

BASE_ENV = {**os.environ,
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
}

# ── Config definitions ─────────────────────────────────────────────────────────
CONFIGS = {
    "ar_qwen3_8b": {
        "label": "Qwen3-8B AR",
        "gpu": 0, "port": 42000,
        "api": "chat", "no_thinking": True,
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", Q3_8B,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--port", "42000",
        ],
        "env_extra": {},
    },
    "mtp_qwen35_9b": {
        "label": "Qwen3.5-9B MTP d=3",
        "gpu": 1, "port": 42001,
        "api": "chat", "no_thinking": True,
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", Q35_9B,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--max-running-requests", "128",
            "--speculative-algorithm", "NEXTN",
            "--speculative-draft-model-path", Q35_9B,
            "--speculative-eagle-topk", "1",
            "--speculative-num-steps", "3",
            "--speculative-num-draft-tokens", "3",
            "--mamba-scheduler-strategy", "extra_buffer",
            "--port", "42001",
        ],
        "env_extra": {"SGLANG_ENABLE_SPEC_V2": "1"},
    },
    "eagle3_qwen3_8b": {
        "label": "Qwen3-8B Eagle3 d=15",
        "gpu": 2, "port": 42002,
        "api": "chat", "no_thinking": True,
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", Q3_8B,
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", EAGLE,
            "--speculative-num-steps", "4",
            "--speculative-eagle-topk", "4",
            "--speculative-num-draft-tokens", "15",
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--port", "42002",
        ],
        "env_extra": {},
    },
    "nemo_fp8_compile_bl32": {
        "label": "Nemotron FP8+compile bl=32",
        "gpu": 3, "port": 42003,
        "api": "chat", "no_thinking": True,
        "is_dllm": True, "compile": True,
        "stats_file": "/tmp/stats_spb_nemo_fp8_compile_bl32.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 32,
                 "causal_context": True, "lora_path": LORA_V3,
                 "stats_file": "/tmp/stats_spb_nemo_fp8_compile_bl32.jsonl"},
        "yaml_path": "/tmp/spb_nemo_fp8_compile_bl32.yaml",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.6",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--quantization", "fp8",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/spb_nemo_fp8_compile_bl32.yaml",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--enable-torch-compile", "--torch-compile-max-bs", "128",
            "--port", "42003",
        ],
        "env_extra": {},
    },
    "nemo_fp8_compile_bl64": {
        "label": "Nemotron FP8+compile bl=64",
        "gpu": 0, "port": 42000,
        "api": "chat", "no_thinking": True,
        "is_dllm": True, "compile": True,
        "stats_file": "/tmp/stats_spb_nemo_fp8_compile_bl64.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 64,
                 "causal_context": True, "lora_path": LORA_V3,
                 "stats_file": "/tmp/stats_spb_nemo_fp8_compile_bl64.jsonl"},
        "yaml_path": "/tmp/spb_nemo_fp8_compile_bl64.yaml",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.6",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--quantization", "fp8",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/spb_nemo_fp8_compile_bl64.yaml",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--enable-torch-compile", "--torch-compile-max-bs", "128",
            "--port", "42000",
        ],
        "env_extra": {},
    },
    "nemo_bf16_compile_bl32": {
        "label": "Nemotron BF16+compile bl=32",
        "gpu": 1, "port": 42001,
        "api": "chat", "no_thinking": True,
        "is_dllm": True, "compile": True,
        "stats_file": "/tmp/stats_spb_nemo_bf16_compile_bl32.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 32,
                 "causal_context": True, "lora_path": LORA_V3,
                 "stats_file": "/tmp/stats_spb_nemo_bf16_compile_bl32.jsonl"},
        "yaml_path": "/tmp/spb_nemo_bf16_compile_bl32.yaml",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.55",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/spb_nemo_bf16_compile_bl32.yaml",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--enable-torch-compile", "--torch-compile-max-bs", "128",
            "--port", "42001",
        ],
        "env_extra": {},
    },
    "nemo_bf16_compile_bl64": {
        "label": "Nemotron BF16+compile bl=64",
        "gpu": 2, "port": 42002,
        "api": "chat", "no_thinking": True,
        "is_dllm": True, "compile": True,
        "stats_file": "/tmp/stats_spb_nemo_bf16_compile_bl64.jsonl",
        "yaml": {"algorithm": "LinearSpec", "block_size": 64,
                 "causal_context": True, "lora_path": LORA_V3,
                 "stats_file": "/tmp/stats_spb_nemo_bf16_compile_bl64.jsonl"},
        "yaml_path": "/tmp/spb_nemo_bf16_compile_bl64.yaml",
        "server_args": [
            VENV, "-m", "sglang.launch_server",
            "--model-path", MODEL_NEMO, "--tokenizer-path", MODEL_NEMO,
            "--trust-remote-code", "--tp-size", "1",
            "--mem-fraction-static", "0.55",
            "--max-running-requests", "128",
            "--attention-backend", "flashinfer",
            "--dllm-algorithm", "LinearSpec",
            "--dllm-algorithm-config", "/tmp/spb_nemo_bf16_compile_bl64.yaml",
            "--cuda-graph-bs", "1", "2", "4", "8", "16", "32", "64", "128",
            "--enable-torch-compile", "--torch-compile-max-bs", "128",
            "--port", "42002",
        ],
        "env_extra": {},
    },
}

ROUNDS = [
    ["ar_qwen3_8b", "mtp_qwen35_9b", "eagle3_qwen3_8b"],
    ["nemo_fp8_compile_bl32", "nemo_fp8_compile_bl64", "nemo_bf16_compile_bl32", "nemo_bf16_compile_bl64"],
]

# ── Helpers ────────────────────────────────────────────────────────────────────
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


def parse_summary(path):
    """Read the JSON summary saved by eval_speedbench.py."""
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def run_eval_for_bs(cfg, port, concurrent, log_prefix):
    """Run SPEED-Bench eval at given concurrency, return metrics dict."""
    log_path = f"{log_prefix}_bs{concurrent}.log"
    summary_path = f"{log_prefix}_bs{concurrent}_summary.json"
    stats_file = cfg.get("stats_file")

    # Clear stats file before this run
    if stats_file and Path(stats_file).exists():
        Path(stats_file).write_text("")

    cmd = [
        VENV, EVAL,
        "--base_url", f"http://localhost:{port}/v1",
        "--max_tokens", str(MAX_TOKENS),
        "--concurrent", str(concurrent),
        "--single_turn_only",
        "--api", cfg.get("api", "chat"),
        "--summary_path", summary_path,
    ]
    if cfg.get("no_thinking"):
        cmd.append("--no_thinking")
    if stats_file:
        cmd += ["--stats_file", stats_file]

    print(f"    [eval] concurrent={concurrent:3d} → {log_path}")
    with open(log_path, "w") as f:
        subprocess.run(cmd, env={**BASE_ENV, **cfg.get("env_extra", {})},
                       stdout=f, stderr=f, timeout=3600)

    summary = parse_summary(summary_path)
    tok_s = summary.get("throughput_toks", 0)
    avg_len = summary.get("total_tokens", 0) / max(summary.get("total_samples", 713), 1)
    tok_fp = summary.get("tok_per_fp", None)
    tok_blk = summary.get("tok_per_blk", None)
    acc_rate = summary.get("acceptance_rate", None)

    metrics = {
        "tok_per_s": round(tok_s, 1),
        "tok_per_s_per_req": round(tok_s / concurrent, 1),
        "avg_len": round(avg_len, 1),
    }
    if tok_fp is not None:
        metrics["tok_per_fp"] = tok_fp
    if tok_blk is not None:
        metrics["tok_per_blk"] = tok_blk
    if acc_rate is not None:
        metrics["acceptance_rate_pct"] = acc_rate

    tok_fp_str = f"{tok_fp:.3f}" if tok_fp else "-"
    print(f"    bs={concurrent:3d}: {tok_s:7.1f} tok/s | {tok_s/concurrent:6.1f} tok/s/req "
          f"| avg_len={avg_len:.0f} | tok/FP={tok_fp_str}")
    return metrics


def sweep_config(name, cfg):
    """Start server, sweep all batch sizes, kill server, return results."""
    print(f"\n{'='*60}")
    print(f"Config: {cfg['label']} (GPU {cfg['gpu']}, port {cfg['port']})")
    print(f"{'='*60}")

    write_yaml(cfg)

    gpu_env = {**BASE_ENV, **cfg.get("env_extra", {}),
               "CUDA_VISIBLE_DEVICES": str(cfg["gpu"])}

    server_log = f"/tmp/spb_sweep_{name}_server.log"
    print(f"  Starting server → {server_log}")
    server = subprocess.Popen(
        cfg["server_args"], env=gpu_env,
        stdout=open(server_log, "w"), stderr=subprocess.STDOUT
    )

    is_compile = cfg.get("compile", False)
    timeout = WARMUP_TIMEOUT + (1200 if is_compile else 0)
    print(f"  Waiting for server (timeout={timeout}s)...")
    if not wait_for_server(cfg["port"], timeout):
        print(f"  ERROR: server on port {cfg['port']} timed out!")
        server.kill()
        return None
    print(f"  Server ready.")

    config_results = {"label": cfg["label"], "batch_sizes": {}}
    log_prefix = f"/tmp/spb_sweep_{name}"

    for bs in BATCH_SIZES:
        metrics = run_eval_for_bs(cfg, cfg["port"], bs, log_prefix)
        config_results["batch_sizes"][str(bs)] = metrics

    print(f"  Killing server PID {server.pid}...")
    server.send_signal(signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
    time.sleep(10)
    return config_results


def run_round(config_names, all_results):
    """Run a group of configs in parallel (each on its own GPU)."""
    print(f"\n{'#'*60}")
    print(f"ROUND: {config_names}")
    print(f"{'#'*60}")

    round_results = {}
    threads = []
    lock = threading.Lock()

    def worker(name):
        result = sweep_config(name, CONFIGS[name])
        with lock:
            if result:
                round_results[name] = result
                # Save incrementally after each config completes
                all_results[name] = result
                Path(RESULTS_JSON).write_text(json.dumps(all_results, indent=2))
                print(f"\n[Saved {name} → {RESULTS_JSON}]")

    for name in config_names:
        if name in all_results:
            print(f"  Skipping {name} (already in results)")
            continue
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
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(13, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]

    for i, (name, v) in enumerate(all_results.items()):
        bss = sorted(v["batch_sizes"].keys(), key=int)
        xs = [v["batch_sizes"][b]["tok_per_s_per_req"] for b in bss]
        ys = [v["batch_sizes"][b]["tok_per_s"] for b in bss]
        ax.plot(xs, ys, marker=markers[i % len(markers)],
                color=colors[i], label=v["label"], linewidth=1.5, markersize=5)
        # Label bs=1 and bs=128 points
        for b, x, y in [(bss[0], xs[0], ys[0]), (bss[-1], xs[-1], ys[-1])]:
            ax.annotate(f"bs={b}", (x, y), textcoords="offset points",
                        xytext=(4, 3), fontsize=6, color=colors[i])

    ax.set_xlabel("Throughput per request (tok/s/req)", fontsize=11)
    ax.set_ylabel("Total throughput (tok/s)", fontsize=11)
    ax.set_title(
        "SPEED-Bench: Throughput vs Per-Request Throughput\n"
        "713 single-turn samples, max_tokens=1024, no_thinking\n"
        "1× NVIDIA B200, batch sizes 1–128",
        fontsize=10
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "/tmp/speedbench_sweep_results.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load existing results (resume support)
    all_results = {}
    if Path(RESULTS_JSON).exists():
        all_results = json.loads(Path(RESULTS_JSON).read_text())
        print(f"Loaded {len(all_results)} existing configs")

    for round_configs in ROUNDS:
        run_round(round_configs, all_results)

    print(f"\n{'='*60}")
    print(f"ALL DONE — {len(all_results)} configs in {RESULTS_JSON}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Config':<30} {'bs':>4} {'tok/s':>8} {'tok/s/req':>10} {'tok/FP':>7}")
    print("-" * 65)
    for name, v in all_results.items():
        for bs in ["1", "32", "128"]:
            if bs not in v["batch_sizes"]:
                continue
            m = v["batch_sizes"][bs]
            tok_fp = f"{m['tok_per_fp']:.3f}" if "tok_per_fp" in m else "    -"
            label = v["label"] if bs == "1" else ""
            print(f"{label:<30} {bs:>4} {m['tok_per_s']:>8.1f} {m['tok_per_s_per_req']:>10.1f} {tok_fp:>7}")
        print()

    generate_plot(all_results)
