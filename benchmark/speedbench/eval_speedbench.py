"""
SPEED-Bench evaluation script for SGLang servers.

Measures throughput (tok/s) and acceptance metrics per forward pass.
Supports both DLLM (LinearSpec) and speculative decoding servers.

Usage:
  python eval_speedbench.py --base_url http://localhost:30001/v1 \
      [--stats_file /tmp/stats.jsonl] [--max_tokens 1024] [--concurrent 1]
"""

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", required=True, help="Server base URL, e.g. http://localhost:30001/v1")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--concurrent", type=int, default=1, help="Number of concurrent requests (1=serial)")
    p.add_argument("--stats_file", default=None, help="Path to server-side stats JSONL (LinearSpec)")
    p.add_argument("--single_turn_only", action="store_true", help="Skip multi-turn prompts")
    p.add_argument("--dataset_cache", default="/data/huggingface/hub")
    p.add_argument("--api", choices=["completion", "chat"], default="completion",
                   help="API type: completion (default, for DLLM) or chat (for Qwen3/Eagle3)")
    p.add_argument("--no_thinking", action="store_true",
                   help="Disable thinking mode for chat API (Qwen3 models)")
    p.add_argument("--summary_path", default=None,
                   help="Override path for summary JSON output")
    return p.parse_args()


def load_speedbench(cache_dir):
    ds = load_dataset(
        "nvidia/SPEED-Bench-Internal",
        "qualitative",
        split="test",
        cache_dir=cache_dir,
    )
    return list(ds)


def build_prompt(turns):
    """Concatenate turns into a single prompt for completion API."""
    return "\n\n".join(turns)


def generate(base_url, prompt_or_turns, max_tokens, model="default", api="completion", no_thinking=False):
    if api == "chat":
        url = f"{base_url}/chat/completions"
        messages = []
        turns = prompt_or_turns if isinstance(prompt_or_turns, list) else [prompt_or_turns]
        for i, t in enumerate(turns):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": t})
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = f"{base_url}/completions"
        prompt = "\n\n".join(prompt_or_turns) if isinstance(prompt_or_turns, list) else prompt_or_turns
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    return completion_tokens


def read_stats_tail(stats_file, n_before):
    """Read new entries appended to stats JSONL since we last checked."""
    if not stats_file or not os.path.exists(stats_file):
        return []
    with open(stats_file) as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines[n_before:]]


def main():
    args = parse_args()
    samples = load_speedbench(args.dataset_cache)

    if args.single_turn_only:
        samples = [s for s in samples if not s.get("multiturn", False)]
        print(f"Single-turn only: {len(samples)} samples")
    else:
        print(f"All samples: {len(samples)} (incl. {sum(1 for s in samples if s.get('multiturn'))} multi-turn)")

    # Track stats_file baseline
    stats_before = 0
    if args.stats_file and os.path.exists(args.stats_file):
        with open(args.stats_file) as f:
            stats_before = sum(1 for _ in f)

    # Per-category accumulators
    cat_tokens = defaultdict(int)
    cat_time = defaultdict(float)
    cat_count = defaultdict(int)
    total_tokens = 0
    t_start = time.time()

    if args.concurrent <= 1:
        # Serial path (original behavior — accurate per-category timing)
        for sample in tqdm(samples, desc="Generating"):
            cat = sample["category"]
            turns = sample["turns"]
            t0 = time.time()
            toks = generate(args.base_url, turns, args.max_tokens, api=args.api, no_thinking=args.no_thinking)
            elapsed = time.time() - t0
            cat_tokens[cat] += toks
            cat_time[cat] += elapsed
            cat_count[cat] += 1
            total_tokens += toks
    else:
        # Concurrent path — send up to args.concurrent requests in parallel
        lock = threading.Lock()

        def _run(sample):
            toks = generate(args.base_url, sample["turns"], args.max_tokens,
                            api=args.api, no_thinking=args.no_thinking)
            return sample["category"], toks

        with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            futures = {executor.submit(_run, s): s for s in samples}
            for future in tqdm(as_completed(futures), total=len(samples), desc=f"Generating (c={args.concurrent})"):
                cat, toks = future.result()
                with lock:
                    cat_tokens[cat] += toks
                    cat_count[cat] += 1
                    total_tokens += toks
        # Per-category wall time is undefined in concurrent mode; set to 0 so
        # per-cat tok/s is not printed (overall tok/s is what matters here).

    total_time = time.time() - t_start
    overall_toks = sum(cat_tokens.values())

    # Read server-side stats (DLLM only)
    new_stats = read_stats_tail(args.stats_file, stats_before)
    has_stats = len(new_stats) > 0

    print("\n" + "=" * 70)
    print(f"SPEED-Bench Results  |  {len(samples)} samples  |  max_tokens={args.max_tokens}")
    print(f"Total time: {total_time:.1f}s  |  Overall throughput: {overall_toks/total_time:.1f} tok/s")
    if has_stats:
        total_fp = sum(d["forward_passes"] for d in new_stats)
        total_tok_stat = sum(d["tokens"] for d in new_stats)
        n_blk = len(new_stats)
        acc_rate = sum(d["acceptance_rate"] for d in new_stats) / n_blk
        print(f"tok/FP: {total_tok_stat/total_fp:.3f}  |  FPs/blk: {total_fp/n_blk:.3f}  |  "
              f"tok/blk: {total_tok_stat/n_blk:.3f}  |  AccRate: {acc_rate*100:.2f}%")
    print()

    # Per-category table
    cats = sorted(cat_tokens.keys())
    print(f"{'Category':<16} {'n':>4} {'tok/s':>8} {'avg_len':>8}", end="")
    if has_stats:
        print(f"  (overall stats above)", end="")
    print()
    print("-" * 42)
    for cat in cats:
        n = cat_count[cat]
        tps = cat_tokens[cat] / cat_time[cat] if cat_time[cat] > 0 else float("nan")
        avg_len = cat_tokens[cat] / n if n > 0 else 0
        tps_str = f"{tps:>8.1f}" if cat_time[cat] > 0 else "       N/A"
        print(f"{cat:<16} {n:>4} {tps_str} {avg_len:>8.1f}")
    print("-" * 42)
    print(f"{'TOTAL':<16} {len(samples):>4} {overall_toks/total_time:>8.1f} {overall_toks/len(samples):>8.1f}")

    # Save summary JSON
    out = {
        "total_samples": len(samples),
        "total_tokens": overall_toks,
        "total_time_s": round(total_time, 2),
        "throughput_toks": round(overall_toks / total_time, 2),
        "max_tokens": args.max_tokens,
        "concurrent": args.concurrent,
        "per_category": {
            cat: {
                "n": cat_count[cat],
                "tokens": cat_tokens[cat],
                "time_s": round(cat_time[cat], 2) if cat_time[cat] > 0 else None,
                "toks_per_s": round(cat_tokens[cat] / cat_time[cat], 2) if cat_time[cat] > 0 else None,
                "avg_gen_len": round(cat_tokens[cat] / cat_count[cat], 1) if cat_count[cat] > 0 else 0,
            }
            for cat in cats
        },
    }
    if has_stats:
        out["tok_per_fp"] = round(total_tok_stat / total_fp, 3)
        out["fps_per_blk"] = round(total_fp / n_blk, 3)
        out["tok_per_blk"] = round(total_tok_stat / n_blk, 3)
        out["acceptance_rate"] = round(acc_rate, 4)

    if args.summary_path:
        summary_path = args.summary_path
    elif args.stats_file:
        summary_path = args.stats_file.replace(".jsonl", "_summary.json")
    else:
        summary_path = "/tmp/speedbench_summary.json"
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
