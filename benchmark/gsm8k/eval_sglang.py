#!/usr/bin/env python3
"""
Standalone GSM8K & MATH500 evaluation via SGLang (OpenAI-compatible API).

Reproduces the exact scoring logic used for Nemotron-Diffusion-Instruct benchmarks.
Requires a running SGLang server (or any OpenAI-compatible endpoint).

Usage:
    # Start SGLang server (example):
    #   python -m sglang.launch_server --model nvidia/Nemotron-Diffusion-Exp-Ministral-8B-Instruct \
    #       --port 30000 --tp 4

    # Evaluate GSM8K (full test set, 1319 samples):
    python eval_sglang.py --benchmark gsm8k --base_url http://localhost:30000/v1

    # Evaluate MATH500:
    python eval_sglang.py --benchmark math --base_url http://localhost:30000/v1

    # Nothink mode (disable chain-of-thought):
    python eval_sglang.py --benchmark gsm8k --no_thinking --base_url http://localhost:30000/v1

    # Subset for quick testing:
    python eval_sglang.py --benchmark gsm8k --num_samples 50 --base_url http://localhost:30000/v1

Model: nvidia/Nemotron-Diffusion-Exp-Ministral-8B-Instruct (or -v2)
Tokenizer: uses its chat template with enable_thinking parameter.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Answer extraction & scoring
# ---------------------------------------------------------------------------

def search_boxed(string: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces.
    From: github.com/NVIDIA-NeMo/Skills nemo_skills/evaluation/math_grader.py"""
    if "\\boxed" not in string:
        return None
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    retval = string[idx : right_brace_idx + 1]
    left = "\\boxed{"
    if retval[: len(left)] == left and retval[-1] == "}":
        return retval[len(left) : -1]
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    if text is None:
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    match = re.search(r'####\s*([+-]?\d[\d,]*\.?\d*)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    boxed = search_boxed(text)
    if boxed is not None:
        return boxed.replace(",", "").strip()
    numbers = re.findall(r'[+-]?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def extract_gsm8k_gold(answer_text: str) -> str:
    match = re.search(r'####\s*([+-]?\d[\d,]*\.?\d*)', answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_text.strip()


def normalize_number(s: str) -> str:
    if s is None:
        return ""
    # Strip LaTeX formatting: \text{...}, \mathrm{...}, \textbf{...}
    s = re.sub(r'\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}', r'\1', s)
    # Strip \$ and $ signs
    s = s.replace('\\$', '').replace('$', '')
    # Strip %, trailing units, and LaTeX spacing
    s = re.sub(r'\\[,;:\s]', ' ', s)  # \, \; \: → space
    s = re.sub(r'\\%', '%', s)  # \% → %
    s = s.strip().rstrip('%')
    # Remove trailing backslash (e.g. from \boxed{60\%} → "60\" after rstrip('%'))
    s = s.rstrip('\\').strip()
    # Remove trailing text units (possibly preceded by "cubic", "square", etc.)
    s = re.sub(r'\s*(?:cubic|square)?\s*(?:minutes|grams|mph|hours|dollars|cents|feet|meters|inches|pounds|ounces|gallons|liters|days|weeks|months|years|people|students|books|miles|cm|kg|lb|oz|ft|in|m|s)\s*$', '', s, flags=re.IGNORECASE)
    s = s.strip()
    try:
        val = float(s.replace(',', ''))
        if not (val == val) or val == float('inf') or val == float('-inf'):
            return s
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, TypeError, OverflowError):
        return s


def score_gsm8k(pred_text: str, gold_answer_text: str) -> dict:
    pred = extract_gsm8k_answer(pred_text)
    gold = extract_gsm8k_gold(gold_answer_text)
    gold_norm = normalize_number(gold)
    pred_norm = normalize_number(pred) if pred else ""
    return {
        "is_correct": gold_norm == pred_norm,
        "pred_answer": pred,
        "pred_normalized": pred_norm,
        "gold_answer": gold,
        "gold_normalized": gold_norm,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k(num_samples: int = -1, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    import random
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = [{"question": row["question"], "answer": row["answer"]} for row in ds]
    if 0 < num_samples < len(samples):
        rng = random.Random(seed)
        indices = rng.sample(range(len(samples)), num_samples)
        indices.sort()
        samples = [samples[i] for i in indices]
    return samples


def load_math500(num_samples: int = -1, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    import random
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    samples = [{"question": row["problem"], "answer": row["answer"]} for row in ds]
    if 0 < num_samples < len(samples):
        rng = random.Random(seed)
        indices = rng.sample(range(len(samples)), num_samples)
        indices.sort()
        samples = [samples[i] for i in indices]
    return samples


# ---------------------------------------------------------------------------
# Answer extraction for MATH500
# ---------------------------------------------------------------------------

def extract_math_answer(text: str) -> str | None:
    """Extract answer from model response for MATH problems."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    # Try \boxed{...} first (standard MATH format), handles nested braces
    boxed = search_boxed(text)
    if boxed is not None:
        return boxed.strip()
    # Try #### format
    match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    # Try "the answer is ..." pattern
    match = re.search(r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def normalize_math_answer(s: str) -> str:
    """Normalize a MATH answer for comparison."""
    s = s.strip()
    # Remove surrounding $ signs
    s = s.strip('$')
    # Remove \text{...}, \mathrm{...} wrappers
    s = re.sub(r'\\(?:text|mathrm)\{([^}]*)\}', r'\1', s)
    # Try numeric normalization
    try:
        val = float(s.replace(',', ''))
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, TypeError):
        pass
    # Remove whitespace for symbolic comparison
    return re.sub(r'\s+', '', s)


def score_math(pred_text: str, gold_answer: str) -> dict:
    pred = extract_math_answer(pred_text)
    gold_norm = normalize_math_answer(gold_answer)
    pred_norm = normalize_math_answer(pred) if pred else ""
    return {
        "is_correct": gold_norm == pred_norm,
        "pred_answer": pred,
        "pred_normalized": pred_norm,
        "gold_answer": gold_answer,
        "gold_normalized": gold_norm,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

GSM8K_SYSTEM = "You are a helpful math assistant. Solve the problem step by step. At the end, provide the final numeric answer after ####."

MATH_SYSTEM = "You are a helpful math assistant. Solve the problem step by step. Put your final answer in \\boxed{}."

GSM8K_V2_INSTRUCTION = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}."


def make_gsm8k_messages(question: str, prompt_style: str = "default") -> list[dict]:
    if prompt_style == "v2":
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"{GSM8K_V2_INSTRUCTION}\n\n{question}"},
        ]
    return [
        {"role": "system", "content": GSM8K_SYSTEM},
        {"role": "user", "content": question},
    ]


def make_math_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": MATH_SYSTEM},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Generation via OpenAI-compatible API
# ---------------------------------------------------------------------------

def generate_batch(
    messages_list: list[list[dict]],
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float = 0.0,
    concurrent: int = 64,
    no_thinking: bool = False,
) -> tuple[list[str], list[int]]:
    import requests as _requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    url = f"{base_url}/chat/completions"
    results = [None] * len(messages_list)
    completion_tokens = [0] * len(messages_list)

    def _call(idx, msgs):
        payload = {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        resp = _requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        ctok = data.get("usage", {}).get("completion_tokens", 0) or 0
        return idx, text, ctok

    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = {pool.submit(_call, i, m): i for i, m in enumerate(messages_list)}
        done = 0
        for fut in as_completed(futures):
            idx, text, ctok = fut.result()
            results[idx] = text
            completion_tokens[idx] = ctok
            done += 1
            if done % 100 == 0 or done == len(messages_list):
                print(f"  [{done}/{len(messages_list)}] generated", flush=True)

    return results, completion_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def eval_gsm8k(args):
    print(f"Loading GSM8K test set (num_samples={args.num_samples})...")
    samples = load_gsm8k(args.num_samples, args.seed)
    print(f"  {len(samples)} samples loaded")

    prompt_style = getattr(args, 'prompt_style', 'default')
    messages_list = [make_gsm8k_messages(s["question"], prompt_style=prompt_style) for s in samples]

    print(f"Generating responses (max_tokens={args.max_tokens}, concurrent={args.concurrent})...")
    t0 = time.time()
    no_thinking = getattr(args, 'no_thinking', False)
    responses, completion_tokens = generate_batch(
        messages_list, args.base_url, args.model, args.max_tokens,
        args.temperature, args.concurrent, no_thinking=no_thinking,
    )
    elapsed = time.time() - t0

    total_tok = sum(completion_tokens)
    tok_per_sec = total_tok / elapsed if elapsed > 0 else 0
    avg_gen_len = total_tok / len(samples) if samples else 0
    print(f"  Generation done in {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/sec)")
    print(f"  Total completion tokens: {total_tok}")
    print(f"  Throughput: {tok_per_sec:.1f} tok/s")
    print(f"  Avg gen length: {avg_gen_len:.1f} tok/sample")

    correct = 0
    results = []
    for i, (sample, resp) in enumerate(zip(samples, responses)):
        sc = score_gsm8k(resp, sample["answer"])
        sc["question"] = sample["question"]
        sc["response"] = resp
        results.append(sc)
        if sc["is_correct"]:
            correct += 1

    acc = 100.0 * correct / len(samples)
    print(f"\nGSM8K Accuracy: {correct}/{len(samples)} = {acc:.1f}%")

    return {"benchmark": "gsm8k", "accuracy": acc, "correct": correct,
            "total": len(samples), "elapsed": elapsed,
            "total_tokens": total_tok, "tok_per_sec": tok_per_sec,
            "avg_gen_length": avg_gen_len, "results": results}


def eval_math(args):
    print(f"Loading MATH500 test set (num_samples={args.num_samples})...")
    samples = load_math500(args.num_samples, args.seed)
    print(f"  {len(samples)} samples loaded")

    messages_list = [make_math_messages(s["question"]) for s in samples]

    print(f"Generating responses (max_tokens={args.max_tokens}, concurrent={args.concurrent})...")
    t0 = time.time()
    no_thinking = getattr(args, 'no_thinking', False)
    responses, completion_tokens = generate_batch(
        messages_list, args.base_url, args.model, args.max_tokens,
        args.temperature, args.concurrent, no_thinking=no_thinking,
    )
    elapsed = time.time() - t0

    total_tok = sum(completion_tokens)
    tok_per_sec = total_tok / elapsed if elapsed > 0 else 0
    avg_gen_len = total_tok / len(samples) if samples else 0
    print(f"  Generation done in {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/sec)")
    print(f"  Total completion tokens: {total_tok}")
    print(f"  Throughput: {tok_per_sec:.1f} tok/s")
    print(f"  Avg gen length: {avg_gen_len:.1f} tok/sample")

    correct = 0
    results = []
    for i, (sample, resp) in enumerate(zip(samples, responses)):
        sc = score_math(resp, sample["answer"])
        sc["question"] = sample["question"]
        sc["response"] = resp
        results.append(sc)
        if sc["is_correct"]:
            correct += 1

    acc = 100.0 * correct / len(samples)
    print(f"\nMATH500 Accuracy: {correct}/{len(samples)} = {acc:.1f}%")

    return {"benchmark": "math500", "accuracy": acc, "correct": correct,
            "total": len(samples), "elapsed": elapsed,
            "total_tokens": total_tok, "tok_per_sec": tok_per_sec,
            "avg_gen_length": avg_gen_len, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["gsm8k", "math"], default="gsm8k")
    parser.add_argument("--base_url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrent", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_thinking", action="store_true", default=True,
                        help="Disable thinking mode (pass enable_thinking=False to chat template). Default: True for v2 model.")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode (overrides --no_thinking)")
    parser.add_argument("--prompt_style", choices=["default", "v2"], default="v2",
                        help="Prompt style: 'default' (#### format) or 'v2' (boxed format with empty system). Default: v2.")
    args = parser.parse_args()

    # --thinking overrides --no_thinking
    if args.thinking:
        args.no_thinking = False

    print(f"Benchmark: {args.benchmark}")
    print(f"API: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Thinking: {'enabled' if not args.no_thinking else 'disabled'}")
    print()

    if args.benchmark == "math":
        out = eval_math(args)
    else:
        out = eval_gsm8k(args)

    if args.output:
        save = {k: v for k, v in out.items() if k != "results"}
        save["examples"] = out["results"][:5]
        save["errors"] = [r for r in out["results"] if not r["is_correct"]][:20]
        save["all_results"] = out["results"]
        with open(args.output, "w") as f:
            json.dump(save, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
