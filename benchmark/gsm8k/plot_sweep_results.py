#!/usr/bin/env python3
"""Plot GSM8K batch sweep: each config = one line, x=batch_size, y=tok/s total and tok/s/req."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

RESULTS_FILE = "/tmp/batch_sweep_results_v2.json"

# Config display order and colors
CONFIG_ORDER = [
    "nemo_fp8_bl32",
    "nemo_fp8_compile_bl32",
    "nemo_fp8_bl64",
    "nemo_fp8_compile_bl64",
    "nemo_bf16_bl32",
    "nemo_bf16_bl64",
    "ar_qwen3_8b",
    "eagle3_qwen3_8b",
    "mtp_qwen35_9b",
]

COLORS = [
    "#e6194b",  # red
    "#f58231",  # orange
    "#ffe119",  # yellow
    "#bfef45",  # lime
    "#3cb44b",  # green
    "#42d4f4",  # cyan
    "#4363d8",  # blue
    "#911eb4",  # purple
    "#f032e6",  # magenta
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def load_data(path):
    return json.loads(Path(path).read_text())


def extract_series(data, metric):
    series = {}
    for name in CONFIG_ORDER:
        if name not in data:
            continue
        cfg = data[name]
        label = cfg.get("label", name)
        xs, ys = [], []
        for bs in BATCH_SIZES:
            v = cfg.get("batch_sizes", {}).get(str(bs), {})
            val = v.get(metric)
            if val is not None and val > 0:
                xs.append(bs)
                ys.append(val)
        if xs:
            series[name] = {"label": label, "xs": xs, "ys": ys}
    return series


def extract_frontier(data):
    """Return per-config series: x=tok/s/req, y=tok/s total, annotated by batch size."""
    series = {}
    for name in CONFIG_ORDER:
        if name not in data:
            continue
        cfg = data[name]
        label = cfg.get("label", name)
        pts = []
        for bs in BATCH_SIZES:
            v = cfg.get("batch_sizes", {}).get(str(bs), {})
            x = v.get("tok_per_s_per_req")
            y = v.get("tok_per_s")
            if x and y and x > 0 and y > 0:
                pts.append((x, y, bs))
        if pts:
            series[name] = {"label": label, "pts": pts}
    return series


def make_plot(data, out_path):
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", which="both", labelsize=10)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, color="#333355", alpha=0.7)

    series = extract_frontier(data)
    for idx, name in enumerate(CONFIG_ORDER):
        if name not in series:
            continue
        s = series[name]
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        xs = [p[0] for p in s["pts"]]
        ys = [p[1] for p in s["pts"]]
        bss = [p[2] for p in s["pts"]]

        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, zorder=2)
        sc = ax.scatter(xs, ys, color=color, marker=marker, s=60, zorder=3, label=s["label"])

        # Annotate the bs=1 point (rightmost, lowest total) and bs=128 (leftmost, highest total)
        for x, y, bs in s["pts"]:
            if bs in (1, 128):
                ax.annotate(
                    f"bs={bs}",
                    (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=6.5, color=color, alpha=0.85,
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Per-Request Throughput (tok/s/req)  →  lower = more batching", color="white", fontsize=11)
    ax.set_ylabel("Total Throughput (tok/s)  →  higher = more GPU utilization", color="white", fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))

    legend = ax.legend(
        fontsize=9, facecolor="#0f3460", edgecolor="#444466",
        labelcolor="white", loc="lower left",
        framealpha=0.9, ncol=1,
    )

    ax.set_title(
        "GSM8K Batch Sweep — Throughput Frontier\n"
        "(each line = one config, dots = batch sizes 1→128, rightmost dot = bs=1)",
        color="white", fontsize=13, pad=12,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    data = load_data(RESULTS_FILE)
    out = "/tmp/batch_sweep_results_v2.png"
    make_plot(data, out)
