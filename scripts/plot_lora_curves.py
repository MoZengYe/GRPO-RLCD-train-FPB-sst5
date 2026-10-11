"""从 verl GRPO 的 train.log 里提取 LoRA 训练指标并画图。

用法（服务器上）：
    python scripts/plot_lora_curves.py [logs/lora_run_XXXX/train.log]
不传参数时自动取最近一次 lora_run_* 的 train.log。
输出：同目录下的 curves.png，并在终端打印指标表。
"""

import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b"
KEYS = [
    ("actor/loss", "loss"),
    ("actor/grad_norm", "grad_norm"),
    ("actor/lr", "lr"),
    ("perf/mfu/actor", "MFU"),
    ("perf/throughput", "throughput(tok/s)"),
    ("rollout_corr/rollout_ppl", "rollout_ppl"),
    ("response_length/mean", "response_len"),
    ("critic/rewards/mean", "reward"),
]


def find_log() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    cands = sorted(glob.glob(os.path.join(ROOT, "logs", "lora_run_*", "train.log")))
    if not cands:
        raise SystemExit("没找到 lora_run_*/train.log")
    return cands[-1]


def main():
    log_path = find_log()
    text = open(log_path, errors="ignore").read().replace("\r", "\n")
    steps = {}
    for line in text.split("\n"):
        m = re.search(r"step:(\d+) - ", line)
        if not m:
            continue
        step = int(m.group(1))
        row = {}
        for key, _ in KEYS:
            mm = re.search(re.escape(key) + r":([-0-9.eE+]+)", line)
            if mm:
                row[key] = float(mm.group(1))
        if row:
            steps[step] = row

    print("log:", log_path)
    header = "step  " + "  ".join(f"{label:>16}" for _, label in KEYS)
    print(header)
    for step in sorted(steps):
        vals = steps[step]
        print(f"{step:>4}  " + "  ".join(f"{vals.get(k, float('nan')):>16.4g}" for k, _ in KEYS))

    xs = sorted(steps)
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for ax, (key, label) in zip(axes.flat, KEYS):
        ys = [steps[s].get(key, float("nan")) for s in xs]
        ax.plot(xs, ys, marker="o")
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle("GRPO + LoRA (r=16, alpha=32, lr=1e-4) — 8 steps")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(log_path)), "curves.png")
    fig.savefig(out, dpi=120)
    print("saved:", out)


if __name__ == "__main__":
    main()
