# -*- coding: utf-8 -*-
"""按步（每16行=1步）提取 staged / 真实通过率 双口径轨迹。"""
import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/outputs/exp2_k8_staged_rewards_v2.csv"

rows = []
with open(path) as f:
    for row in csv.reader(f):
        rows.append((float(row[1]), float(row[2]) if len(row) >= 3 else float(row[1])))

STEP = 16  # 每步 16 个 completion（2 prompts × k=8）
n_steps = len(rows) // STEP
staged_steps = [sum(r[0] for r in rows[i * STEP:(i + 1) * STEP]) / STEP for i in range(n_steps)]
frac_steps = [sum(r[1] for r in rows[i * STEP:(i + 1) * STEP]) / STEP for i in range(n_steps)]


def win(series, a, b):
    seg = series[a:b]
    return sum(seg) / len(seg)


for name, series in [("阶梯分", staged_steps), ("真实通过率", frac_steps)]:
    print(f"[{name}] first30(步1-30)={win(series, 0, 30):.3f} "
          f"mid30(步135-165)={win(series, 134, 165):.3f} "
          f"last30(步271-300)={win(series, n_steps - 30, n_steps):.3f} (n_steps={n_steps})")
