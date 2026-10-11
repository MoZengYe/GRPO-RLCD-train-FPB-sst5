# -*- coding: utf-8 -*-
"""复现文章的"训练前检查"口径：k=4、temp=0.7 下，基础模型的平均 reward 与
mixed-reward group 比例。用 k8_author_logged 的 CSV（8样本组）对半切模拟 k=4。"""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/outputs/exp2_k8_rewards.csv"

# 每 16 行 = 1 步 = 2 个 prompt × 8 采样；前 8 行是 prompt A 的组，后 8 行是 B
rows = []
with open(path) as f:
    for row in csv.reader(f):
        rows.append((int(row[0]), float(row[1])))

groups8 = [rows[i:i + 8] for i in range(0, len(rows), 8)]
sub4 = []          # 所有 4 样本子组（每步 4 个：2 prompt × 2 半组）
for g in groups8:
    sub4.append([r for _, r in g[:4]])
    sub4.append([r for _, r in g[4:]])

n = len(sub4)
mixed = sum(1 for g in sub4 if len(set(g)) > 1)
all0 = sum(1 for g in sub4 if all(r == 0 for r in g))
all1 = sum(1 for g in sub4 if all(r == 1 for r in g))
means = [sum(g) / len(g) for g in sub4]

print(f"=== 文章检查口径的复现（k=4, temp=0.7, 基础模型, MBPP train）===")
print(f"4样本组总数: {n}")
print(f"平均 reward: {statistics.mean(means):.3f}   （文章: 0.22）")
print(f"mixed-reward 组（有方差）: {mixed} = {mixed/n*100:.1f}%   （文章: 约 50%）")
print(f"全 0 组: {all0} = {all0/n*100:.1f}%   |   全 1 组: {all1} = {all1/n*100:.1f}%")
