# -*- coding: utf-8 -*-
"""分析逐题 reward 轨迹 CSV：死区/学习区/已掌握/解锁。

CSV 两列口径（staged 运行为三列）：task_id, 奖励分, 真实通过率。
奖励分与真实通过率分别统计"区域分布"——阶梯分下死区=从未得分(0.0)，
真实通过率下死区=从未通过任何 assert。
"""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/outputs/exp2_k8_rewards.csv"

by_task = defaultdict(list)       # 奖励分（staged 运行=阶梯分）
by_task_frac = defaultdict(list)  # 真实通过率
with open(path) as f:
    for row in csv.reader(f):
        by_task[int(row[0])].append(float(row[1]))
        by_task_frac[int(row[0])].append(float(row[2]) if len(row) >= 3 else float(row[1]))


def zone_stats(source):
    zones = {"死区（历次全0）": [], "已掌握（历次全1）": [],
             "学习区（混合）": [], "解锁（先0后>0）": []}
    for tid, rs in source.items():
        if all(r == 0 for r in rs):
            zones["死区（历次全0）"].append(tid)
        elif all(r == 1 for r in rs):
            zones["已掌握（历次全1）"].append(tid)
        else:
            zones["学习区（混合）"].append(tid)
        visits = [rs[i:i + 8] for i in range(0, len(rs), 8)]
        visit_means = [sum(v) / len(v) for v in visits if v]
        if len(visit_means) >= 2 and visit_means[0] == 0 and any(m > 0 for m in visit_means[1:]):
            zones["解锁（先0后>0）"].append(tid)
    return zones


n_rows = sum(len(v) for v in by_task.values())
n_tasks = len(by_task)
print(f"总记录: {n_rows} 条（{n_rows // 8} 个 8 样本组）| 不同题目: {n_tasks} / 374")
for label, source in [("奖励分口径", by_task), ("真实通过率口径", by_task_frac)]:
    zones = zone_stats(source)
    for k, v in zones.items():
        print(f"[{label}] {k}: {len(v)} 题 = {len(v) / n_tasks * 100:.1f}%")
    per_task_mean = [sum(rs) / len(rs) for rs in source.values()]
    print(f"[{label}] 逐题均值分布: min={min(per_task_mean):.2f} "
          f"中位={statistics.median(per_task_mean):.2f} max={max(per_task_mean):.2f}")
    dead = len(zones["死区（历次全0）"])
    print(f"[{label}] 若死区题永不解锁，上限 ≈ {1 - dead / n_tasks:.2f}")
    print()
