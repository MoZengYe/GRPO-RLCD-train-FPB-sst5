# -*- coding: utf-8 -*-
"""量化 paced600 运行中"长样本"的稀缺度：completion 触顶/截断/未终止统计。"""
import re
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/logs/exp3_paced600.log"

maxl, clip, term, mlen = [], [], [], []
for line in open(path, encoding="utf-8", errors="ignore"):
    if not line.startswith("{'loss'"):
        continue
    for key, store in [("completions/max_length", maxl),
                       ("completions/clipped_ratio", clip),
                       ("completions/mean_terminated_length", term),
                       ("completions/mean_length", mlen)]:
        m = re.search("'" + key + "': '([0-9.]+)'", line)
        if m:
            store.append(float(m.group(1)))

n = len(maxl)
print(f"步数: {n}")
print(f"completion 最大长度 >=512: {sum(1 for x in maxl if x >= 511)} 步 | "
      f">=450: {sum(1 for x in maxl if x >= 450)} 步 | "
      f">=300: {sum(1 for x in maxl if x >= 300)} 步")
print(f"clipped_ratio > 0 的步数: {sum(1 for x in clip if x > 0)} | "
      f"> 0.05: {sum(1 for x in clip if x > 0.05)} | "
      f"> 0.3: {sum(1 for x in clip if x > 0.3)}")
print(f"全程: max_length 最大 = {max(maxl):.0f} | "
      f"mean_length 均值 = {statistics.mean(mlen):.1f} | "
      f"mean_terminated 均值 = {statistics.mean(x for x in term if x):.1f}")
# 未终止(截断)样本占比的代理：mean_terminated 与 mean_length 的差
gaps = [(m - t) / m for m, t in zip(mlen, term) if m > 0]
print(f"截断样本占比代理（(mean_len - terminated_len)/mean_len）: "
      f"均值 = {statistics.mean(gaps)*100:.1f}%，最大 = {max(gaps)*100:.1f}%")
print("前5步与后5步的 max_length:", [int(x) for x in maxl[:5]], "...", [int(x) for x in maxl[-5:]])
