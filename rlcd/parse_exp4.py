# -*- coding: utf-8 -*-
"""解析 exp4 RLCD 日志：accuracy / 零方差 / 熵 / 长度 轨迹。"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/logs/exp4_rlcd_G16_v2.log"

keys = [("rewards/r_correct/mean", "acc"),
        ("frac_reward_zero_std", "zs"),
        ("grad_norm", "grad"),
        ("entropy", "ent"),
        ("completions/mean_length", "len"),
        ("completions/clipped_ratio", "clip")]

data = {k: [] for _, k in keys}
for line in open(path, encoding="utf-8", errors="ignore"):
    if not line.startswith("{'loss'"):
        continue
    for full, short in keys:
        m = re.search("'" + full + "': '([0-9.eE+-]+)'", line)
        if m:
            data[short].append(float(m.group(1)))

n = len(data["acc"])


def win(series, a, b):
    seg = series[a:b]
    return sum(seg) / max(len(seg), 1)


nz = sum(1 for x in data["grad"] if x > 0.001)
print(f"步数: {n} | 有梯度步: {nz}/{n} ({nz/n*100:.0f}%) | clipped>0 步: {sum(1 for x in data['clip'] if x > 0)}")
for a in range(0, n, 60):
    b = min(a + 60, n)
    print(f"  步{a+1:3d}-{b:3d}: acc={win(data['acc'],a,b):.3f} zero_std={win(data['zs'],a,b):.2f} "
          f"entropy={win(data['ent'],a,b):.3f} len={win(data['len'],a,b):.1f}")
