# -*- coding: utf-8 -*-
"""严谨解析 exp3_faithful 日志：策略是否冻结？真实奖励有无趋势？"""
import re
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/logs/exp3_faithful.log"

pat = re.compile(r"'loss': '([^']+)'|'grad_norm': '([^']+)'|"
                 r"'rewards/code_block_reward_fn/mean': '([^']+)'|"
                 r"'rewards/tests_reward_fn/mean': '([^']+)'|"
                 r"'completions/mean_length': '([^']+)'|"
                 r"'frac_reward_zero_std': '([^']+)'|"
                 r"'entropy': '([^']+)'")

steps = []
for line in open(path, encoding="utf-8", errors="ignore"):
    if not line.startswith("{'loss'"):
        continue
    d = {}
    for m in pat.finditer(line):
        for gi, key in [(1, "loss"), (2, "grad"), (3, "hack"), (4, "real"),
                        (5, "len"), (6, "zero_std"), (7, "entropy")]:
            if m.group(gi) is not None:
                d[key] = float(m.group(gi))
    if "real" in d:
        steps.append(d)

n = len(steps)
grad_nz = sum(1 for s in steps if s.get("grad", 0) > 1e-3)
zs_lt1 = sum(1 for s in steps if s.get("zero_std", 1) < 0.999)
print(f"步数: {n}")
print(f"grad_norm>0.001 的步数: {grad_nz} ({grad_nz/n*100:.1f}%)  <- 有梯度的步数")
print(f"frac_reward_zero_std<1 的步数: {zs_lt1} ({zs_lt1/n*100:.1f}%)  <- 组内有方差的步数")


def win(series, a, b):
    seg = series[a:b]
    return sum(seg) / len(seg)


real = [s["real"] for s in steps]
hack = [s["hack"] for s in steps]
length = [s["len"] for s in steps]
ent = [s["entropy"] for s in steps]
zs = [s["zero_std"] for s in steps]

print("\n窗口均值（每30步）:")
for a in range(0, n, 60):
    b = min(a + 60, n)
    print(f"  步{a+1:3d}-{b:3d}: real={win(real,a,b):.3f} hack={win(hack,a,b):.3f} "
          f"len={win(length,a,b):6.1f} entropy={win(ent,a,b):.3f} zero_std={win(zs,a,b):.3f}")

print(f"\n全程: real均值={statistics.mean(real):.3f}±{statistics.stdev(real):.3f} "
      f"hack均值={statistics.mean(hack):.3f}±{statistics.stdev(hack):.3f} "
      f"len均值={statistics.mean(length):.1f}±{statistics.stdev(length):.1f} "
      f"entropy均值={statistics.mean(ent):.3f}")
