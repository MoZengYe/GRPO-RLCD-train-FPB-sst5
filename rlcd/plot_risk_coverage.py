# -*- coding: utf-8 -*-
"""画 FPB 风险-覆盖曲线（base vs GRPO+RLCD）。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

data = json.loads(Path("/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/outputs/fpb_risk_coverage.json")
                  .read_text(encoding="utf-8"))
curves = data["curves"]

fig, ax = plt.subplots(figsize=(8, 5.5))
styles = {"base_zero_shot": ("gray", "Base (zero-shot, 过自信)"),
          "grpo_rlcd": ("crimson", "GRPO+RLCD (Ours)")}
for key, (color, label) in styles.items():
    covs = [c * 100 for c, a in curves[key]]
    accs = [a * 100 for c, a in curves[key]]
    ax.plot(covs, accs, marker="o", ms=4, color=color, label=label, lw=2)

ax.axhline(90, ls="--", color="green", alpha=0.6, label="Target accuracy 90%")
ax.axhline(85, ls="--", color="orange", alpha=0.6, label="Target accuracy 85%")
ax.set_xlabel("Direct-answer coverage (% of requests handled by the 0.6B model)")
ax.set_ylabel("Accuracy on covered subset (%)")
ax.set_title("FPB risk-coverage curve: calibration makes small-model\nconfidence a usable routing signal")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
out = "/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/outputs/fpb_risk_coverage.png"
fig.savefig(out, dpi=150)
print("SAVED", out)
