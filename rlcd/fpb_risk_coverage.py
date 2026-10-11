# -*- coding: utf-8 -*-
"""FPB 风险-覆盖（级联调度）评估：base vs GRPO+RLCD。

对 test 969 题逐题导出 (置信度, 正确与否)，按置信度降序构造风险-覆盖曲线，
并输出关键业务数字（不同阈值下的直答比例与直答准确率）。
同时画图（matplotlib）保存 PNG。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from common import MODEL_PATH  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from fpb_rlcd import PROMPT_TMPL, digit_token_ids3  # noqa: E402


def per_example(model_path, tok, ds, digit_ids_t, tag):
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": PROMPT_TMPL.format(sentence=ex["sentence"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for ex in ds]
    recs = []
    B = 64
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], return_tensors="pt", padding=True, padding_side="left").to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        d_logits = logits[:, digit_ids_t.to(logits.device)]
        p6 = torch.softmax(d_logits, dim=-1)
        p3 = p6.view(p6.size(0), 2, 3).sum(dim=1)
        confs, preds = p3.max(dim=-1)
        for j in range(len(texts[i:i + B])):
            g = int(ds[i + j]["label"])
            recs.append({"conf": round(confs[j].item(), 4),
                         "pred": int(preds[j].item()), "gold": g,
                         "correct": int(preds[j].item() == g)})
    del model
    torch.cuda.empty_cache()
    return recs


def risk_coverage(recs, n_points=20):
    """按置信度降序的风险-覆盖曲线。返回 [(coverage, accuracy), ...]。"""
    s = sorted(recs, key=lambda r: -r["conf"])
    n = len(s)
    out = []
    for k in range(1, n + 1):
        acc = sum(r["correct"] for r in s[:k]) / k
        out.append((k / n, acc))
    # 降采样到 n_points 个点（含首尾）
    idx = [round(i * (len(out) - 1) / (n_points - 1)) for i in range(n_points)]
    return [out[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--out", default="fpb_risk_coverage.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("FinanceInc/auditor_sentiment", split="test")
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    digit_ids_t = torch.tensor(digit_token_ids3(tok))

    curves = {}
    if not args.skip_base:
        recs_base = per_example(str(MODEL_PATH), tok, ds, digit_ids_t, "base")
        curves["base_zero_shot"] = risk_coverage(recs_base)
        Path(args.out).write_text(json.dumps(
            {"base_per_example": recs_base, "curves": curves}, ensure_ascii=False), encoding="utf-8")
    if args.checkpoint:
        recs_rlcd = per_example(args.checkpoint, tok, ds, digit_ids_t, "rlcd")
        curves["grpo_rlcd"] = risk_coverage(recs_rlcd)
        # 合并保存（读旧文件补 RLCD 明细）
        try:
            old = json.loads(Path(args.out).read_text(encoding="utf-8"))
            old["rlcd_per_example"] = recs_rlcd
            old["curves"] = curves
            Path(args.out).write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
        except Exception:
            Path(args.out).write_text(json.dumps(
                {"rlcd_per_example": recs_rlcd, "curves": curves}, ensure_ascii=False), encoding="utf-8")

    def summarize(name, curve):
        print(f"--- {name} ---")
        for cov, acc in curve:
            print(f"  覆盖 {cov*100:5.1f}% → 覆盖子集准确率 {acc*100:.1f}%")
        # 关键业务数字
        best = max(a for _, a in curve)
        cov_at_best = next(c for c, a in curve if a == best)
        print(f"  最高覆盖子集准确率: {best*100:.1f}% @ 覆盖 {cov_at_best*100:.0f}%")

    for name, curve in curves.items():
        summarize(name, curve)

    # 阈值表：为保持 85% / 90% 准确率，各自能直答多少
    for target in (0.85, 0.90):
        line = f"目标准确率 ≥{target*100:.0f}% 时可直答比例: "
        for name, curve in curves.items():
            covs = [c for c, a in curve if a >= target]
            line += f"{name}={max(covs)*100:.0f}% " if covs else f"{name}=0% "
        print(line)


if __name__ == "__main__":
    main()
