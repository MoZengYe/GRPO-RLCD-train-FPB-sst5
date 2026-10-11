# -*- coding: utf-8 -*-
"""exp5 训练后评估：sst5 validation 上 accuracy + ECE（10 等宽 bin）。

用法：python eval_calibration.py <checkpoint_dir> [--split validation]
纪律：validation 用于校准调参；test 只在最终用一次。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from common import MODEL_PATH  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from exp5_rlcd_calibrated import PROMPT_TMPL, digit_token_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=1101)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("SetFit/sst5", split=args.split).select(range(min(args.n, 1101)))

    digit_ids = torch.tensor(digit_token_ids(tok))

    def build(ex):
        content = PROMPT_TMPL.format(text=ex["text"])
        return tok.apply_chat_template([{"role": "user", "content": content}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    golds, probs5_all, preds, confs = [], [], [], []
    B = 64
    texts = [build(ds[i]) for i in range(len(ds))]
    for i in range(0, len(texts), B):
        batch = texts[i:i + B]
        enc = tok(batch, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()      # 最后一个 prompt 位置预测首 token
        d_logits = logits[:, digit_ids.to(logits.device)]
        p10 = torch.softmax(d_logits, dim=-1)
        p5 = p10.view(p10.size(0), 2, 5).sum(dim=1)             # (B,5)
        preds.extend(p5.argmax(dim=-1).tolist())
        confs.extend(p5.max(dim=-1).values.tolist())
        probs5_all.append(p5.cpu())
        golds.extend(int(ds[j]["label"]) for j in range(i, min(i + B, len(ds))))

    golds_t = torch.tensor(golds)
    preds_t = torch.tensor(preds)
    confs_t = torch.tensor(confs)
    acc = (golds_t == preds_t).float().mean().item()

    # 10 等宽 bin ECE
    ece, n = 0.0, len(golds_t)
    for lo in [x / 10 for x in range(10)]:
        hi = lo + 0.1
        mask = (confs_t > lo) & (confs_t <= hi) if lo > 0 else (confs_t >= 0) & (confs_t <= hi)
        if mask.sum() > 0:
            ece += mask.sum().item() / n * abs(confs_t[mask].mean().item() - (golds_t[mask] == preds_t[mask]).float().mean().item())

    result = {"checkpoint": args.checkpoint, "split": args.split, "n": n,
              "accuracy": round(acc, 4), "ece_10bin": round(ece, 4),
              "mean_confidence": round(confs_t.mean().item(), 4),
              "mean_true_accuracy": round((golds_t == preds_t).float().mean().item(), 4)}
    print(json.dumps(result, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
