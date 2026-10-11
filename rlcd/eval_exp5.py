# -*- coding: utf-8 -*-
"""exp5 双模型对照评估：checkpoint vs base，sst5 validation 的 acc + ECE。

纪律：validation 用于校准调参；test 只在最终用一次。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from common import MODEL_PATH  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from exp5_rlcd_calibrated import PROMPT_TMPL, digit_token_ids  # noqa: E402


def evaluate(model_path, tok, ds, digit_ids_t, tag):
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    texts = []
    for ex in ds:
        content = PROMPT_TMPL.format(text=ex["text"])
        texts.append(tok.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False))
    golds, preds, confs = [], [], []
    B = 64
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], return_tensors="pt", padding=True, padding_side="left").to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        d_logits = logits[:, digit_ids_t.to(logits.device)]
        p10 = torch.softmax(d_logits, dim=-1)
        p5 = p10.view(p10.size(0), 2, 5).sum(dim=1)
        preds.extend(p5.argmax(dim=-1).tolist())
        confs.extend(p5.max(dim=-1).values.tolist())
        golds.extend(int(ds[j]["label"]) for j in range(i, min(i + B, len(texts))))
    del model
    torch.cuda.empty_cache()

    golds_t, preds_t, confs_t = torch.tensor(golds), torch.tensor(preds), torch.tensor(confs)
    acc = (golds_t == preds_t).float().mean().item()
    n = len(golds_t)
    ece = 0.0
    bins = {}
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        mask = ((confs_t > lo) & (confs_t <= hi)) if b > 0 else ((confs_t >= 0) & (confs_t <= hi))
        if mask.sum() > 0:
            conf_m = confs_t[mask].mean().item()
            acc_m = (golds_t[mask] == preds_t[mask]).float().mean().item()
            ece += mask.sum().item() / n * abs(conf_m - acc_m)
            bins[f"{lo:.1f}-{hi:.1f}"] = {"n": int(mask.sum()), "conf": round(conf_m, 3), "acc": round(acc_m, 3)}
    print(json.dumps({"model": tag, "n": n, "accuracy": round(acc, 4),
                      "ece_10bin": round(ece, 4),
                      "mean_conf": round(confs_t.mean().item(), 4)}, ensure_ascii=False))
    for k, v in bins.items():
        if v["n"] > 30:
            print(f"    bin {k}: n={v['n']} conf={v['conf']} acc={v['acc']}")


def main():
    from datasets import load_dataset

    ds = load_dataset("SetFit/sst5", split="validation").select(range(1101))
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    digit_ids_t = torch.tensor(digit_token_ids(tok))

    evaluate(str(MODEL_PATH), tok, ds, digit_ids_t, "base(Qwen3-0.6B)")
    evaluate("/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments/outputs/exp5_calibrated/checkpoint-300",
             tok, ds, digit_ids_t, "exp5_calibrated")


if __name__ == "__main__":
    main()
