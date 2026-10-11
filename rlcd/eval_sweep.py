# -*- coding: utf-8 -*-
"""批量评估 sweep checkpoints：sst5 validation 的 acc + ECE。

用法：python eval_sweep.py checkpoint_dir1 [checkpoint_dir2 ...]
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
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": PROMPT_TMPL.format(text=ex["text"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for ex in ds]
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
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        mask = ((confs_t > lo) & (confs_t <= hi)) if b > 0 else ((confs_t >= 0) & (confs_t <= hi))
        if mask.sum() > 0:
            ece += mask.sum().item() / n * abs(confs_t[mask].mean().item()
                                              - (golds_t[mask] == preds_t[mask]).float().mean().item())
    print(json.dumps({"model": tag, "n": n, "accuracy": round(acc, 4),
                      "ece_10bin": round(ece, 4),
                      "mean_conf": round(confs_t.mean().item(), 4)}, ensure_ascii=False))


def main():
    from datasets import load_dataset

    ds = load_dataset("SetFit/sst5", split="validation").select(range(1101))
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    tok.padding_side = "left"
    digit_ids_t = torch.tensor(digit_token_ids(tok))
    for ckpt in sys.argv[1:]:
        evaluate(ckpt, tok, ds, digit_ids_t, Path(ckpt).name)


if __name__ == "__main__":
    main()
