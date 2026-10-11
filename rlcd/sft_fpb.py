# -*- coding: utf-8 -*-
"""FPB 上的 SFT（带 wandb 曲线版）：prompt + gold token 的交叉熵。

相对初版新增：
  - wandb 集成（项目 grpo-rlcd，loss 每步记录、validation acc 每 100 步）
  - train/val 切分（尾部 300 条为 val——与 fpb_rlcd.py 的切法一致）
  - 训练中周期性 val-acc 检查（曲线能看到分类能力随训练的增长）

输出：outputs/SFT-FPB-E2（供 eval_fpb.py 终测）。
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
import wandb  # noqa: E402
from common import EXP_DIR, MODEL_PATH, setup_wandb_env  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from fpb_rlcd import PROMPT_TMPL, build_fpb_dataset, digit_token_ids3  # noqa: E402


def val_accuracy(model, tok, val_ds, digit_ids_t):
    """快速置信度读出评估：digit 分布 argmax 的 accuracy（val 300 条）。"""
    model.eval()
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": PROMPT_TMPL.format(sentence=ex["sentence"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for ex in val_ds]
    golds, preds = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], return_tensors="pt", padding=True,
                      padding_side="left").to("cuda")
            logits = model(**enc).logits[:, -1, :].float()
            d = logits[:, digit_ids_t.to(logits.device)]
            p6 = torch.softmax(d, dim=-1)
            p3 = p6.view(p6.size(0), 2, 3).sum(dim=1)
            preds.extend(p3.argmax(dim=-1).tolist())
            golds.extend(int(ex["label"]) for ex in val_ds[i:i + 64])
    model.train()
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--name", default="SFT-FPB-E2")
    ap.add_argument("--temperature", type=float, default=1.0)  # 占位（SFT 无采样）
    args = ap.parse_args()

    setup_wandb_env(args.name, project="grpo-rlcd")
    wandb.init(project="grpo-rlcd", name=args.name,
               config={"method": "SFT", "lr": args.lr, "epochs": args.epochs,
                       "batch_size": args.batch_size, "model": "Qwen3-0.6B"})

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    ds_full = build_fpb_dataset(tok, split="train")
    val_ds = ds_full.select(range(len(ds_full) - 300, len(ds_full)))   # 尾部 300 = val
    ds = ds_full.select(range(len(ds_full) - 300))
    digit_ids_t = torch.tensor(digit_token_ids3(tok))
    print(f"FPB train: {len(ds)} | val: {len(val_ds)}")

    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=torch.bfloat16).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pad_id = tok.pad_token_id or tok.eos_token_id

    gold_token = {d: tok.encode(" " + str(d), add_special_tokens=False)[0] for d in range(3)}
    idx = list(range(len(ds)))
    random.seed(42)
    steps_per_epoch = (len(idx) + args.batch_size - 1) // args.batch_size
    total = steps_per_epoch * args.epochs
    step, t0 = 0, time.time()

    for epoch in range(args.epochs):
        random.shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            batch = idx[i:i + args.batch_size]
            seqs, labels_batch = [], []
            for j in batch:
                ex = ds[j]
                pids = tok(ex["prompt"], add_special_tokens=False).input_ids
                gids = tok(" " + str(ex["gold"]), add_special_tokens=False).input_ids
                seqs.append(pids + gids)
                labels_batch.append([-100] * len(pids) + gids)
            maxlen = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            labels = torch.full((len(seqs), maxlen), -100, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, (s, l) in enumerate(zip(seqs, labels_batch)):
                input_ids[r, :len(s)] = torch.tensor(s)
                labels[r, :len(l)] = torch.tensor(l)
            out = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                        labels=labels.cuda())
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0 or step == total:
                wandb.log({"train/loss": out.loss.item(), "step": step}, step=step)
                print(f"step {step}/{total} loss={out.loss.item():.4f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            if step % 100 == 0 or step == total:
                acc = val_accuracy(model, tok, val_ds, digit_ids_t)
                wandb.log({"val/accuracy": acc, "step": step}, step=step)
                print(f"  [val] acc={acc:.3f} @step {step}", flush=True)

    out_dir = str(EXP_DIR / "outputs" / args.name)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    wandb.finish()
    print(f"SAVED {out_dir} | total {time.time()-t0:.0f}s | steps {step}", flush=True)


if __name__ == "__main__":
    main()
