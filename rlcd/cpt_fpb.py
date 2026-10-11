# -*- coding: utf-8 -*-
"""FPB 上的继续预训练（CPT，带 wandb 曲线版）：裸句子的 next-token prediction。

曲线：train/loss 每步记录（CPT 无标签，无 accuracy 可言——曲线只有 loss）。
输出：outputs/CPT-FPB-3E。
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
from datasets import load_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--name", default="CPT-FPB-3E")
    args = ap.parse_args()

    setup_wandb_env(args.name, project="grpo-rlcd")
    wandb.init(project="grpo-rlcd", name=args.name,
               config={"method": "CPT", "lr": args.lr, "epochs": args.epochs,
                       "batch_size": args.batch_size, "model": "Qwen3-0.6B"})

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    ds = load_dataset("FinanceInc/auditor_sentiment", split="train")
    sentences = [ex["sentence"] for ex in ds]

    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=torch.bfloat16).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    eos = tok.eos_token_id
    pad_id = tok.pad_token_id or eos

    idx = list(range(len(sentences)))
    random.seed(42)
    steps_per_epoch = (len(idx) + args.batch_size - 1) // args.batch_size
    total = steps_per_epoch * args.epochs
    step, t0 = 0, time.time()

    for epoch in range(args.epochs):
        random.shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            batch = [sentences[j] for j in idx[i:i + args.batch_size]]
            seqs = [tok(s, add_special_tokens=False).input_ids + [eos] for s in batch]
            maxlen = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            labels = torch.full((len(seqs), maxlen), -100, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, s in enumerate(seqs):
                input_ids[r, :len(s)] = torch.tensor(s)
                labels[r, :len(s)] = torch.tensor(s)
                attn[r, :len(s)] = 1
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

    out_dir = str(EXP_DIR / "outputs" / args.name)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    wandb.finish()
    print(f"SAVED {out_dir} | total {time.time()-t0:.0f}s | steps {step}", flush=True)


if __name__ == "__main__":
    main()
