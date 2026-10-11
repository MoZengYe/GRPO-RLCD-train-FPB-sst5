# -*- coding: utf-8 -*-
"""FPB-GRPO-RLCD：金融新闻 3 类情感的校准决策训练（E1 = 1 epoch）。

数据：FinanceInc/auditor_sentiment（FPB allagree 子集，标注者全票同意）
  train 3877 / test 969，3 类：0=negative, 1=neutral, 2=positive
方法：与 exp5 相同的 CalibratedGRPOTrainer（Brier 进 loss），类别数=3
命名：wandb run = FPB-GRPO-RLCD-E1（项目 grpo-rlcd）

用法：
  python fpb_rlcd.py                 # 1 epoch（约 1688 步）
  python fpb_rlcd.py --max-steps 3   # 冒烟
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from common import EXP_DIR, MODEL_PATH, setup_wandb_env  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from exp5_rlcd_calibrated import CalibratedGRPOTrainer  # noqa: E402

CANDIDATES = ["negative", "neutral", "positive"]

PROMPT_TMPL = (
    "Classify the sentiment of this financial news sentence.\n"
    "Sentence: {sentence}\n"
    "Candidates:\n"
    "0: negative\n"
    "1: neutral\n"
    "2: positive\n"
    "Answer with the number only. Answer:"
)


def build_fpb_dataset(tokenizer, split="train"):
    from datasets import load_dataset

    ds = load_dataset("FinanceInc/auditor_sentiment", split=split)

    def _fmt(ex):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": PROMPT_TMPL.format(sentence=ex["sentence"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return {"prompt": prompt, "gold": int(ex["label"])}

    return ds.map(_fmt, remove_columns=[c for c in ds.column_names if c != "gold"])


def digit_token_ids3(tokenizer):
    """3 类候选 token："0"~"2" 与 " 0"~" 2"（s-major，view(2,3) 聚合）。"""
    ids = []
    for s in ("", " "):
        for d in range(3):
            t = tokenizer.encode(s + str(d), add_special_tokens=False)
            ids.append(t[0])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=1688)  # 3377 train ÷ 2 prompt/步 ≈ 1 epoch
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lambda-brier", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--name", default="FPB-GRPO-RLCD-E1")
    args = parser.parse_args()

    setup_wandb_env(args.name, project="grpo-rlcd")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    train_ds = build_fpb_dataset(tokenizer, split="train")
    # 从 train 尾部切 300 条做 validation（官方 test 969 封存，终测一次）
    val_ds = train_ds.select(range(len(train_ds) - 300, len(train_ds)))
    train_ds = train_ds.select(range(len(train_ds) - 300))
    print(f"FPB train: {len(train_ds)} | val: {len(val_ds)} | G={args.num_generations} "
          f"| lam={args.lambda_brier}")

    digit_ids = digit_token_ids3(tokenizer)

    config = GRPOConfig(
        output_dir=str(EXP_DIR / "outputs" / args.name),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=8,
        learning_rate=2e-6,
        temperature=args.temperature,
        max_steps=args.max_steps,
        logging_steps=1,
        bf16=True,
        report_to="wandb",
        save_strategy="steps",
        save_steps=args.max_steps,
        save_total_limit=1,
        seed=42,
        reward_weights=[1.0],
        model_init_kwargs={"dtype": "bfloat16"},
    )

    def r_correct(prompts, completions, gold=None, **kwargs):
        out = []
        for c, g in zip(completions, gold):
            a = -1
            for ch in c.strip():
                if ch.isdigit() and 0 <= int(ch) <= 2:
                    a = int(ch)
                    break
            out.append(1.0 if a == g else 0.0)
        return out

    r_correct.__name__ = "r_correct"

    trainer = CalibratedGRPOTrainer(
        model=str(MODEL_PATH),
        reward_funcs=[r_correct],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.set_brier_cfg(args.lambda_brier, digit_ids, num_classes=3)
    trainer.train()


if __name__ == "__main__":
    main()
