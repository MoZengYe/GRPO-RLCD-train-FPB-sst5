# -*- coding: utf-8 -*-
"""实验五（方案A 完整版）：RLCD 校准——Brier 进 loss + 新 wandb 项目。

配方（Jev/RLCD 会话）：
  - GRPO reward 只留 r_correct（候选编号 == gold）
  - 多类 Brier 作为可微辅助损失直接加进 loss：total = grpo_loss + λ·Brier
    （校准信号不依赖组内方差——方案B 的 r_cal 会被组内减均值消掉，方案A 修复）

关键实现：CalibratedGRPOTrainer 覆写两个方法
  1. _generate_and_score_completions：把数据集列 gold 注回训练 inputs
     （TRL 的训练 inputs 不携带数据集额外列——gold 只流向奖励函数 kwargs，
      直接在 compute_loss 里读会 KeyError: 'gold'，冒烟已踩坑；
      且 dataloader 按"行字典列表"批处理，需兼容 list 形态）
  2. compute_loss：super() 算 GRPO 项 + 二次前向取首个 completion token
     位置的候选 logits（"0"~"4" 与 " 0"~" 4" 共 10 变体）→ softmax →
     聚合空格变体成 5 维分布 → 多类 Brier vs one-hot(gold) → λ·Brier

wandb：新项目 grpo-rlcd（与 grpo-trl-replica 的曲线分开）

用法：
  python exp5_rlcd_calibrated.py                 # 300 步
  python exp5_rlcd_calibrated.py --max-steps 3   # 冒烟
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from common import EXP_DIR, MODEL_PATH, setup_wandb_env  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

CANDIDATES = ["very negative", "negative", "neutral", "positive", "very positive"]

PROMPT_TMPL = (
    "Classify the sentiment of the movie review.\n"
    "Review: {text}\n"
    "Candidates:\n"
    "0: very negative\n"
    "1: negative\n"
    "2: neutral\n"
    "3: positive\n"
    "4: very positive\n"
    "Answer with the number only. Answer:"
)


def build_sst5_dataset(tokenizer, n=2000, split="train"):
    from datasets import load_dataset

    ds = load_dataset("SetFit/sst5", split=split)

    def _fmt(ex):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": PROMPT_TMPL.format(text=ex["text"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return {"prompt": prompt, "gold": int(ex["label"])}

    ds = ds.map(_fmt, remove_columns=[c for c in ds.column_names if c != "gold"])
    return ds.select(range(min(n, len(ds))))

    def _fmt(ex):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": PROMPT_TMPL.format(text=ex["text"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return {"prompt": prompt, "gold": int(ex["label"])}

    return ds.map(_fmt, remove_columns=[c for c in ds.column_names if c != "gold"])


def digit_token_ids(tokenizer):
    """候选 token："0"~"4" 与 " 0"~" 4"（s-major 排列，便于 view(2,5) 聚合）。"""
    ids = []
    for s in ("", " "):
        for d in range(5):
            t = tokenizer.encode(s + str(d), add_special_tokens=False)
            ids.append(t[0])  # 数字应为单 token
    return ids


class CalibratedGRPOTrainer(GRPOTrainer):
    """GRPO loss + λ·多类Brier（首个 completion token 位置的候选分布校准）。"""

    def set_brier_cfg(self, lambda_brier, digit_ids, num_classes=5):
        self.lambda_brier = lambda_brier
        self._num_classes = num_classes
        self._digit_ids_cpu = torch.tensor(digit_ids)
        self._digit_ids = None  # 惰性搬到设备

    def _digit_ids_dev(self, device):
        if self._digit_ids is None or self._digit_ids.device != device:
            self._digit_ids = self._digit_ids_cpu.to(device)
        return self._digit_ids

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        # 数据集额外列（gold）只会传给奖励函数 kwargs，不会进训练 inputs——
        # 在这里注回去，供 compute_loss 的 Brier 使用。
        # dataloader 按"行字典列表"批处理，需兼容 list 形态。
        if isinstance(inputs, list):
            gold = [row.get("gold") for row in inputs]
        else:
            gold = inputs.get("gold")
        if gold is not None:
            out["gold"] = torch.tensor([int(g) for g in gold],
                                       device=out["completion_ids"].device)
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        grpo_loss = super().compute_loss(model, inputs, return_outputs=False,
                                         num_items_in_batch=num_items_in_batch)
        if self.lambda_brier <= 0:
            return grpo_loss
        full_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attn = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        logits = model(input_ids=full_ids, attention_mask=attn,
                       logits_to_keep=inputs["completion_ids"].size(1) + 1).logits  # (B, C+1, V)
        first_logits = logits[:, 0, :].float()               # 预测首个 completion token
        digit_ids = self._digit_ids_dev(first_logits.device)
        digit_logits = first_logits[:, digit_ids]            # (B, 2*C)
        probs = torch.softmax(digit_logits, dim=-1)
        C = self._num_classes
        probsC = probs.view(probs.size(0), 2, C).sum(dim=1)  # 聚合空格变体 → (B, C)
        gold = inputs["gold"]
        gold = gold.to(probsC.device).long()
        onehot = torch.nn.functional.one_hot(gold, num_classes=C).float()
        brier = ((probsC - onehot) ** 2).sum(dim=-1).mean()  # 多类 Brier ∈ [0,2]
        self._last_brier = float(brier.item())
        return grpo_loss + self.lambda_brier * brier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--lambda-brier", type=float, default=0.5)
    parser.add_argument("--name", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="每步 completion 总数，必须 >= num_generations（G 扫描用 32/64）")
    parser.add_argument("--n-train", type=int, default=2000,
                        help="sst5 train 取用条数（数据规模实验用 4000）")
    args = parser.parse_args()

    run_name = args.name or f"exp5_calibrated_lam{args.lambda_brier}"
    setup_wandb_env(run_name, project="grpo-rlcd")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    train_ds = build_sst5_dataset(tokenizer, n=args.n_train)
    print(f"sst5 train prompts: {len(train_ds)} | lambda_brier={args.lambda_brier} "
          f"| G={args.num_generations} | batch={args.batch_size}")

    config = GRPOConfig(
        output_dir=str(EXP_DIR / "outputs" / run_name),   # 按实验名区分，防 checkpoint 覆盖
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
        save_strategy="steps",            # 训练结束存一次 checkpoint（eval 用）
        save_steps=args.max_steps,
        save_total_limit=1,
        seed=42,
        reward_weights=[1.0],             # GRPO 只优化 r_correct
        model_init_kwargs={"dtype": "bfloat16"},
    )

    def r_correct(prompts, completions, gold=None, **kwargs):
        out = []
        for c, g in zip(completions, gold):
            a = -1
            for ch in c.strip():
                if ch.isdigit() and 0 <= int(ch) <= 4:
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
    trainer.set_brier_cfg(args.lambda_brier, digit_token_ids(tokenizer))
    trainer.train()


if __name__ == "__main__":
    main()
