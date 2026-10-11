# -*- coding: utf-8 -*-
"""实验四（方案B）：RLCD 落地 GRPO —— 校准决策的最小可行版。

配方来自 Jev/RLCD 学习会话：
  r = w_correct * r_correct + w_cal * r_cal_action
  r_correct = (choice == gold)
  r_cal     = 1 - (p_a - y_a)^2      （按采样动作计分，不改 trainer——方案B）

数据：SetFit/sst5（句子情感 5 分类），候选 = 5 个情感类别。
Prompt 停在 "Answer:"，completion 期望单 token（候选编号 0-4）。

监控（weight=0 只记录不优化）：
  r_brier_monitor = 1 - 多类 Brier（完整分布的校准质量仪表）

用法：
  python exp4_rlcd_grpo.py                 # 300 步
  python exp4_rlcd_grpo.py --max-steps 3   # 冒烟
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
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


def build_sst5_dataset(tokenizer, n=2000):
    from datasets import load_dataset

    ds = load_dataset("SetFit/sst5", split="train").select(range(n))

    def _fmt(ex):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": PROMPT_TMPL.format(text=ex["text"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return {"prompt": prompt, "gold": int(ex["label"])}

    return ds.map(_fmt, remove_columns=[c for c in ds.column_names if c != "gold"])


def _parse_choice(completion: str):
    """从 completion 提取第一个数字 0-4；提取不到返回 -1。"""
    for ch in completion.strip():
        if ch.isdigit() and 0 <= int(ch) <= 4:
            return int(ch)
    return -1


def make_reward_fns():
    """方案B：r_correct + r_cal_action（都需要采样出的动作，共享解析）。"""
    state = {"parsed": []}

    def _both(prompts, completions, gold=None, **kwargs):
        choices = [_parse_choice(c) for c in completions]
        state["parsed"] = choices
        rs_correct, rs_cal = [], []
        for a, g in zip(choices, gold):
            rs_correct.append(1.0 if a == g else 0.0)
            # 方案B：只对采样动作计分校准（p_a=1 因为模型只输出了编号，
            # y_a = 1 if a==gold else 0）→ 退化为与 r_correct 同信号
            y_a = 1.0 if a == g else 0.0
            rs_cal.append(1.0 - (1.0 - y_a) ** 2)
        return rs_correct

    def _cal(prompts, completions, gold=None, **kwargs):
        choices = state["parsed"] if state["parsed"] else [_parse_choice(c) for c in completions]
        out = []
        for a, g in zip(choices, gold):
            y_a = 1.0 if a == g else 0.0
            out.append(1.0 - (1.0 - y_a) ** 2)
        return out

    _both.__name__ = "r_correct"
    _cal.__name__ = "r_cal_action"
    return _both, _cal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--w-correct", type=float, default=0.5)
    parser.add_argument("--w-cal", type=float, default=0.5)
    parser.add_argument("--name", default=None)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="生成温度；分类任务组内易零方差，默认 1.0")
    args = parser.parse_args()

    setup_wandb_env(args.name or f"exp4_rlcd_G{args.num_generations}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    train_ds = build_sst5_dataset(tokenizer)
    print(f"sst5 train prompts: {len(train_ds)}")

    r_correct, r_cal_action = make_reward_fns()

    config = GRPOConfig(
        output_dir=str(EXP_DIR / "outputs" / "exp4_rlcd"),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=8,          # 期望单 token，留一点余量
        learning_rate=2e-6,
        temperature=args.temperature,
        max_steps=args.max_steps,
        logging_steps=1,
        bf16=True,
        report_to="wandb",
        save_strategy="no",
        seed=42,
        reward_weights=[args.w_correct, args.w_cal],
        model_init_kwargs={"dtype": "bfloat16"},
    )

    trainer = GRPOTrainer(
        model=str(MODEL_PATH),
        reward_funcs=[r_correct, r_cal_action],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
