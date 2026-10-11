# -*- coding: utf-8 -*-
"""实验一：最小化 GRPO——奖励 = completion 长度接近 20 字符。

观察点：completions/mean_length 在几十步内被拉向 20。
    "奖励函数返回什么，策略就优化什么。"

用法：
    python exp1_minimal_grpo.py                # 完整 300 步（wandb online）
    python exp1_minimal_grpo.py --max-steps 3  # 冒烟
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import EXP_DIR, MODEL_PATH, build_prompts_dataset_with_template, setup_wandb_env  # noqa: E402

from datasets import Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402


def reward_len(prompts, completions, **kwargs):
    """TRL quickstart 同款：长度离 20 越近奖励越高。"""
    return [-abs(len(completion) - 20) / 40 for completion in completions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf",
                        help="vllm=colocate 加速生成（需环境内有 vllm）")
    parser.add_argument("--lr", type=float, default=2e-6, help="作者配置 2e-6；TRL 默认 1e-6")
    parser.add_argument("--temperature", type=float, default=0.7, help="作者配置 0.7；TRL 默认 1.0")
    parser.add_argument("--max-completion-length", type=int, default=256,
                        help="作者配置 256；消融实验用 128 复现截断问题")
    parser.add_argument("--name", default=None, help="wandb run 名后缀（消融实验用）")
    args = parser.parse_args()

    setup_wandb_env(args.name or f"exp1_minimal_length_k{args.num_generations}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    raw_prompts = [f"Tell me something interesting about the number {i}." for i in range(1, 65)]
    train_ds = build_prompts_dataset_with_template(raw_prompts, tokenizer)

    engine_kwargs = {}
    if args.engine == "vllm":
        engine_kwargs = dict(use_vllm=True, vllm_mode="colocate",
                             vllm_gpu_memory_utilization=0.3)

    config = GRPOConfig(
        output_dir=str(EXP_DIR / "outputs" / "exp1"),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        temperature=args.temperature,
        max_steps=args.max_steps,
        logging_steps=1,
        bf16=True,
        report_to="wandb",
        save_strategy="no",
        seed=42,
        model_init_kwargs={"dtype": "bfloat16"},
        **engine_kwargs,
    )

    trainer = GRPOTrainer(
        model=str(MODEL_PATH),
        reward_funcs=[reward_len],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
