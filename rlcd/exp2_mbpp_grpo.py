# -*- coding: utf-8 -*-
"""实验二：可验证奖励——MBPP 代码生成，奖励 = 通过 assert 的比例。

扫组大小 k（num_generations）∈ {2, 8, 16}，复现文章结论：k=8 最优。
    k 大 → 组内 baseline 更稳，但每步覆盖的 prompt 少（= generation_batch_size / k）。

用法：
    python exp2_mbpp_grpo.py --num-generations 8              # 完整 300 步
    python exp2_mbpp_grpo.py --num-generations 2 --max-steps 3  # 冒烟
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    EXP_DIR, MBPP_PROMPT_TMPL_AUTHOR, MODEL_PATH, build_mbpp_dataset,
    make_tests_reward_fn, setup_wandb_env,
)

from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf",
                        help="vllm=colocate 加速生成（需环境内有 vllm）")
    parser.add_argument("--prompt-style", choices=["ours", "author"], default="ours",
                        help="author=作者的模板（含'只写代码块'指令）")
    parser.add_argument("--extract", choices=["join_all", "last_block"], default="join_all",
                        help="last_block=作者的代码提取策略（取最后一个代码块）")
    parser.add_argument("--name", default=None, help="覆盖 wandb run 名（复现对照实验用）")
    parser.add_argument("--reward-log", default=None,
                        help="把每步每个 (task_id, reward) 追加到该 CSV（逐题轨迹分析用）")
    parser.add_argument("--reward-style", choices=["pass_frac", "staged"], default="pass_frac",
                        help="staged=方案1阶梯奖励（0/0.1/0.2/Pn/1），CSV 会同时记录真实通过率")
    args = parser.parse_args()

    setup_wandb_env(args.name or f"exp2_mbpp_k{args.num_generations}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    template = MBPP_PROMPT_TMPL_AUTHOR if args.prompt_style == "author" else None
    train_ds = build_mbpp_dataset(tokenizer, split="train", template=template)
    print(f"MBPP train prompts: {len(train_ds)} | prompt_style={args.prompt_style} | extract={args.extract}")

    engine_kwargs = {}
    if args.engine == "vllm":
        engine_kwargs = dict(use_vllm=True, vllm_mode="colocate",
                             vllm_gpu_memory_utilization=0.3)

    config = GRPOConfig(
        output_dir=str(EXP_DIR / "outputs" / f"exp2_k{args.num_generations}"),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=512,
        learning_rate=2e-6,
        temperature=0.7,
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
        reward_funcs=[make_tests_reward_fn(args.extract,
                                           log_file=args.reward_log,
                                           staged=(args.reward_style == "staged"))],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
