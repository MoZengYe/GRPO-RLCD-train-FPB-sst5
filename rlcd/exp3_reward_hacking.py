# -*- coding: utf-8 -*-
"""实验三：Reward Hacking——忠实复现作者 train_grpo_pathological.py。

被优化的奖励 hacked_reward：数 completion 里 ```python 代码块数，
min(块数, 4) / 4 归一化——四级跑道（0.25/0.5/0.75/1.0），不看代码对错。
真实奖励 real_reward：MBPP assert 通过率，reward_weights 设为 0（仅记录，不参与优化）。

预期现象（wandb，作者原文）：
    - 被优化的 reward 很快饱和（0.25→1.0 四级攀爬）；
    - 真实测试奖励持续下降（能力坍塌）；
    - completion length 卡在最大值（length gaming，4 个块需要大量 token）。

用法：
    python exp3_reward_hacking.py               # 完整 300 步
    python exp3_reward_hacking.py --max-steps 3  # 冒烟
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    EXP_DIR, MBPP_PROMPT_TMPL_AUTHOR, MODEL_PATH, build_mbpp_dataset,
    code_block_reward_fn, setup_wandb_env, tests_reward_fn,
)

from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf",
                        help="vllm=colocate 加速生成（需环境内有 vllm）")
    parser.add_argument("--name", default=None, help="覆盖 wandb run 名")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度；调高以增加采样多样性（飞轮点火实验用 1.0）")
    parser.add_argument("--vllm-is-correction", type=lambda x: str(x).lower() in ("1", "true", "yes"),
                        default=True,
                        help="vLLM 重要性采样修正开关；诊断实验用 False（怀疑其抑制长样本梯度）")
    args = parser.parse_args()

    setup_wandb_env(args.name or "exp3_reward_hacking_faithful")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    train_ds = build_mbpp_dataset(tokenizer, split="train", template=MBPP_PROMPT_TMPL_AUTHOR)

    engine_kwargs = {}
    if args.engine == "vllm":
        engine_kwargs = dict(use_vllm=True, vllm_mode="colocate",
                             vllm_gpu_memory_utilization=0.3)

    config = GRPOConfig(
        output_dir=str(EXP_DIR / "outputs" / "exp3_faithful"),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=512,
        learning_rate=2e-6,
        temperature=args.temperature,
        vllm_importance_sampling_correction=args.vllm_is_correction,
        max_steps=args.max_steps,
        logging_steps=1,
        bf16=True,
        report_to="wandb",
        save_strategy="no",
        seed=42,
        reward_weights=[1.0, 0.0],  # 真实奖励权重 0：只记录，不优化
        model_init_kwargs={"dtype": "bfloat16"},
        **engine_kwargs,
    )

    trainer = GRPOTrainer(
        model=str(MODEL_PATH),
        reward_funcs=[code_block_reward_fn, tests_reward_fn],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
