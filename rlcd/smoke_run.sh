#!/usr/bin/env bash
# 冒烟测试：每个实验 3 步，wandb offline，验证链路能跑通
set -e
PY=/home/jpzhao_team/zymo/my_env/bin/python
cd /home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export WANDB_MODE=offline
export HF_HOME=/home/jpzhao_team/zymo/.hf_cache

echo "=== smoke exp1 $(date +%T) ==="
$PY scripts/exp1_minimal_grpo.py --max-steps 3
echo "=== smoke exp2 k=2 $(date +%T) ==="
$PY scripts/exp2_mbpp_grpo.py --num-generations 2 --max-steps 3
echo "=== smoke exp3 $(date +%T) ==="
$PY scripts/exp3_reward_hacking.py --max-steps 3
echo "SMOKE_ALL_OK $(date +%T)"
