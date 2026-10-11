#!/usr/bin/env bash
# 全量串行：exp1 → exp2(k=2,8,16) → exp3，wandb online，单个失败即停
set -e
PY=/home/jpzhao_team/zymo/my_env/bin/python
cd /home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export HF_HOME=/home/jpzhao_team/zymo/.hf_cache
mkdir -p logs/wandb

run() {
    name=$1; shift
    echo "=== $name START $(date '+%F %T') ==="
    if ! $PY "$@" > "logs/$name.log" 2>&1; then
        echo "=== $name FAILED $(date '+%F %T')，详见 logs/$name.log ==="
        exit 1
    fi
    echo "=== $name DONE $(date '+%F %T') ==="
}

run exp1_minimal_length   scripts/exp1_minimal_grpo.py
run exp2_mbpp_k2          scripts/exp2_mbpp_grpo.py --num-generations 2
run exp2_mbpp_k8          scripts/exp2_mbpp_grpo.py --num-generations 8
run exp2_mbpp_k16         scripts/exp2_mbpp_grpo.py --num-generations 16
run exp3_reward_hacking   scripts/exp3_reward_hacking.py
echo "ALL_DONE $(date '+%F %T')"
