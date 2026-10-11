#!/usr/bin/env bash
# vLLM 加速版串行：exp2(k=2,8,16) → exp3，跑在 verl_env（有 vllm）+ GPU 2
# exp1 仍由原 my_env 链路在 GPU 1 上跑完，互不干扰
set -e
PY=/home/jpzhao_team/zymo/verl_env/bin/python
cd /home/jpzhao_team/zymo/grpo_qwen3.5_0.8b/trl_experiments
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
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

run exp1_minimal_length   scripts/exp1_minimal_grpo.py --engine vllm
run exp2_mbpp_k2   scripts/exp2_mbpp_grpo.py --num-generations 2  --engine vllm
run exp2_mbpp_k8   scripts/exp2_mbpp_grpo.py --num-generations 8  --engine vllm
run exp2_mbpp_k16  scripts/exp2_mbpp_grpo.py --num-generations 16 --engine vllm
run exp3_reward_hacking scripts/exp3_reward_hacking.py --engine vllm
echo "ALL_DONE $(date '+%F %T')"
