#!/usr/bin/env bash
# GRPO + LoRA + wandb：8 步小实验（在原 GRPO 脚本基础上只加 LoRA 与 wandb）
#
# 用法：
#   bash ~/zymo/grpo_qwen3.5_0.8b/scripts/run_grpo_lora_wandb.sh
#   bash .../run_grpo_lora_wandb.sh actor_rollout_ref.model.lora.rank=32
#   WANDB_MODE=online WANDB_API_KEY=<你的key> bash .../run_grpo_lora_wandb.sh   # 用自己的账号上传
#
# 说明：
#   - 默认 wandb online：自动读 ~/zymo/.wandb_key（你自己的 key），上传到你自己的账号
#     想完全本地跑就加 WANDB_MODE=offline（曲线只写在 logs/wandb 下）
#   - 默认 LoRA rank=16 / alpha=32 / dropout=0.05 / merge=True，lr 放大到 1e-4（LoRA 惯例）
#   - 只跑 8 步，save_freq=1000（最后一步仍会存一次 checkpoint，跑完可删）
cd "$(dirname "$0")"
set -x
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VERL_USE_UV=0

# ---- wandb：key 与缓存都放在 zymo 内，不碰共享 home ----
export WANDB_MODE=${WANDB_MODE:-online}
if [ -f "$HOME/zymo/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/zymo/.wandb_key")
fi
ROOT=/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b
export WANDB_DIR=$ROOT/logs/wandb
export WANDB_CACHE_DIR=$ROOT/logs/wandb/cache
export WANDB_CONFIG_DIR=$ROOT/logs/wandb/config
export WANDB_DATA_DIR=$ROOT/logs/wandb/data

VENV_PY=$HOME/zymo/verl_env/bin/python
MODEL_PATH=$ROOT/models/Qwen3-0.6B
TRAIN_FILES="[$ROOT/data/gsm8k/train.parquet]"
VAL_FILES="[$ROOT/data/gsm8k/test.parquet]"
LOG_DIR=$ROOT/logs/lora_run_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

# checkpoint 目录默认按 experiment_name 分目录，避免两次实验互相覆盖
RUN_NAME=lora_gsm8k_qwen3_0_6b
for arg in "$@"; do
    case "$arg" in
        trainer.experiment_name=*) RUN_NAME="${arg#trainer.experiment_name=}" ;;
    esac
done
CKPT_DIR=${CKPT_DIR:-$ROOT/checkpoints/$RUN_NAME}

$VENV_PY -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_megatron_trainer \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.return_raw_chat=True \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.use_remove_padding=False \
    \
    actor_rollout_ref.model.lora.rank=16 \
    actor_rollout_ref.model.lora.alpha=32 \
    actor_rollout_ref.model.lora.dropout=0.05 \
    actor_rollout_ref.model.lora.merge=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.masked_softmax_fusion=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_grpo_lora_qwen3_0_6b \
    trainer.experiment_name=lora_r16_a32_lr1e-4_8steps \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=8 \
    model_engine=megatron \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.resume_mode=disable \
    "$@" \
    hydra.run.dir="$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/train.log"
