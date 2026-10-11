#!/usr/bin/env bash
# GRPO 阶段（SFT 热启动版）：actor/ref = Megatron，rollout = vLLM。
# 与 run_grpo_qwen35_0.8b.sh 的唯一区别：model.path 指向 SFT 输出 latest_hf。
# 对比实验：先跑原始脚本（base 模型），再跑本脚本（SFT 热启动），看 reward 是否更快更高。
#
# 用法：bash grpo_qwen3.5_0.8b/scripts/run_grpo_from_sft_qwen3_0.6b.sh [额外 key=value 参数...]
# 例  ：.../run_grpo_from_sft_qwen3_0.6b.sh trainer.total_training_steps=8 trainer.save_freq=1000
cd "$(dirname "$0")"
set -x
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VERL_USE_UV=0

ROOT=/home/jpzhao_team/zymo/grpo_qwen3.5_0.8b
SFT_ROOT=/home/jpzhao_team/zymo/sft_qwen3_0.6b
VENV_PY=$HOME/zymo/verl_env/bin/python

MODEL_PATH=$SFT_ROOT/output/latest_hf #模型从SFT训练的输出加载的
TRAIN_FILES="[$ROOT/data/gsm8k/train.parquet]" #训练文件
VAL_FILES="[$ROOT/data/gsm8k/test.parquet]" #验证测试文件
LOG_DIR=$ROOT/logs/warmstart_run_$(date +%Y%m%d_%H%M%S) #日志存放
mkdir -p "$LOG_DIR"

"$VENV_PY" -m verl.trainer.main_ppo \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
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
    trainer.logger='["console"]' \
    trainer.project_name=verl_grpo_qwen35_0_8b \
    trainer.experiment_name=warmstart_gsm8k_qwen3_0_6b \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    model_engine=megatron \
    trainer.default_local_dir="$ROOT/checkpoints/warmstart_gsm8k_qwen3_0_6b" \
    trainer.resume_mode=disable \
    "$@" \
    hydra.run.dir="$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/train.log"
