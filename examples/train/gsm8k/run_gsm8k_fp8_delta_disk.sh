set -x

# FP8 quantized rollout with DELTA weight sync over the DISK transport.
# Non-colocated GRPO, Qwen2.5-1.5B-Instruct, GSM8K. Inference engine fp8; trainer bf16.
# Trainer rank 0 writes versioned sparse-bf16 delta files to SYNC_DIR; inference workers read
# them, merge onto the bf16 ShardShadow, then re-quantize to fp8. No NCCL group needed.
# Off-policy correction (sequence TIS) compensates the fp8 rollout / bf16 train mismatch.
# Needs 2*NUM_GPUS GPUs: NUM_GPUS training + NUM_GPUS inference engines.

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${SYNC_DIR:="/tmp/skyrl-delta-sync-fp8"}"

mkdir -p "$SYNC_DIR"

export _SKYRL_USE_NEW_INFERENCE=1

uv run --isolated --extra fsdp --env-file .env.ray -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.model_dtype=bfloat16 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.off_policy_correction.tis_ratio_type=sequence \
  trainer.algorithm.off_policy_correction.sequence_tis_ratio_clip_high=4.0 \
  generator.sampling_params.logprobs=0 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=delta \
  generator.inference_engine.delta_weight_sync_config.transport=disk \
  generator.inference_engine.delta_weight_sync_config.sync_dir="$SYNC_DIR" \
  generator.inference_engine.engine_init_kwargs.quantization=fp8 \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-delta-weight-sync" \
  trainer.run_name="gsm8k_fp8_delta_disk" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs-fp8-delta-disk" \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_fp8_delta_disk_ckpt" \
  $@
