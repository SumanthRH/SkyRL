set -x

# Single-node, non-colocated GRPO for Qwen2.5-1.5B-Instruct on GSM8K with the delta
# weight-sync backend over the *disk* transport: the trainer (rank 0) writes versioned
# sparse-bf16 delta files to a shared directory and every inference worker reads them.
# This is the cross-DC / low-bandwidth carrier; on a single node the "shared FS" is just a
# local directory (SYNC_DIR below).
#
# Needs 2*NUM_GPUS GPUs on one node: NUM_GPUS training + NUM_GPUS inference engines.
#
# Usage:
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/gsm8k/run_gsm8k_delta_disk.sh

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout

: "${INFERENCE_BACKEND:=vllm}"

# Shared-filesystem directory the trainer writes deltas to and the inference engines read
# from. On a single node any local path works; for multi-node it must be on a shared FS
# reachable by both sides (a cloud URI like gs://... or s3://... also works). Writes are
# append-only: each sync writes a fresh weight_v{version} dir and nothing is deleted.
: "${SYNC_DIR:="/tmp/skyrl-delta-sync"}"

case "$SYNC_DIR" in
  s3://*|gs://*|az://*) ;;
  *) mkdir -p "$SYNC_DIR" ;;
esac

# Delta weight sync runs only on the new inference path (the receiver is vLLM's
# DeltaWeightTransferEngine). With the disk transport there is no NCCL group between trainer
# and inference; the first sync also skips shipping full weights (the inference engine
# reconstructs the bf16 base locally at load).
export _SKYRL_USE_NEW_INFERENCE=1

uv run --isolated --extra fsdp --extra gcp --env-file .env.ray -m skyrl.train.entrypoints.main_base \
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
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=delta \
  generator.inference_engine.delta_weight_sync_config.transport=disk \
  generator.inference_engine.delta_weight_sync_config.sync_dir="$SYNC_DIR" \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-delta-weight-sync" \
  trainer.run_name="gsm8k_delta_disk_single_node" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs-delta-disk" \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_delta_disk_ckpt" \
  $@
