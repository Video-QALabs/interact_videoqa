#!/bin/bash
set -x

# Official InterVL3-38B Fine-tuning Script with LoRA
# Based on: https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html
#
# USAGE EXAMPLES:
# 1. Set your partition and run:
#    export PARTITION=your_gpu_partition && bash scripts/fine_tune.sh
# 
# 2. Custom configuration:
#    PARTITION=gpu_v100 GPUS_PER_NODE=8 PER_DEVICE_BATCH_SIZE=2 bash scripts/fine_tune.sh
#
# 3. Single GPU (no SLURM):
#    GPUS_PER_NODE=1 bash scripts/fine_tune.sh

# SLURM Configuration (if using SLURM)
PARTITION=${PARTITION:-'your_partition_name'}  # ⚠️ SET YOUR ACTUAL PARTITION NAME HERE
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # For InterVL3-38B: use 8-16 GPUs
NODES=${NODES:-1}
GPUS=$((NODES * GPUS_PER_NODE))

# Validate partition name
if [ "$PARTITION" = "your_partition_name" ]; then
  echo "⚠️ WARNING: Please set your actual SLURM partition name!"
  echo "   Export PARTITION=your_actual_partition before running"
  echo "   Example: export PARTITION=gpu && bash scripts/fine_tune.sh"
fi

# Training Configuration
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# Model and Data Configuration
MODEL_PATH="./pretrained/InternVL3-38B"  # Local model path as per official docs
HUGGINGFACE_MODEL="OpenGVLab/InternVL3-38B"  # Fallback to HuggingFace
OUTPUT_DIR='work_dirs/internvl3_38b_video_qa_lora'
DATA_DIR="$(pwd)/training_data"

# Create directories
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

if [ ! -d "$DATA_DIR" ]; then
  mkdir -p "$DATA_DIR"
fi

echo "================================================"
echo "InterVL3-38B Video QA Fine-tuning (Official)"
echo "================================================"
echo "Partition: $PARTITION"
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $GPUS"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Per device batch: $PER_DEVICE_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACC"
echo "================================================"

# Check if model exists locally, otherwise use HuggingFace
if [ ! -d "$MODEL_PATH" ]; then
  echo "⚠️ Local model not found at $MODEL_PATH"
  echo "📥 Using HuggingFace model: $HUGGINGFACE_MODEL"
  echo "💡 To download locally, run: bash scripts/download_models.sh"
  MODEL_PATH="$HUGGINGFACE_MODEL"
fi

# Prepare data if not exists
if [ ! -f "$DATA_DIR/meta_config.json" ]; then
  echo "📊 Preparing training data..."
  python main.py --prepare_data
  if [ $? -ne 0 ]; then
    echo "❌ Error: Data preparation failed"
    exit 1
  fi
fi

# Check if using SLURM
if command -v srun &> /dev/null; then
  echo "🚀 Running with SLURM on partition: $PARTITION"
  echo "   GPUs per node: $GPUS_PER_NODE"
  echo "   Total nodes: $NODES"
  
  # Official SLURM command for InterVL3-38B
  LAUNCHER_CMD="srun -p $PARTITION \
    --gres=gpu:$GPUS_PER_NODE \
    --ntasks=$NODES \
    --ntasks-per-node=1 \
    --cpus-per-task=32 \
    --mem=0 \
    --time=24:00:00 \
    --job-name=internvl3_finetune"
    
elif command -v torchrun &> /dev/null; then
  echo "🚀 Running with torchrun (local multi-GPU)..."
  LAUNCHER_CMD="torchrun \
    --nnodes=$NODES \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT"
else
  echo "❌ Neither srun nor torchrun found!"
  echo "   Please install torch distributed or use SLURM"
  exit 1
fi

# Official InterVL3-38B training command with LoRA
echo "🏋️ Starting InterVL3-38B LoRA fine-tuning..."
echo "   LoRA Rank: 16, Alpha: 32, Dropout: 0.1"
echo "   Frozen: LLM=True, MLP=True, Backbone=True"
echo "   Using partition: $PARTITION with $GPUS GPUs"

$LAUNCHER_CMD python scripts/train_lora.py \
  --model_name_or_path "$MODEL_PATH" \
  --conv_style "Hermes-2" \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$DATA_DIR/meta_config.json" \
  --overwrite_output_dir \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm \
  --freeze_mlp \
  --freeze_backbone \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACC \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train \
  --grad_checkpoint \
  --group_by_length \
  --dynamic_image_size \
  --use_thumbnail \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
  echo "Model saved to: $OUTPUT_DIR"
  echo "Training log: ${OUTPUT_DIR}/training_log.txt"
else
  echo "Training failed!"
  exit 1
fi

# Optional: Run validation
echo "Training completed. You can now run inference with:"
echo "python main.py --model_path $OUTPUT_DIR"