#!/bin/bash

# Qwen2-VL LoRA Vision Fine-tuning for Video QA
# Adapted for your video dataset structure

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Get paths from environment variables or use defaults
DATA_PATH=${DATA_PATH:-"training_data/meta_config.json"}
VIDEO_FOLDER=${VIDEO_FOLDER:-"data"}
OUTPUT_DIR=${OUTPUT_DIR:-"output/qwen_video_qa"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
BATCH_SIZE=${BATCH_SIZE:-2}

# Set TRITON_CACHE_DIR to current working directory to avoid NFS issues
export TRITON_CACHE_DIR="$(pwd)/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"
echo "💾 Set TRITON_CACHE_DIR to: $TRITON_CACHE_DIR"

# Install missing dependencies
echo "📦 Installing exact dependencies from Qwen2-VL-Finetune requirements..."
pip uninstall trl -y --quiet
pip install trl==0.17.0 --quiet
pip install accelerate==1.10.1 --quiet
pip install peft --quiet
pip install liger-kernel --quiet
pip install deepspeed==0.17.5 --quiet
echo "✅ Dependencies installed with exact versions"

# Pre-download the model weights from HuggingFace
echo "📥 Pre-downloading Qwen2.5-VL-3B-Instruct model weights..."
export HF_HOME="/scratch/jnolas77/VideoQA/interact_videoqa/interAct VideoQA/Qwen-VL2-7B-hf/.hf_cache"
export TRANSFORMERS_CACHE="/scratch/jnolas77/VideoQA/interact_videoqa/interAct VideoQA/Qwen-VL2-7B-hf/.transformers_cache"
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
echo "💾 Model cache directory set to: $TRANSFORMERS_CACHE"

python -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import os
print('🔄 Downloading processor...')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir='$TRANSFORMERS_CACHE')
print('🔄 Downloading model weights...')
model = AutoModelForVision2Seq.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    dtype=torch.bfloat16,
    device_map='cpu',
    cache_dir='$TRANSFORMERS_CACHE',
    trust_remote_code=True
)
print('✅ Model downloaded successfully!')
print(f'📁 Model cached in: $TRANSFORMERS_CACHE')
del model, processor
torch.cuda.empty_cache()
"

# Change to the Qwen2-VL-Finetune directory and fix Python path
cd Qwen2-VL-Finetune

# Set proper PYTHONPATH from the Qwen2-VL-Finetune directory
export PYTHONPATH="$(pwd)/src:$(pwd):$PYTHONPATH"
echo "🐍 PYTHONPATH set to: $PYTHONPATH"

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=$BATCH_SIZE
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "🚀 Starting LoRA Vision Fine-tuning..."
echo "📊 Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Data: $DATA_PATH"
echo "  Videos: $VIDEO_FOLDER" 
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_PER_DEVICE"

# Convert relative paths to absolute paths from parent directory
DATA_PATH="../$DATA_PATH"
VIDEO_FOLDER="../$VIDEO_FOLDER"
OUTPUT_DIR="../$OUTPUT_DIR"

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, because the merger is included in the vision_tower.

deepspeed src/train/train_sft.py \
    --use_liger False \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id "$MODEL_NAME" \
        --data_path "/scratch/jnolas77/VideoQA/interact_videoqa/interAct VideoQA/Qwen-VL2-7B-hf/training_data/meta_config.json" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --nframes 10 \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --max_seq_length 2048 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 0

echo "✅ Training completed!"