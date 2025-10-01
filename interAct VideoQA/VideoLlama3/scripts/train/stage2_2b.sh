#!/bin/bash
# ============================
# Minimal Working 4-bit LoRA Script
# ============================

module load ffmpeg-6.0-gcc-12.1.0

# Minimal environment setup
export HF_HOME="/scratch/kkota3/huggingface_cache"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0  # Force single GPU

# Cleanup
pkill -f python 2>/dev/null || true
sleep 3

export PYTHONPATH="/scratch/kkota3/interact_videoqa/interAct VideoQA/VideoLlama3:$PYTHONPATH"

DATA_DIR="/scratch/kkota3/interact_videoqa/interAct VideoQA/VideoLlama3/data/InterAct_Video_Reasoning_Rich_Video_QA_for_Urban_Traffic"
OUTP_DIR="/scratch/kkota3/interact_videoqa/interAct VideoQA/VideoLlama3/videollama3_training_output_7b"

# Single GPU, no torchrun, minimal setup
python "/scratch/kkota3/interact_videoqa/interAct VideoQA/VideoLlama3/train.py" \
    --model_type videollama3_qwen2 \
    --model_path "${OUTP_DIR}/checkpoint-187" \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_attn_implementation eager \
    --mm_projector_type mlp2x_gelu \
    --data_path "${DATA_DIR}/training_data_filtered.jsonl" \
    --data_folder "${DATA_DIR}" \
    --image_merge_size 1 \
    --video_merge_size 1 \
    --fps 1 \
    --max_frames 2 \
    --model_max_length 256 \
    --mm_max_length 64 \
    --bf16 True \
    --fp16 False \
    --bits 4 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --double_quant True \
    --quant_type nf4 \
    --output_dir "${OUTP_DIR}/stage2_minimal" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --llm_lr 2e-4 \
    --mm_projector_lr 2e-4 \
    --vision_encoder_lr 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
    --use_batch_flattening False \
    --report_to "none" \
    --run_name "minimal_4bit"