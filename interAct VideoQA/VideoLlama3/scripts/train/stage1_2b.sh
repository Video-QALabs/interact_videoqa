#!/bin/bash
# ============================
# Stage 1 Training Script
# ============================

WORLD_SIZE=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-16667}
RANK=${RANK:-0}

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

GLOBAL_BATCH_SIZE=12
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (WORLD_SIZE * NPROC_PER_NODE * LOCAL_BATCH_SIZE)))
echo "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"

export WANDB_PROJECT=videollama3_qwen2.5_2b
RUN_NAME=stage1
DATA_DIR=""
OUTP_DIR=""
WEIGHTS_DIR=""

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
            /videollama3/train.py \
    --deepspeed '{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "zero_optimization": {
        "stage": 1,
        "overlap_comm": false,
        "contiguous_gradients": true
    }
}' \
    --model_type videollama3_qwen2 \
    --model_path /weights/videollama3_7b_local \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path ${DATA_DIR}/annotations.jsonl \
    --data_folder ${DATA_DIR} \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 180 \
    --model_max_length 16384 \
    --mm_max_length 10240 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir $OUTP_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --mm_projector_lr 1e-3 \
    --vision_encoder_lr 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --run_name $RUN_NAME
