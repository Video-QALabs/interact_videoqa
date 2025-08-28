# Configuration file for LLaVA-Next-Video training

# Training configuration
TRAINING_CONFIG = {
    "lr": 5e-5,
    "max_epochs": 3,
    "accumulate_grad_batches": 2,
    "gradient_clip_val": 1.0,
    "batch_size": 1,
    "max_train_batches": 356,
    "max_eval_batches": 20,
    "eval_interval": 20,
    "warmup_steps": 100,
    "weight_decay": 0.01
}

# Model configuration
MODEL_CONFIG = {
    "model_id": "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "num_frames": 8,
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True
}

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "VL"
}

# Data paths
DATA_PATHS = {
    "train_csv": "./data/train/qa.csv",
    "eval_csv": "./data/eval/qa.csv",
    "train_video_dir": "./data/train/videos",
    "eval_video_dir": "./data/eval/videos",
    "checkpoint_dir": "checkpoints",
    "save_dir": "saved_models"
}

# Generation configuration
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": False,
    "num_beams": 3,
    "temperature": 0.7,
    "top_p": 0.9
}