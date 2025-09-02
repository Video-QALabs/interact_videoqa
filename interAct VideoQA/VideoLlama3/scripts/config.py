import os
import torch

# Check system compatibility for flash attention
def check_flash_attention_compatibility():
    """Check if flash attention is available and compatible."""
    try:
        import flash_attn
        return True
    except ImportError:
        print("Flash Attention not available - using default attention implementation")
        return False
    except Exception as e:
        print(f"Flash Attention compatibility issue: {e}")
        print("Falling back to default attention implementation")
        return False

# Configuration dictionary
config_dict = {
    "max_epochs": 1,
    "learning_rate": 2e-5,
    "batch_size": 1,  # per-device batch size
    "num_workers": 4,
    "num_frames": 8,  # Use 8 frames per video (VideoLLaMA3 recommended)
    "max_new_tokens": 64,
    "target_frame_size": None,  # VideoLLaMA3 handles frame sizing internally
}

# Training configuration
TRAINING_CONFIG = {
    "lr": 5e-5,
    "max_epochs": 1,
    "accumulate_grad_batches": 2,
    "gradient_clip_val": 1.0,
    "batch_size": 1,
    "max_train_batches": 356,
    "max_eval_batches": 20,
    "eval_interval": 20,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "save_steps": 100,
    "logging_steps": 1,
    "fp16": True,
}

# Check Flash Attention compatibility
FLASH_ATTENTION_AVAILABLE = check_flash_attention_compatibility()

# Model configuration for VideoLLaMA3
MODEL_CONFIG = {
    "model_id": "DAMO-NLP-SG/VideoLLaMA3-7B",
    "cache_dir": "",
    "num_frames": 8,
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "torch_dtype": "bfloat16",
    # Use flash attention only if available and compatible
    "attn_implementation": "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "eager",
}

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

# Data paths
DATA_PATHS = {
    "train_csv": "../qa.csv",
    "eval_csv": "../eval/qa.csv",
    "train_video_dir": "../data/1.43pm_10.1mins_clips_60",
    "eval_video_dir": "../eval/videos",
    "checkpoint_dir": "checkpoints",
    "save_dir": "saved_models",
    "data_root": "./datasets",
}

# Generation configuration
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": False,
    "num_beams": 3,
    "temperature": 0.7,
    "top_p": 0.9,
}

# Hardware configuration with compatibility checks
def get_device_config():
    """Get device configuration with compatibility checks."""
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        print("CUDA not available, using CPU")
        return "cpu"

HARDWARE_CONFIG = {
    "device": get_device_config(),
    "use_flash_attention": FLASH_ATTENTION_AVAILABLE,
    "use_deepspeed": False,
    "deepspeed_config": "scripts/zero2.json",
}

