# Configuration file for Qwen2.5-VL training with memory optimization

# Training configuration - MEMORY OPTIMIZED
TRAINING_CONFIG = {
    "lr": 5e-5,
    "max_epochs": 1,
    "accumulate_grad_batches": 8,  # Increased to compensate for smaller batch size
    "gradient_clip_val": 1.0,
    "batch_size": 1,  # Keep at 1 for video processing
    "max_train_batches": 50,  # Reduced for testing
    "max_eval_batches": 10,  # Reduced for memory
    "eval_interval": 10,  # More frequent for debugging
    "logging_steps": 5,  # Add logging frequency
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "fp16": True,  # Enable mixed precision
    "gradient_checkpointing": True,  # Enable gradient checkpointing
}

# Model configuration - MEMORY OPTIMIZED
MODEL_CONFIG = {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "cache_dir": "/scratch/kvinod/VideoQA/interact_videoqa/interAct VideoQA/Llava-Next-Video/cache",
    "max_pixels": 120 * 140,  # FURTHER REDUCED for memory safety
    "fps": 0.3,  # Further reduced FPS for fewer frames
    "max_frames": 3,  # Further reduced frames
    "max_new_tokens": 32,  # Reduced for training
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "video_max_length": 20,  # Limit video length in seconds
    "image_size": 168,  # Smaller image size for memory
}

# LoRA configuration - MEMORY OPTIMIZED
LORA_CONFIG = {
    "r": 8,  # Reduced from 16
    "lora_alpha": 16,  # Reduced accordingly
    "lora_dropout": 0.1,  # Slightly higher dropout
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

# Data paths
DATA_PATHS = {
    "train_csv": "../qa.csv",
    "eval_csv": "../eval/qa.csv",
    "train_video_dir": "../data/1.43pm_10.1mins_clips_60",
    "eval_video_dir": "../eval/videos",
    "checkpoint_dir": "checkpoints",
    "save_dir": "saved_models"
}

# Generation configuration - MEMORY OPTIMIZED
GENERATION_CONFIG = {
    "max_new_tokens": 32,  # Reduced for memory
    "do_sample": False,  # Deterministic for consistency
    "num_beams": 1,  # No beam search for memory
    "temperature": 0.7,
    "top_p": 0.9
}

# Memory optimization settings
MEMORY_CONFIG = {
    "empty_cache_frequency": 5,  # More frequent cache clearing
    "max_memory_per_gpu": "10GB",  # Limit GPU memory usage
    "offload_to_cpu": True,  # Offload when possible
    "use_8bit": True,  # Use 8-bit optimization
    "dataloader_num_workers": 0,  # Avoid multiprocessing overhead
    "pin_memory": False,  # Disable pin memory to save GPU memory
}

# Environment setup for memory optimization
def setup_memory_optimized_environment():
    """Setup environment variables for memory optimization."""
    import os
    import torch
    
    # PyTorch memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Only enable for debugging
    
    # Disable some features that use extra memory
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Set memory fraction if needed
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory max
    
    print("✓ Memory-optimized environment configured")

# Memory monitoring utility
class MemoryMonitor:
    """Simple memory monitoring utility."""
    
    def __init__(self):
        self.peak_memory_gb = 0.0
        self.oom_events = 0
        self.successful_steps = 0
        self.total_steps = 0
    
    def update(self, allocated_gb, reserved_gb):
        """Update memory statistics."""
        self.peak_memory_gb = max(self.peak_memory_gb, allocated_gb)
    
    def log_oom(self):
        """Log an OOM event."""
        self.oom_events += 1
    
    def log_success(self):
        """Log a successful step."""
        self.successful_steps += 1
        self.total_steps += 1
    
    def log_failure(self):
        """Log a failed step."""
        self.total_steps += 1
    
    def get_stats(self):
        """Get memory statistics."""
        success_rate = self.successful_steps / self.total_steps if self.total_steps > 0 else 0
        return {
            "peak_memory_gb": self.peak_memory_gb,
            "oom_events": self.oom_events,
            "success_rate": success_rate,
            "successful_steps": self.successful_steps,
            "total_steps": self.total_steps
        }