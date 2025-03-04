import os

# Set environment variables.
os.environ["CUDA_HOME"] = "/packages/apps/nvhpc/247/Linux_x86_64/24.7/compilers"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configuration dictionary.
config_dict = {
    "max_epochs": 1,
    "learning_rate": 2e-5,
    "batch_size": 1,         # per-device batch size
    "num_workers": 4,
    "num_frames": 4,         # Use 4 frames per video
    "max_new_tokens": 64,
    "target_frame_size": (64, 64)  # Resize frames to (64,64)
}

