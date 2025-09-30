# Qwen-VL2-7B-hf Configuration File
import os
from pathlib import Path

# Image normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model configuration
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME = "Qwen2.5-VL-3B-Instruct"

# Paths configuration
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = str(BASE_DIR / "cache")
DATA_DIR = str(BASE_DIR / "data")
OUTPUT_DIR = str(BASE_DIR / "output")
TRAINING_DATA_DIR = str(BASE_DIR / "training_data")
INFERENCE_RESULTS_DIR = str(BASE_DIR / "inference_results")

# Video processing configuration
DEFAULT_NUM_SEGMENTS = 8
DEFAULT_FPS = 1.0
DEFAULT_VIDEO_MAX_PIXELS = 360 * 420

# Training configuration
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_NUM_EPOCHS = 1

# Inference configuration
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_NUM_BEAMS = 1
DEFAULT_DO_SAMPLE = False

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(INFERENCE_RESULTS_DIR, exist_ok=True)