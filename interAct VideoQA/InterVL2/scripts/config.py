# InterVL3 Configuration File
import os
from pathlib import Path

# Image normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model configuration
MODEL_PATH = "OpenGVLab/InternVL3-38B"
MODEL_NAME = "InternVL3-38B"

# Paths configuration
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = str(BASE_DIR / "cache")
DATA_DIR = str(BASE_DIR / "data")
OUTPUT_DIR = str(BASE_DIR / "outputs")

# Video processing configuration
DEFAULT_NUM_SEGMENTS = 8
DEFAULT_MAX_NUM_PATCHES = 6
DEFAULT_INPUT_SIZE = 448

# Training configuration
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_NUM_EPOCHS = 3
DEFAULT_MAX_SEQ_LENGTH = 2048

# LoRA configuration
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)