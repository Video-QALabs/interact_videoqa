import argparse
import torch
import os
from pathlib import Path
from scripts.dataset import build_json
from scripts.inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="VideoLLaMA3 Dataset Builder / Inference / Training")
    parser.add_argument("--create_dataset", action="store_true", help="Run dataset building")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")

    # dataset args
    parser.add_argument("--annotations_csv", type=str, default="/data/annotations.csv")
    parser.add_argument("--source_video_dir", type=str, default="/data/train/1.43pm/video/")
    parser.add_argument("--out_dataset", type=str, default="/data/")

    args = parser.parse_args()

    print("Torch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        os.environ["DISABLE_FLASH_ATTN"] = "1"
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    if args.create_dataset:
        print("[INFO] Creating dataset...")
        build_json(
            annotations_csv=args.annotations_csv,
            source_video_dir=args.source_video_dir,
            out_root=args.out_dataset,
            use_absolute_paths=True,
            separate_conversations=True
        )

    elif args.inference:
        print("[INFO] Running inference...")
        run_inference()

    else:
        print("[ERROR] You must specify one of: --create_dataset, --inference, or --train")


if __name__ == "__main__":
    main()
