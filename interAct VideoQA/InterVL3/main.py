#!/usr/bin/env python3
"""
Main entry point for the InterVL3 Video QA pipeline.
Provides CLI for training, data preparation, and inference.
"""

import argparse
import os
import sys
import subprocess

# Add the script's directory to the Python path to allow for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions with error handling
def train_with_lora(args):
    """Call the updated train_lora.py script"""
    import subprocess
    from pathlib import Path
    
    # Build command for the new training script
    script_path = Path(__file__).parent / "scripts" / "train_lora.py"
    
    cmd = [
        "python", str(script_path),
        "--model_name", args.model_name,
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--use_llm_lora", str(args.use_llm_lora),
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--max_seq_length", str(args.max_seq_length)
    ]
    
    if args.bf16:
        cmd.append("--bf16")
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    
    print("Running training command:")
    print(" ".join(cmd))
    
    result = subprocess.run(cmd)
    return result.returncode == 0

try:
    from scripts.prepare_data import prepare_data
except ImportError as e:
    print("❌ Failed to import 'prepare_data' from 'scripts.prepare_data'.")
    print(f"Original Error: {e}")
    # Define a placeholder that matches the expected signature to avoid TypeErrors
    def prepare_data(root_dir, output_file):
        print("Please manually prepare your data or fix the import error in scripts/prepare_data.py")
    # Re-raising is better for debugging
    raise

def run_inference_wrapper(args):
    """Wrapper to call the batch inference script"""
    import subprocess
    from pathlib import Path
    
    script_path = Path(__file__).parent / "scripts" / "inference.py"
    
    cmd = [
        "python", str(script_path),
        "--model_path", args.model_path,
        "--annotations_path", args.annotations_path,
        "--videos_dir", args.videos_dir
    ]
    
    print("Running batch inference command:")
    print(" ".join(cmd))
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    """Main function to handle CLI arguments."""
    
    parser = argparse.ArgumentParser(
        description="🔥 InterVL3 Video Chat - Complete Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Mode selection
    parser.add_argument("--train", action="store_true", help="🚀 Start LoRA fine-tuning.")
    parser.add_argument("--prepare_data", action="store_true", help="📊 Prepare the video QA dataset.")
    parser.add_argument("--inference", action="store_true", help="💬 Run batch inference on a dataset.")

    # Training arguments
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-38B")
    train_group.add_argument("--data_path", type=str, default="training_data/meta_config.json")
    train_group.add_argument("--output_dir", type=str, default="work_dirs/internvl3_38b_video_qa_lora")
    train_group.add_argument("--use_llm_lora", type=int, default=16)
    train_group.add_argument("--num_train_epochs", type=int, default=1)
    train_group.add_argument("--per_device_train_batch_size", type=int, default=1)
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=8)
    train_group.add_argument("--learning_rate", type=float, default=2e-5)
    train_group.add_argument("--max_seq_length", type=int, default=2048)
    train_group.add_argument("--bf16", action="store_true", default=True)
    train_group.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Inference arguments
    inference_group = parser.add_argument_group('Inference Arguments')
    inference_group.add_argument("--model_path", type=str, default="work_dirs/internvl3_38b_video_qa_lora", help="Path to the model directory.")
    inference_group.add_argument("--annotations_path", type=str, help="Path to the JSONL file with questions for inference.")
    inference_group.add_argument("--videos_dir", type=str, help="Directory containing video files for inference.")

    # Data preparation arguments
    data_prep_group = parser.add_argument_group('Data Preparation Arguments')
    data_prep_group.add_argument("--root_dir", type=str, default="data", help="Root directory containing the raw dataset to prepare.")
    data_prep_group.add_argument("--output_file", type=str, default="training_data/meta_config.json", help="Path for the output JSONL file.")

    args = parser.parse_args()
    
    print("==================================================")
    
    if args.train:
        print("🚀 Training Mode")
        print("🚀 Starting LoRA Fine-tuning...")
        if not train_with_lora(args):
            print("❌ Error during training.")
            sys.exit(1)
            
    elif args.prepare_data:
        print("📊 Data Preparation Mode")
        prepare_data(root_dir=args.root_dir, output_file=args.output_file)
        
    elif args.inference:
        print("💬 Inference Mode")
        if not args.annotations_path or not args.videos_dir:
            print("❌ Error: For inference, you must provide --annotations_path and --videos_dir.")
            parser.print_help()
            sys.exit(1)
        if not run_inference_wrapper(args):
            print("❌ Error during inference.")
            sys.exit(1)
            
    else:
        print("🤔 No mode selected. Use --train, --prepare_data, or --inference.")
        parser.print_help()

if __name__ == "__main__":
    main()

