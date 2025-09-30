#!/usr/bin/env python3
"""
Main entry point for the Qwen-VL2-7B-hf Video QA pipeline.
Provides CLI for training, data preparation, evaluation, and inference.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add the script's directory to the Python path to allow for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_prepare_data(args):
    """Wrapper to call the data preparation script"""
    print("📊 Data Preparation Mode")
    try:
        from scripts.prepare_data import prepare_data
        prepare_data(root_dir=args.root_dir, output_file=args.data_path)
    except ImportError:
        print("❌ Failed to import prepare_data. Creating script...")
        create_prepare_data_script()
        from scripts.prepare_data import prepare_data
        prepare_data(root_dir=args.root_dir, output_file=args.data_path)

def run_training(args):
    """Wrapper to call the official LoRA vision training script"""
    print("🚀 Training Mode")
    print("🚀 Starting LoRA Vision Fine-tuning...")
    
    # Validate required arguments
    if not args.data_path or not Path(args.data_path).exists():
        print(f"❌ Error: Training data file not found: {args.data_path}")
        print("💡 Tip: Run 'python main.py --prepare_data' first to prepare your data.")
        sys.exit(1)
    
    if not args.videos_dir or not Path(args.videos_dir).exists():
        print(f"❌ Error: Video directory not found: {args.videos_dir}")
        sys.exit(1)
    
    # Set environment variables for the script
    env = os.environ.copy()
    env["DATA_PATH"] = str(args.data_path)
    env["VIDEO_FOLDER"] = str(args.videos_dir)
    env["OUTPUT_DIR"] = str(args.output_dir)
    env["NUM_EPOCHS"] = str(args.num_train_epochs)
    env["LEARNING_RATE"] = str(args.learning_rate)
    env["BATCH_SIZE"] = str(args.per_device_train_batch_size)
    
    script_path = Path(__file__).parent / "scripts" / "train_lora_vision.sh"
    
    # Use bash to run the shell script
    if os.name == 'nt':  # Windows
        cmd = ["bash", str(script_path)]
    else:  # Linux/Unix
        cmd = ["bash", str(script_path)]
    
    print(f"Running LoRA vision training with:")
    print(f"  Data: {args.data_path}")
    print(f"  Videos: {args.videos_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.per_device_train_batch_size}")
    
    result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("❌ Error during training.")
        sys.exit(1)
    print("✅ Training completed successfully!")
    return True

def run_evaluation(args):
    """Wrapper to call the evaluation script"""
    print("📈 Evaluation Mode")
    script_path = Path(__file__).parent / "scripts" / "evaluate.py"
    
    cmd = [
        "python", str(script_path),
        "--mode", "eval",
        "--checkpoint", args.output_dir,
        "--eval_csv", args.annotations_path or args.data_path,
        "--eval_video_dir", args.videos_dir
    ]
    
    print("Running evaluation command:")
    print(" ".join(cmd))
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Error during evaluation.")
        sys.exit(1)
    print("✅ Evaluation completed successfully!")
    return True

def run_inference(args):
    """Wrapper to call the batch inference script"""
    print("💬 Inference Mode")
    script_path = Path(__file__).parent / "scripts" / "inference.py"
    
    cmd = [
        "python", str(script_path),
        "--mode", "inference",
        "--checkpoint", args.output_dir,
        "--eval_csv", args.annotations_path,
        "--eval_video_dir", args.videos_dir
    ]
    
    print("Running batch inference command:")
    print(" ".join(cmd))
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Error during inference.")
        sys.exit(1)
    print("✅ Inference completed successfully!")
    return True

def run_demo(args):
    """Wrapper to start the interactive demo"""
    print("🤖 Demo Mode")
    script_path = Path(__file__).parent / "scripts" / "inference.py"
    
    cmd = [
        "python", str(script_path),
        "--mode", "test",
        "--checkpoint", args.output_dir
    ]
    
    print("Starting interactive demo...")
    result = subprocess.run(cmd)
    return True

def create_prepare_data_script():
    """Create the prepare_data.py script if it doesn't exist"""
    prepare_data_content = '''import os
import pandas as pd
import json
from pathlib import Path
import argparse

def prepare_data(root_dir='data', output_file='training_data/meta_config.json'):
    """
    Prepares video QA data from multiple subdirectories containing annotations.csv
    and converts it into a single JSONL file for training.
    """
    print(f"🔍 Starting data preparation from root directory: {root_dir}")
    
    all_annotations = []
    
    # Ensure the output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Walk through all subdirectories to find annotations.csv
    for dirpath, _, filenames in os.walk(root_dir):
        if 'annotations.csv' in filenames:
            annotations_file = os.path.join(dirpath, 'annotations.csv')
            print(f"Processing annotations from: {annotations_file}")
            
            try:
                df = pd.read_csv(annotations_file)
                
                # Process each row in the CSV
                for _, row in df.iterrows():
                    video_file = row.get('video_filename') or row.get('filename') or row.get('video_file_path')
                    question = row.get('question')
                    answer = row.get('answer')

                    if not all([video_file, question, answer]):
                        continue

                    # Skip files that don't have the correct extension
                    if not isinstance(video_file, str) or not video_file.endswith('.mp4'):
                        continue
                    
                    # Get the relative path of the video from the root_dir
                    relative_video_path = os.path.relpath(os.path.join(dirpath, video_file), root_dir)

                    # Create the JSONL structure for Qwen-VL format
                    json_record = {
                        "id": f"{Path(dirpath).name}_{video_file.replace('.mp4', '')}",
                        "conversations": [
                            {"from": "user", "value": f"<video>\\n{question}"},
                            {"from": "assistant", "value": answer}
                        ],
                        "video": relative_video_path.replace('\\\\', '/')
                    }
                    all_annotations.append(json_record)
                    
            except Exception as e:
                print(f"❌ Error processing {annotations_file}: {e}")

    # Write all annotations to the output file
    if all_annotations:
        print(f"Writing {len(all_annotations)} records to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, indent=2)
        print("✅ Data preparation complete!")
    else:
        print("🤷 No annotations found. The output file was not created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare video QA data.")
    parser.add_argument('--root_dir', default='data', help='Root directory of the dataset')
    parser.add_argument('--output_file', default='training_data/meta_config.json', help='Output JSON file for training config')
    args = parser.parse_args()
    
    prepare_data(args.root_dir, args.output_file)
'''
    
    script_path = Path(__file__).parent / "scripts" / "prepare_data.py"
    with open(script_path, 'w') as f:
        f.write(prepare_data_content)
    print(f"✅ Created prepare_data.py at {script_path}")

def main():
    """Main function to handle CLI arguments."""
    
    parser = argparse.ArgumentParser(
        description="🔥 Qwen-VL2-7B Video Chat - Complete Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Mode selection
    parser.add_argument("--prepare_data", action="store_true", help="📊 Prepare the video QA dataset.")
    parser.add_argument("--train", action="store_true", help="🚀 Start LoRA fine-tuning.")
    parser.add_argument("--evaluate", action="store_true", help="📈 Evaluate the fine-tuned model.")
    parser.add_argument("--inference", action="store_true", help="💬 Run batch inference on a dataset.")
    parser.add_argument("--pipeline", action="store_true", help="⚙️ Run the complete pipeline: prepare → train → evaluate.")
    parser.add_argument("--demo", action="store_true", help="🤖 Start interactive chat demo.")

    # Data preparation arguments
    data_prep_group = parser.add_argument_group('Data Preparation Arguments')
    data_prep_group.add_argument("--root_dir", type=str, default="data", help="Root directory containing the raw dataset to prepare.")
    data_prep_group.add_argument("--data_path", type=str, default="training_data/meta_config.json", help="Path for the prepared data file.")

    # Model and training arguments
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model name.")
    train_group.add_argument("--output_dir", type=str, default="output/qwen_video_qa", help="Output directory for trained model.")
    train_group.add_argument("--deepspeed_config", type=str, default="scripts/zero3_offload.json", help="DeepSpeed configuration file.")
    train_group.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    train_group.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device.")
    train_group.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    train_group.add_argument("--videos_dir", type=str, default="data", help="Directory containing video files.")

    # Inference arguments
    inference_group = parser.add_argument_group('Inference Arguments')
    inference_group.add_argument("--annotations_path", type=str, help="Path to the JSONL file with questions for inference.")

    args = parser.parse_args()
    
    print("==================================================")
    
    if args.pipeline:
        print("⚙️ Pipeline Mode")
        run_prepare_data(args)
        run_training(args)
        run_evaluation(args)
        print("✅ Complete pipeline finished successfully!")
        
    elif args.prepare_data:
        run_prepare_data(args)
        
    elif args.train:
        run_training(args)
        
    elif args.evaluate:
        run_evaluation(args)
        
    elif args.inference:
        if not args.annotations_path or not args.videos_dir:
            print("❌ Error: For inference, you must provide --annotations_path and --videos_dir.")
            parser.print_help()
            sys.exit(1)
        run_inference(args)
        
    elif args.demo:
        run_demo(args)
        
    else:
        print("🤔 No mode selected. Use --train, --prepare_data, --evaluate, --inference, --pipeline, or --demo.")
        parser.print_help()

if __name__ == "__main__":
    main()