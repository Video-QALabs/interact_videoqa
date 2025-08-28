#!/usr/bin/env python3
"""
Main script for LLaVA-Next-Video fine-tuning and inference.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

# Import our modules
from config import TRAINING_CONFIG, DATA_PATHS, MODEL_CONFIG
from dataset import LlavaVideoQADataset
from model_setup import setup_llava_model
from train import train_model
from evaluate import run_evaluation, quick_evaluation
from inference import inference_single_video, batch_inference
from save_model import save_model, load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="LLaVA-Next-Video Training and Inference")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "eval", "inference", "demo"],
                       help="Operation mode")
    
    # Data paths
    parser.add_argument("--train_csv", type=str, default=DATA_PATHS["train_csv"],
                       help="Path to training CSV file")
    parser.add_argument("--eval_csv", type=str, default=DATA_PATHS["eval_csv"],
                       help="Path to evaluation CSV file")
    parser.add_argument("--train_video_dir", type=str, default=DATA_PATHS["train_video_dir"],
                       help="Directory containing training videos")
    parser.add_argument("--eval_video_dir", type=str, default=DATA_PATHS["eval_video_dir"],
                       help="Directory containing evaluation videos")
    
    # Model paths
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--save_dir", type=str, default=DATA_PATHS["save_dir"],
                       help="Directory to save trained model")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG["batch_size"],
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["max_epochs"],
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=TRAINING_CONFIG["lr"],
                       help="Learning rate")
    
    # Inference parameters
    parser.add_argument("--video_path", type=str, default=None,
                       help="Path to video for single inference")
    parser.add_argument("--question", type=str, default=None,
                       help="Question for single inference")
    
    # Other options
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    
    # Execute based on mode
    if args.mode == "train":
        run_training(args, device)
    elif args.mode == "eval":
        run_evaluation_mode(args, device)
    elif args.mode == "inference":
        run_inference_mode(args, device)
    elif args.mode == "demo":
        run_demo_mode(args, device)

def run_training(args, device):
    """Run training mode."""
    print("\n=== TRAINING MODE ===")
    
    # Check if data files exist
    if not os.path.exists(args.train_csv):
        print(f"ERROR: Training CSV not found: {args.train_csv}")
        return
    
    if not os.path.exists(args.train_video_dir):
        print(f"ERROR: Training video directory not found: {args.train_video_dir}")
        return
    
    # Setup model
    print("Setting up model...")
    model, processor = setup_llava_model(checkpoint_path=args.checkpoint)
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = LlavaVideoQADataset(
        csv_file=args.train_csv,
        video_dir=args.train_video_dir,
        processor=processor
    )
    
    if args.max_samples:
        train_dataset.data = train_dataset.data.head(args.max_samples)
        print(f"Limited training to {args.max_samples} samples")
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Setup evaluation dataset if available
    eval_loader = None
    if os.path.exists(args.eval_csv) and os.path.exists(args.eval_video_dir):
        print("Loading evaluation dataset...")
        eval_dataset = LlavaVideoQADataset(
            csv_file=args.eval_csv,
            video_dir=args.eval_video_dir,
            processor=processor
        )
        
        if args.max_samples:
            eval_dataset.data = eval_dataset.data.head(min(args.max_samples // 4, 50))
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
            num_workers=0
        )
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Train model
    print("Starting training...")
    trained_model = train_model(
        model=model,
        processor=processor,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device
    )
    
    # Save model
    print("Saving trained model...")
    save_model(trained_model, processor, args.save_dir)
    
    print(f"✓ Training complete! Model saved to: {args.save_dir}")

def run_evaluation_mode(args, device):
    """Run evaluation mode."""
    print("\n=== EVALUATION MODE ===")
    
    if not args.checkpoint and not os.path.exists(args.save_dir):
        print("ERROR: No checkpoint or saved model specified for evaluation")
        return
    
    checkpoint_path = args.checkpoint or args.save_dir
    
    # Run comprehensive evaluation
    results = run_evaluation(
        checkpoint_path=checkpoint_path,
        eval_csv=args.eval_csv,
        eval_video_dir=args.eval_video_dir,
        batch_size=args.batch_size,
        max_eval_samples=args.max_samples
    )
    
    print("✓ Evaluation complete!")
    return results

def run_inference_mode(args, device):
    """Run inference mode."""
    print("\n=== INFERENCE MODE ===")
    
    if not args.checkpoint and not os.path.exists(args.save_dir):
        print("ERROR: No checkpoint or saved model specified for inference")
        return
    
    if not args.video_path or not args.question:
        print("ERROR: Both --video_path and --question are required for inference mode")
        return
    
    if not os.path.exists(args.video_path):
        print(f"ERROR: Video file not found: {args.video_path}")
        return
    
    checkpoint_path = args.checkpoint or args.save_dir
    
    # Setup model
    print("Loading model...")
    model, processor = setup_llava_model(checkpoint_path=checkpoint_path)
    
    # Run inference
    print(f"Running inference on: {args.video_path}")
    print(f"Question: {args.question}")
    
    answer = inference_single_video(
        model=model,
        processor=processor,
        video_path=args.video_path,
        question=args.question,
        device=device
    )
    
    print(f"\n=== RESULT ===")
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")
    
    return answer

def run_demo_mode(args, device):
    """Run demo mode with interactive input."""
    print("\n=== DEMO MODE ===")
    
    if not args.checkpoint and not os.path.exists(args.save_dir):
        print("ERROR: No checkpoint or saved model specified for demo")
        return
    
    checkpoint_path = args.checkpoint or args.save_dir
    
    # Setup model
    print("Loading model...")
    model, processor = setup_llava_model(checkpoint_path=checkpoint_path)
    print("✓ Model loaded successfully!")
    
    print("\nDemo mode ready! Enter video paths and questions.")
    print("Type 'quit' to exit.")
    
    while True:
        try:
            # Get video path
            video_path = input("\nEnter video path (or 'quit'): ").strip()
            if video_path.lower() == 'quit':
                break
            
            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {video_path}")
                continue
            
            # Get question
            question = input("Enter your question: ").strip()
            if not question:
                print("Error: Please enter a question")
                continue
            
            # Run inference
            print("Processing...")
            answer = inference_single_video(
                model=model,
                processor=processor,
                video_path=video_path,
                question=question,
                device=device
            )
            
            print(f"\n--- RESULT ---")
            print(f"Video: {os.path.basename(video_path)}")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("Demo ended.")

if __name__ == "__main__":
    main()