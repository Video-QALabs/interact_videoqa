#!/usr/bin/env python3
"""
Main script for Qwen2.5-VL fine-tuning and inference with memory optimizations.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

# Import our fixed modules
from scripts.config import TRAINING_CONFIG, DATA_PATHS, MODEL_CONFIG, setup_memory_optimized_environment
from scripts.dataset import MemoryOptimizedQwenVideoQADataset, create_memory_efficient_dataloader
from scripts.model_setup import setup_qwen_model
from scripts.train import train_model
from scripts.evaluate import run_evaluation, quick_evaluation
from scripts.inference import memory_safe_inference, batch_inference, test_model_setup
from scripts.save_model import save_model, load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Video QA Training and Inference")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "eval", "inference", "test"],
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
    
    # Setup memory-optimized environment first
    setup_memory_optimized_environment()
    
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
    elif args.mode == "test":
        run_test_mode(args, device)

def run_training(args, device):
    """Run training mode with memory optimizations."""
    print("\n=== MEMORY-OPTIMIZED TRAINING MODE ===")
    
    # Check if data files exist
    if not os.path.exists(args.train_csv):
        print(f"ERROR: Training CSV not found: {args.train_csv}")
        return
    
    if not os.path.exists(args.train_video_dir):
        print(f"ERROR: Training video directory not found: {args.train_video_dir}")
        return
    
    # Setup model
    print("Setting up model...")
    model, processor, tokenizer = setup_qwen_model(checkpoint_path=args.checkpoint)
    
    # Test model setup first
    if not test_model_setup(model, processor, tokenizer, device):
        print("ERROR: Model setup test failed. Please check your configuration.")
        return
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = MemoryOptimizedQwenVideoQADataset(
        csv_file=args.train_csv,
        video_dir=args.train_video_dir,
        processor=processor,
        tokenizer=tokenizer
    )
    
    if args.max_samples:
        train_dataset.data = train_dataset.data.head(args.max_samples)
        print(f"Limited training to {args.max_samples} samples")
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Create memory-efficient data loader
    train_loader = create_memory_efficient_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Setup evaluation dataset if available
    eval_loader = None
    if os.path.exists(args.eval_csv) and os.path.exists(args.eval_video_dir):
        print("Loading evaluation dataset...")
        eval_dataset = MemoryOptimizedQwenVideoQADataset(
            csv_file=args.eval_csv,
            video_dir=args.eval_video_dir,
            processor=processor,
            tokenizer=tokenizer
        )
        
        if args.max_samples:
            eval_dataset.data = eval_dataset.data.head(min(args.max_samples // 4, 10))
        
        eval_loader = create_memory_efficient_dataloader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Train model
    print("Starting memory-optimized training...")
    try:
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
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Attempt emergency save
        try:
            emergency_dir = os.path.join(args.save_dir, "emergency_checkpoint")
            model.save_pretrained(emergency_dir)
            processor.save_pretrained(emergency_dir)
            print(f"Emergency checkpoint saved to: {emergency_dir}")
        except:
            print("Failed to save emergency checkpoint")

def run_evaluation_mode(args, device):
    """Run evaluation mode."""
    print("\n=== EVALUATION MODE ===")
    
    if not args.checkpoint and not os.path.exists(args.save_dir):
        print("ERROR: No checkpoint or saved model specified for evaluation")
        return
    
    checkpoint_path = args.checkpoint or args.save_dir
    
    try:
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
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

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
    
    try:
        # Setup model
        print("Loading model...")
        model, processor, tokenizer = setup_qwen_model(checkpoint_path=checkpoint_path)
        
        # Test model setup
        if not test_model_setup(model, processor, tokenizer, device):
            print("WARNING: Model setup test failed, but proceeding with inference...")
        
        # Run inference
        print(f"Running inference on: {args.video_path}")
        print(f"Question: {args.question}")
        
        answer = memory_safe_inference(
            model=model,
            processor=processor,
            video_path=args.video_path,
            question=args.question,
            device=device,
            tokenizer=tokenizer
        )
        
        print(f"\n=== RESULT ===")
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
        
        return answer
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

def run_test_mode(args, device):
    """Run test mode to verify setup."""
    print("\n=== TEST MODE ===")
    
    try:
        # Test 1: Basic model setup
        print("Testing basic model setup...")
        model, processor, tokenizer = setup_qwen_model(checkpoint_path=args.checkpoint)
        
        if test_model_setup(model, processor, tokenizer, device):
            print("✓ Basic model setup test passed")
        else:
            print("✗ Basic model setup test failed")
            return False
        
        # Test 2: Dataset loading
        if os.path.exists(args.train_csv) and os.path.exists(args.train_video_dir):
            print("Testing dataset loading...")
            test_dataset = MemoryOptimizedQwenVideoQADataset(
                csv_file=args.train_csv,
                video_dir=args.train_video_dir,
                processor=processor,
                tokenizer=tokenizer
            )
            
            # Limit to first few samples for testing
            test_dataset.data = test_dataset.data.head(3)
            print(f"✓ Dataset loaded successfully: {len(test_dataset)} samples")
            
            # Test dataloader
            test_loader = create_memory_efficient_dataloader(
                test_dataset,
                batch_size=1,
                shuffle=False
            )
            
            print("Testing dataloader...")
            for i, batch in enumerate(test_loader):
                if i >= 2:  # Test only first 2 batches
                    break
                print(f"✓ Batch {i+1} loaded successfully")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            
            print("✓ Dataloader test passed")
        else:
            print("⚠ Dataset test skipped (files not found)")
        
        # Test 3: Memory usage
        print("Testing memory usage...")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            print(f"Total GPU Memory: {max_memory:.2f}GB")
            
            if allocated < max_memory * 0.8:  # Less than 80% usage
                print("✓ Memory usage looks reasonable")
            else:
                print("⚠ High memory usage detected")
        
        print("\n=== TEST SUMMARY ===")
        print("✓ All basic tests completed")
        print("Your setup appears to be working correctly!")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()