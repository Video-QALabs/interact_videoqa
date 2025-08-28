import argparse
from scripts.dataset import LlavaVideoQADataset
from scripts.model_setup import setup_llava_model
from scripts.train import train
from scripts.evaluate import evaluate

def main(args=None):
    parser = argparse.ArgumentParser(description="Llava-Next-Video for VideoQA")
    parser.add_argument("--mode", type=str, default="train",
                      help="Mode: train, eval, or inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                      help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to model checkpoint for evaluation/inference")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for training/evaluation")
    parser.add_argument("--num_epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    
    # Setup model and processor
    model, processor = setup_llava_model()
    
    if args.mode == "train":
        # Create training dataset
        train_dataset = LlavaVideoQADataset(
            video_dir="data/train/videos",
            csv_file="data/train/qa.csv",
            processor=processor
        )
        
        # Train the model
        train(
            model=model,
            train_dataset=train_dataset,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir="checkpoints"
        )
        
    elif args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("Checkpoint path is required for evaluation mode")
            
        # Create evaluation dataset
        eval_dataset = LlavaVideoQADataset(
            video_dir="data/eval/videos",
            csv_file="data/eval/qa.csv",
            processor=processor
        )
        
        # Evaluate the model
        evaluate(
            model=model,
            eval_dataset=eval_dataset,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size
        )
        
    elif args.mode == "inference":
        if args.checkpoint is None:
            raise ValueError("Checkpoint path is required for inference mode")
            
        # TODO: Add inference logic here
        pass

if __name__ == "__main__":
    main()
