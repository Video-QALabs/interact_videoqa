import argparse
from scripts.model import setup_videollama_model
from scripts.train import train
from scripts.inference import inference
from scripts.dataset import VideoLlamaDataset

def main(args=None):
    parser = argparse.ArgumentParser(description="VideoLlama2 Project")
    parser.add_argument(
        "--mode", 
        choices=["train", "eval", "inference"],
        default="train",
        help="Select mode: train, eval, or inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation/inference"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training/evaluation"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # Setup model and processor
    model, processor = setup_videollama_model()
    
    if args.mode == "train":
        # Create training dataset
        train_dataset = VideoLlamaDataset(
            video_dir="data/train/videos",
            csv_file="data/train/qa.csv",
            processor=processor
        )
        
        # Train the model
        train(
            model=model,
            train_dataset=train_dataset,
            processor=processor,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
    
    elif args.mode in ["eval", "inference"]:
        if args.checkpoint is None:
            raise ValueError("Checkpoint path is required for evaluation/inference mode")
        
        inference(
            model=model,
            processor=processor,
            checkpoint_path=args.checkpoint
        )

if __name__ == "__main__":
    main()
    main()
