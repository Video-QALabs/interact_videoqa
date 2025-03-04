import argparse

def main():
    parser = argparse.ArgumentParser(description="VideoLlama2 Project")
    parser.add_argument(
        "--mode", 
        choices=["train", "inference"],
        default="train",
        help="Select 'train' to fine-tune the model or 'inference' to run inference."
    )
    args = parser.parse_args()

    if args.mode == "train":
        from train import main as train_main
        train_main()
    elif args.mode == "inference":
        from inference import main as inference_main
        inference_main()

if __name__ == "__main__":
    main()
