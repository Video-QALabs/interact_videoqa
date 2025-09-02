import argparse
from scripts.train import train_videollama3
from scripts.inference import run_inference
from scripts.evaluate import evaluate_on_dataset
from scripts.config import DATA_PATHS

def main():
    parser = argparse.ArgumentParser(description="VideoLLaMA3 pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Training
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--train_csv", type=str, default=DATA_PATHS["train_csv"])
    p_train.add_argument("--video_dir", type=str, default=DATA_PATHS["train_video_dir"])
    p_train.add_argument("--output_dir", type=str, default="./outputs")

    # Inference
    p_infer = sub.add_parser("infer", help="Run inference on one video")
    p_infer.add_argument("--video", type=str, required=True)
    p_infer.add_argument("--prompt", type=str, required=True)

    # Evaluation
    p_eval = sub.add_parser("eval", help="Evaluate on a CSV file")
    p_eval.add_argument("--csv", type=str, default=DATA_PATHS["eval_csv"])
    p_eval.add_argument("--video_root", type=str, default=DATA_PATHS["eval_video_dir"])
    p_eval.add_argument("--out", type=str, default="eval_outputs.json")

    args = parser.parse_args()

    if args.cmd == "train":
        train_videollama3(
            train_csv=args.train_csv,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
        )
    elif args.cmd == "infer":
        print(run_inference(args.video, args.prompt))
    elif args.cmd == "eval":
        results = evaluate_on_dataset(
            eval_csv=args.csv,
            eval_video_dir=args.video_root,
            output_file=args.out,
        )
        print(f"Evaluation complete. Results saved to {args.out}")

if __name__ == "__main__":
    main()
