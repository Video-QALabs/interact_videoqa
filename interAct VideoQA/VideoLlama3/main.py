
import argparse, os
from scripts.train import train
from scripts.inference import generate
from scripts.evaluate import evaluate
from scripts.config import DATA

def main():
    parser = argparse.ArgumentParser(description="VideoLLaMA3 pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")

    p_infer = sub.add_parser("infer", help="Run inference on one video")
    p_infer.add_argument("--video", type=str, required=True)
    p_infer.add_argument("--prompt", type=str, required=True)

    p_eval = sub.add_parser("eval", help="Evaluate on a CSV file")
    p_eval.add_argument("--csv", type=str, default=DATA.eval_csv)
    p_eval.add_argument("--video_root", type=str, default=DATA.video_root)
    p_eval.add_argument("--out", type=str, default="eval_outputs.jsonl")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.resume)
    elif args.cmd == "infer":
        print(generate(args.video, args.prompt))
    elif args.cmd == "eval":
        path = evaluate(args.csv, args.video_root, args.out)
        print(f"Wrote {path}")

if __name__ == "__main__":
    main()
