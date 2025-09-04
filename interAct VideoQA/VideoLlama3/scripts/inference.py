import os
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

def run_inference(csv_file_path="../data/eval/eval.csv",
             video_folder="../data/eval/videos",
             output_file="predictions.json"):

    device = "cuda:0"
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"

    # Load model & processor
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Load annotations
    df = pd.read_csv(csv_file_path)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_file = os.path.join(video_folder, row["video_file_path"])
        question = row["question"]
        gt_answer = row["answer"]

        if not os.path.exists(video_file):
            print(f"⚠️ Missing video: {video_file}")
            prediction = "[Video not found]"
        else:
            # Build conversation
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": {"video_path": video_file, "fps": 1, "max_frames": 180}},
                        {"type": "text", "text": question},
                    ],
                },
            ]

            # Prepare inputs
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            # Generate prediction
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=256)
                prediction = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        results.append({
            "index": int(row["index"]),
            "video_file": row["video_file_path"],
            "question": question,
            "ground_truth": gt_answer,
            "prediction": prediction
        })

    # Save results as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved predictions to {output_file}")