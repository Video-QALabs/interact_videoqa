#!/usr/bin/env python3
"""
Batch Inference Script for Fine-Tuned InternVL3-38B Video Question Answering
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path

# Add parent directory to path to allow imports from scripts
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModel
from scripts.dataset import load_video

def run_batch_inference(args):
    """Load the fine-tuned model and run inference on a dataset."""
    
    print("🚀 Starting Batch Inference with Fine-Tuned InterVL3 Model")
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / "inference_results.json"
    
    # Load tokenizer and model
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto'
    ).eval()

    # Load annotations
    print(f"Loading annotations from: {args.annotations_path}")
    if not os.path.exists(args.annotations_path):
        print(f"❌ Error: Annotations file not found at {args.annotations_path}")
        return

    with open(args.annotations_path, 'r', encoding='utf-8') as f:
        annotations = [json.loads(line) for line in f if line.strip()]

    results = []
    # Process each item in the annotation file, with an index for checkpointing
    for i, item in enumerate(annotations, 1):
        if "custom_video_qa" in item:
            continue  # Skip metadata line

        video_filename = item.get("video")
        conversations = item.get("conversations", [])
        
        if not video_filename or not conversations:
            continue

        # Extract question and ground_truth answer
        question = ""
        ground_truth = ""
        for conv in conversations:
            if conv.get('from') == 'human':
                question = conv.get('value', '').replace('<video>', '').strip()
            elif conv.get('from') == 'gpt':
                ground_truth = conv.get('value', '')
        
        if not question:
            continue

        video_path = os.path.join(args.videos_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"⚠️ Warning: Video file not found, skipping: {video_path}")
            continue

        print("-" * 80)
        print(f"Processing video {i}/{len(annotations)}: {video_filename}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")

        try:
            # Load and process video
            pixel_values, num_patches_list = load_video(video_path, num_segments=8)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            # Create video prefix for multi-frame input
            video_prefix = ''.join([f'Frame{j+1}: <image>\n' for j in range(len(num_patches_list))])
            full_question = video_prefix + question

            # Run inference
            with torch.no_grad():
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    full_question,
                    generation_config={
                        "max_new_tokens": 512,
                        "do_sample": False,
                        "num_beams": 1,
                    },
                    num_patches_list=num_patches_list,
                    history=[],
                    return_history=True
                )
            
            print(f"Model Response (Predicted): {response}")

            # Store result with the new keys
            results.append({
                "folder_name": os.path.dirname(video_filename),
                "file_name": os.path.basename(video_filename),
                "question": question,
                "ground_truth": ground_truth,
                "predicted": response
            })

            # Checkpoint saving every 100 results
            if i % 100 == 0:
                print(f"\n💾 Checkpoint: Saving {len(results)} results to {output_file_path}...")
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)

        except Exception as e:
            print(f"❌ Error processing {video_filename}: {e}")

    # Save all results to a single JSON file at the end
    print(f"\n💾 Final Save: Saving {len(results)} results to {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print("✅ Batch inference complete!")

def main():
    parser = argparse.ArgumentParser(description="Batch inference with fine-tuned InternVL3 model.")
    parser.add_argument("--model_path", type=str, default="work_dirs/internvl3_38b_video_qa_lora", help="Path to the fine-tuned model directory.")
    parser.add_argument("--annotations_path", type=str, required=True, help="Path to the JSONL file with questions.")
    parser.add_argument("--videos_dir", type=str, required=True, help="Directory containing the video files.")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save the inference results JSON file.")
    
    args = parser.parse_args()
    run_batch_inference(args)

if __name__ == "__main__":
    main()




