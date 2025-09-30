#!/usr/bin/env python3
"""
Batch Inference Script for Fine-Tuned Qwen-VL2-7B Video Question Answering
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path

# Add parent directory to path to allow imports from scripts
sys.path.append(str(Path(__file__).parent.parent))

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from scripts.config import MODEL_PATH, DEFAULT_MAX_NEW_TOKENS, DEFAULT_NUM_BEAMS, DEFAULT_DO_SAMPLE

def load_video_frames(video_path, num_frames=8):
    """Load video frames for Qwen-VL processing"""
    import cv2
    import numpy as np
    from PIL import Image
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
    
    cap.release()
    return frames

def run_batch_inference(args):
    """Load the fine-tuned model and run inference on a dataset."""
    
    print("🚀 Starting Batch Inference with Fine-Tuned Qwen-VL Model")
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / "inference_results.json"
    
    # Load model and processor
    print(f"Loading model from: {args.model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Load annotations
    print(f"Loading annotations from: {args.annotations_path}")
    if not os.path.exists(args.annotations_path):
        print(f"❌ Error: Annotations file not found at {args.annotations_path}")
        return

    with open(args.annotations_path, 'r', encoding='utf-8') as f:
        if args.annotations_path.endswith('.json'):
            annotations = json.load(f)
        else:  # JSONL format
            annotations = [json.loads(line) for line in f if line.strip()]

    results = []
    total_annotations = len(annotations)
    
    # Process each item in the annotation file
    for i, item in enumerate(annotations, 1):
        # Skip metadata if present
        if isinstance(item, dict) and "custom_video_qa" in item:
            continue
            
        video_filename = item.get("video")
        conversations = item.get("conversations", [])
        
        if not video_filename or not conversations:
            continue

        # Extract question and ground_truth answer
        question = ""
        ground_truth = ""
        for conv in conversations:
            if conv.get('from') in ['human', 'user']:
                question = conv.get('value', '').replace('<video>', '').strip()
            elif conv.get('from') in ['gpt', 'assistant']:
                ground_truth = conv.get('value', '')
        
        if not question:
            continue

        video_path = os.path.join(args.videos_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"⚠️ Warning: Video file not found, skipping: {video_path}")
            continue

        print("-" * 80)
        print(f"Processing video {i}/{total_annotations}: {video_filename}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")

        try:
            # Load video frames
            frames = load_video_frames(video_path, num_frames=8)
            if not frames:
                print(f"❌ No frames extracted from {video_filename}")
                continue
            
            # Prepare messages for Qwen-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Apply chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    num_beams=DEFAULT_NUM_BEAMS,
                    do_sample=DEFAULT_DO_SAMPLE
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            print(f"Model Response (Predicted): {response}")

            # Store result
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
                    json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Error processing {video_filename}: {e}")

    # Save all results to a single JSON file at the end
    print(f"\n💾 Final Save: Saving {len(results)} results to {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("✅ Batch inference complete!")

def main():
    parser = argparse.ArgumentParser(description="Batch inference with fine-tuned Qwen-VL model.")
    parser.add_argument("--model_path", type=str, default="output/qwen_video_qa", help="Path to the fine-tuned model directory.")
    parser.add_argument("--annotations_path", type=str, required=True, help="Path to the JSON/JSONL file with questions.")
    parser.add_argument("--videos_dir", type=str, required=True, help="Directory containing the video files.")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save the inference results JSON file.")
    
    args = parser.parse_args()
    run_batch_inference(args)

if __name__ == "__main__":
    main()