#!/usr/bin/env python3
"""
Inference Script for Fine-Tuned Qwen-VL2-7B Video Question Answering with LoRA
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path

# Add parent directory to path to allow imports from scripts
sys.path.append(str(Path(__file__).parent.parent))

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

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

def run_single_inference(args):
    """Load the fine-tuned LoRA model and run inference on a single video."""
    
    print("🚀 Starting Single Video Inference with Fine-Tuned Qwen-VL LoRA Model")
    
    # Load base model
    print(f"Loading base model: Qwen/Qwen2.5-VL-3B-Instruct")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Use sdpa instead of flash_attention_2
        device_map="auto"
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {args.model_path}")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    
    # Load processor from the fine-tuned model directory
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found at {args.video_path}")
        return

    print(f"Processing video: {args.video_path}")
    print(f"Question: {args.question}")

    try:
        # Load video frames (using 10 frames like in training)
        frames = load_video_frames(args.video_path, num_frames=10)
        if not frames:
            print(f"❌ No frames extracted from {args.video_path}")
            return
        
        print(f"✅ Extracted {len(frames)} frames from video")
        
        # Prepare messages for Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": args.question}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        print(f"Formatted prompt: {text[:200]}...")
        
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
        
        print("🤔 Generating response...")
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=0.7 if args.do_sample else None
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print("=" * 80)
        print(f"📹 Video: {os.path.basename(args.video_path)}")
        print(f"❓ Question: {args.question}")
        print(f"🤖 Model Response: {response}")
        print("=" * 80)
        
        # Save result if output path is provided
        if args.output_path:
            result = {
                "video_path": args.video_path,
                "question": args.question,
                "response": response,
                "model_path": args.model_path
            }
            
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"💾 Result saved to: {args.output_path}")

    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Single video inference with fine-tuned Qwen-VL LoRA model.")
    parser.add_argument("--model_path", type=str, default="../output/qwen_video_qa", 
                       help="Path to the fine-tuned LoRA model directory.")
    parser.add_argument("--video_path", type=str, required=True, 
                       help="Path to the video file.")
    parser.add_argument("--question", type=str, required=True, 
                       help="Question to ask about the video.")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save the inference result JSON file.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate.")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search.")
    parser.add_argument("--do_sample", action="store_true",
                       help="Whether to use sampling for generation.")
    
    args = parser.parse_args()
    run_single_inference(args)

if __name__ == "__main__":
    main()