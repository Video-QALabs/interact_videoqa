import torch
from tqdm import tqdm
import os
import cv2
from PIL import Image
import numpy as np
from scripts.config import GENERATION_CONFIG, MODEL_CONFIG
from qwen_vl_utils import process_vision_info
import os

def _to_file_uri(p: str) -> str:
    """Return a file:// URI for local paths, keep existing URIs unchanged."""
    if p.startswith("http://") or p.startswith("https://") or p.startswith("file://"):
        return p
    abs_path = os.path.abspath(p)
  
    return f"file:///{abs_path}"

def extract_video_frames(video_path: str, max_frames: int = 3, image_size: int = 168) -> list:
    """Extract frames from video with aggressive memory optimization."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames very conservatively
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(i) for i in np.linspace(0, total_frames-1, max_frames)]
        
        frames = []
        for idx in frame_indices[:max_frames]:  # Ensure we don't exceed max_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Aggressively resize
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame)
        
        cap.release()
        
        # Pad with last frame if needed
        while len(frames) < max_frames:
            if frames:
                frames.append(frames[-1])
            else:
                dummy = Image.new('RGB', (image_size, image_size), color='black')
                frames.append(dummy)
        
        return frames[:max_frames]
        
    except Exception as e:
        print(f"Frame extraction error: {e}")
        # Return dummy frames
        dummy = Image.new('RGB', (image_size, image_size), color='black')
        return [dummy] * max_frames

def memory_safe_inference(model, processor, video_path, question, device, tokenizer=None):
    """
    Memory-safe inference for Qwen2.5-VL with multiple fallback approaches.
    """
    model.eval()
    
    if tokenizer is None:
        tokenizer = getattr(processor, 'tokenizer', processor)
    
    try:
        # Method 1: Direct video processing (preferred for Qwen2.5-VL)
        print("Attempting direct video processing...")
        result = inference_single_video(model, processor, video_path, question, device, tokenizer)
        if result and not result.startswith("Error:"):
            return result
    except Exception as e:
        print(f"Direct video processing failed: {e}")
    
    try:
        # Method 2: Extracted frames approach
        print("Attempting frame extraction approach...")
        result = inference_with_extracted_frames(model, processor, video_path, question, device, tokenizer)
        if result and not result.startswith("Error:"):
            return result
    except Exception as e:
        print(f"Frame extraction failed: {e}")
    
    try:
        # Method 3: Text-only fallback
        print("Using text-only fallback...")
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"I cannot process the video file, but the question was: {question}. Please provide a general response."
                    }
                ]
            }
        ]
        
        text_prompt = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
            
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            
            response = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
            
        return f"[Video processing unavailable] {response}"
        
    except Exception as e:
        return f"Error: All inference methods failed - {e}"
    finally:
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
def inference_single_video(model, processor, video_path, question, device, tokenizer):
    """
    Run inference on a single video using Qwen2.5-VL's recommended pipeline:
    messages -> apply_chat_template -> process_vision_info -> processor(...) -> generate.
    """
    try:
        # Ensure videos are passed as a proper URI (or remote URL)
        video_uri = _to_file_uri(video_path)

        # 1) Build messages (Qwen2.5-VL style)
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_uri,
                    # Carry your memory-safe constraints via message fields
                    "max_pixels": MODEL_CONFIG.get("max_pixels", 120 * 140),
                    "fps":        MODEL_CONFIG.get("fps", 0.3),
                },
                {"type": "text", "text": question},
            ],
        }]

        # 2) Convert messages to a text prompt
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 3) Turn the visual parts into model-ready tensors/kwargs
        #    This step is CRUCIAL to avoid "Incorrect format used for video" + arange() errors.
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        # 4) Build inputs with the processor (single call)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        # Move to device safely
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # 5) Generate (keep use_cache=False if you’re being extra careful with memory)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=GENERATION_CONFIG.get("max_new_tokens", 32),
                do_sample=GENERATION_CONFIG.get("do_sample", False),
                num_beams=GENERATION_CONFIG.get("num_beams", 1),
                temperature=GENERATION_CONFIG.get("temperature", 0.7) if GENERATION_CONFIG.get("do_sample", False) else None,
                top_p=GENERATION_CONFIG.get("top_p", 0.9) if GENERATION_CONFIG.get("do_sample", False) else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

            # Trim the input prompt portion (per model-card pattern)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            text_out = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        return text_out.strip()

    except RuntimeError as e:
        # Memory-friendly handling
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "Error: Out of memory during video processing"
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"
def inference_with_extracted_frames(model, processor, video_path, question, device, tokenizer):
    """
    Alternative method: extract a few frames, save them as tiny JPEGs, and feed them
    as a 'video' = list of image URIs. This uses the same recommended pipeline to
    avoid format/type issues.
    """
    try:
        print("Using extracted frames approach...")

        # 0) Extract small frames (you already have extract_video_frames configured aggressively)
        video_frames = extract_video_frames(
            video_path,
            max_frames=MODEL_CONFIG.get("max_frames", 3),
            image_size=MODEL_CONFIG.get("image_size", 168),
        )

        # 1) Persist frames as small JPEGs and turn into file:// URIs
        tmp_paths = []
        for i, frame in enumerate(video_frames):
            # Keep size small; quality 70–80 is usually fine for VLMs
            tmp_file = f"/tmp/qwen_frame_{os.getpid()}_{i}.jpg"
            try:
                frame.save(tmp_file, format="JPEG", quality=75, optimize=True)
            except Exception:
                # Fallback without optimize if PIL complains
                frame.save(tmp_file, format="JPEG", quality=75)
            tmp_paths.append(_to_file_uri(tmp_file))

        # 2) Build messages that describe a "video" composed of these frames
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": tmp_paths},  # <= list-of-images as a video
                {"type": "text", "text": f"These {len(tmp_paths)} frames are from a video. {question}"},
            ],
        }]

        # 3) Template + 4) process_vision_info
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # 5) Pack inputs
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # 6) Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=GENERATION_CONFIG.get("max_new_tokens", 32),
                do_sample=False,
                num_beams=GENERATION_CONFIG.get("num_beams", 1),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            text_out = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        return text_out.strip()

    except Exception as e:
        print(f"Frame extraction method failed: {e}")
        return f"Error: Frame processing failed - {e}"

def batch_inference(model, processor, eval_dataset, device, batch_size=1, tokenizer=None):
    """
    Run inference on a batch of videos with improved error handling.
    """
    model.eval()
    
    if tokenizer is None:
        tokenizer = getattr(processor, 'tokenizer', processor)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running Inference")):
            try:
                # Clear cache periodically
                if batch_idx % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Validate batch structure
                if not batch or "input_ids" not in batch:
                    print(f"Invalid batch {batch_idx}: missing required data")
                    continue
                
                # Move batch to device (exclude ground_truth)
                gen_inputs = {}
                for k, v in batch.items():
                    if k == "ground_truth":
                        continue
                    if isinstance(v, torch.Tensor):
                        gen_inputs[k] = v.to(device)
                    else:
                        gen_inputs[k] = v
                
                # Validate we have minimum required inputs
                if "input_ids" not in gen_inputs or "attention_mask" not in gen_inputs:
                    print(f"Batch {batch_idx}: missing required inputs")
                    # Add dummy predictions
                    batch_size_actual = len(batch.get("ground_truth", [""]))
                    predictions.extend(["Error: Invalid input"] * batch_size_actual)
                    ground_truths.extend(batch.get("ground_truth", ["N/A"] * batch_size_actual))
                    continue
                
                # Generate predictions with memory constraints
                try:
                    output_ids = model.generate(
                        **gen_inputs,
                        max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                        do_sample=False,  # Deterministic for evaluation
                        num_beams=GENERATION_CONFIG.get("num_beams", 1),
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False  # Disable cache for memory
                    )
                    
                    # Get only the generated text (remove input prompt)
                    input_len = batch["input_ids"].shape[1]
                    generated_ids = output_ids[:, input_len:]
                    
                    # Decode predictions
                    pred_texts = tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    
                    # Clean and store predictions
                    cleaned_preds = [pred.strip() for pred in pred_texts]
                    predictions.extend(cleaned_preds)
                    ground_truths.extend(batch.get("ground_truth", ["N/A"] * len(pred_texts)))
                    
                    # Show progress for first few batches
                    if batch_idx < 3:
                        print(f"\nBatch {batch_idx + 1} sample:")
                        print(f"  Prediction: '{cleaned_preds[0] if cleaned_preds else 'Empty'}'")
                        print(f"  Ground Truth: '{batch.get('ground_truth', ['N/A'])[0]}'")
                    
                except RuntimeError as gen_e:
                    if "out of memory" in str(gen_e).lower():
                        print(f"OOM during generation at batch {batch_idx}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        print(f"Generation error at batch {batch_idx}: {gen_e}")
                    
                    # Add error predictions
                    batch_size_actual = len(batch.get("ground_truth", [""]))
                    predictions.extend(["Error: Generation failed"] * batch_size_actual)
                    ground_truths.extend(batch.get("ground_truth", ["N/A"] * batch_size_actual))
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                # Add placeholder predictions for this batch
                batch_size_actual = len(batch.get("ground_truth", [""]))
                predictions.extend(["Error: Processing failed"] * batch_size_actual)
                ground_truths.extend(batch.get("ground_truth", ["N/A"] * batch_size_actual))
    
    print(f"\nInference complete: {len(predictions)} predictions generated")
    return predictions, ground_truths

def evaluate_predictions(predictions, ground_truths):
    """
    Evaluate the quality of predictions with detailed analysis.
    """
    print("\n=== Evaluation Results ===")
    print(f"Total samples: {len(predictions)}")
    
    # Print sample predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(predictions))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prediction: '{predictions[i]}'")
        print(f"Ground Truth: '{ground_truths[i]}'")
        
        # Simple similarity check
        if ground_truths[i] != "N/A":
            pred_words = set(predictions[i].lower().split())
            gt_words = set(ground_truths[i].lower().split())
            if pred_words and gt_words:
                overlap = len(pred_words & gt_words) / len(pred_words | gt_words)
                print(f"Word overlap: {overlap:.2%}")
    
    # Calculate statistics
    pred_lengths = [len(pred.split()) for pred in predictions]
    gt_lengths = [len(gt.split()) for gt in ground_truths if gt != "N/A"]
    
    empty_predictions = sum(1 for pred in predictions if len(pred.strip()) == 0)
    error_predictions = sum(1 for pred in predictions if pred.startswith("Error:"))
    valid_predictions = len(predictions) - empty_predictions - error_predictions
    
    print(f"\n=== Statistics ===")
    print(f"Valid predictions: {valid_predictions} ({valid_predictions/len(predictions)*100:.1f}%)")
    print(f"Empty predictions: {empty_predictions} ({empty_predictions/len(predictions)*100:.1f}%)")
    print(f"Error predictions: {error_predictions} ({error_predictions/len(predictions)*100:.1f}%)")
    
    if pred_lengths:
        print(f"Average prediction length: {sum(pred_lengths)/len(pred_lengths):.2f} words")
        print(f"Prediction length range: {min(pred_lengths)} - {max(pred_lengths)} words")
    
    if gt_lengths:
        print(f"Average ground truth length: {sum(gt_lengths)/len(gt_lengths):.2f} words")
    
    return {
        "total_samples": len(predictions),
        "valid_predictions": valid_predictions,
        "empty_predictions": empty_predictions,
        "error_predictions": error_predictions,
        "avg_pred_length": sum(pred_lengths)/len(pred_lengths) if pred_lengths else 0,
        "avg_gt_length": sum(gt_lengths)/len(gt_lengths) if gt_lengths else 0,
        "success_rate": valid_predictions/len(predictions) if predictions else 0
    }

def test_model_setup(model, processor, tokenizer, device):
    """
    Test model setup with a simple text-only input.
    """
    print("\n=== Testing Model Setup ===")
    
    try:
        # Simple text-only test
        test_conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, can you respond to this test message?"
                    }
                ]
            }
        ]
        
        text_prompt = processor.apply_chat_template(
            test_conversation, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        print("Testing model forward pass...")
        with torch.no_grad():
            # Test forward pass first
            outputs = model(**inputs)
            print(f"✓ Forward pass successful. Output type: {type(outputs)}")
            
            # Test generation
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
            
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            response = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
            print(f"✓ Test generation successful: '{response}'")
            
        return True
        
    except Exception as e:
        print(f"✗ Model setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False