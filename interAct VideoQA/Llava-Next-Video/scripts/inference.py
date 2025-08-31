import torch
from tqdm import tqdm
import os
from scripts.dataset import LlavaVideoQADataset
from scripts.config import GENERATION_CONFIG

def inference_single_video(model, processor, video_path, question, device):
    """
    Run inference on a single video-question pair.
    
    Args:
        model: The trained model
        processor: The processor
        video_path: Path to video file
        question: Question about the video
        device: Device to run on
    
    Returns:
        str: Generated answer
    """
    # Put model in evaluation mode
    model.eval()
    
    # Create dataset for single video
    dataset = LlavaVideoQADataset(
        csv_file=None,
        video_dir=os.path.dirname(video_path),
        processor=processor,
        single_video_mode=True,
        video_path=os.path.basename(video_path),
        question=question
    )
    
    try:
        # Get the processed inputs
        inputs = dataset[0]
        
        # Remove ground_truth if present
        if "ground_truth" in inputs:
            inputs.pop("ground_truth")
        
        # Move everything to the right device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            # Add batch dimension
            gen_kwargs = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] 
                for k, v in inputs.items() 
                if k in ["input_ids", "attention_mask", "pixel_values_videos"]
            }
            
            output_ids = model.generate(
                **gen_kwargs,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                do_sample=GENERATION_CONFIG["do_sample"],
                num_beams=GENERATION_CONFIG.get("num_beams", 3),
                temperature=GENERATION_CONFIG.get("temperature", 0.7),
                top_p=GENERATION_CONFIG.get("top_p", 0.9),
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
            
            # Remove the input prompt to get only the generated text
            prompt_length = inputs["input_ids"].shape[0]
            generated_ids = output_ids[0, prompt_length:]
            
            # Decode the generated text
            generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        return generated_text.strip()
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return f"Error: Could not process video - {e}"

def batch_inference(model, processor, eval_dataset, device, batch_size=1):
    """
    Run inference on a batch of videos.
    
    Args:
        model: The trained model
        processor: The processor
        eval_dataset: Dataset containing video-question pairs
        device: Device to run on
        batch_size: Batch size for inference
    
    Returns:
        tuple: (predictions, ground_truths)
    """
    # Put model in evaluation mode
    model.eval()
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn
    )
    
    # Lists to store predictions and ground truth
    predictions = []
    ground_truths = []
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            try:
                # Move batch to device
                gen_inputs = {k: v for k, v in batch.items() 
                            if k in ["input_ids", "attention_mask", "pixel_values_videos"]}
                gen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in gen_inputs.items()}
                
                # Generate predictions
                output_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                    do_sample=False,  # Deterministic for evaluation
                    num_beams=GENERATION_CONFIG.get("num_beams", 3),
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
                
                # Get only the generated text (remove input prompt)
                prompt_length = batch["input_ids"].shape[1]
                generated_ids = output_ids[:, prompt_length:]
                
                # Decode predictions
                pred_texts = processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                predictions.extend([pred.strip() for pred in pred_texts])
                ground_truths.extend(batch.get("ground_truth", ["N/A"] * len(pred_texts)))
                
            except Exception as e:
                print(f"Error in batch inference: {e}")
                # Add placeholder predictions for this batch
                batch_size_actual = batch["input_ids"].shape[0] if "input_ids" in batch else 1
                predictions.extend(["Error: Could not process"] * batch_size_actual)
                ground_truths.extend(batch.get("ground_truth", ["N/A"] * batch_size_actual))
    
    return predictions, ground_truths

def evaluate_predictions(predictions, ground_truths):
    """
    Evaluate the quality of predictions.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
    """
    print("\n=== Evaluation Results ===")
    print(f"Total samples: {len(predictions)}")
    
    # Print sample predictions
    print("\nSample Predictions:")
    for i in range(min(10, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"Question/Context: (See dataset)")
        print(f"Prediction: {predictions[i]}")
        print(f"Ground Truth: {ground_truths[i]}")
        print("-" * 80)
    
    # Calculate basic statistics
    pred_lengths = [len(pred.split()) for pred in predictions]
    gt_lengths = [len(gt.split()) for gt in ground_truths if gt != "N/A"]
    
    print(f"\nLength Statistics:")
    print(f"Average prediction length: {sum(pred_lengths)/len(pred_lengths):.2f} words")
    if gt_lengths:
        print(f"Average ground truth length: {sum(gt_lengths)/len(gt_lengths):.2f} words")
    
    # Count empty/error predictions
    empty_predictions = sum(1 for pred in predictions if len(pred.strip()) == 0)
    error_predictions = sum(1 for pred in predictions if "Error:" in pred)
    
    print(f"Empty predictions: {empty_predictions} ({empty_predictions/len(predictions)*100:.1f}%)")
    print(f"Error predictions: {error_predictions} ({error_predictions/len(predictions)*100:.1f}%)")
    
    return {
        "total_samples": len(predictions),
        "avg_pred_length": sum(pred_lengths)/len(pred_lengths),
        "avg_gt_length": sum(gt_lengths)/len(gt_lengths) if gt_lengths else 0,
        "empty_predictions": empty_predictions,
        "error_predictions": error_predictions
    }