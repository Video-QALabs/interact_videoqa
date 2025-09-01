import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from scripts.dataset import MemoryOptimizedQwenVideoQADataset  # Fixed import
from scripts.model_setup import setup_qwen_model
from scripts.inference import batch_inference, evaluate_predictions
from scripts.config import DATA_PATHS, TRAINING_CONFIG

def run_evaluation(checkpoint_path=None, eval_csv=None, eval_video_dir=None, 
                  batch_size=1, max_eval_samples=None):
    """
    Run comprehensive evaluation on the model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        eval_csv: Path to evaluation CSV file
        eval_video_dir: Directory containing evaluation videos
        batch_size: Batch size for evaluation
        max_eval_samples: Maximum number of samples to evaluate
    
    Returns:
        dict: Evaluation results
    """
    # Set default paths
    eval_csv = eval_csv or DATA_PATHS["eval_csv"]
    eval_video_dir = eval_video_dir or DATA_PATHS["eval_video_dir"]
    
    # Setup model
    print("Setting up model...")
    model, processor, tokenizer = setup_qwen_model(checkpoint_path=checkpoint_path)

    # Get device
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # Create evaluation dataset
    print(f"Loading evaluation dataset from: {eval_csv}")
    eval_dataset = MemoryOptimizedQwenVideoQADataset(
        csv_file=eval_csv,
        video_dir=eval_video_dir,
        processor=processor,
        tokenizer=tokenizer
    )
    
    # Limit dataset size if specified
    if max_eval_samples and len(eval_dataset) > max_eval_samples:
        eval_dataset.data = eval_dataset.data.head(max_eval_samples)
        print(f"Limited evaluation to {max_eval_samples} samples")
    
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Run inference
    print("Running inference...")
    predictions, ground_truths = batch_inference(
        model=model,
        processor=processor,
        eval_dataset=eval_dataset,
        device=device,
        batch_size=batch_size,
        tokenizer=tokenizer
    )
    
    # Evaluate predictions
    print("Evaluating predictions...")
    eval_results = evaluate_predictions(predictions, ground_truths)
    
    # Save results
    results = {
        "evaluation_summary": eval_results,
        "predictions": predictions[:100],  # Save first 100 predictions
        "ground_truths": ground_truths[:100],
        "model_checkpoint": checkpoint_path,
        "eval_dataset": eval_csv,
        "total_samples": len(predictions)
    }
    
    # Save to file
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"eval_results_{timestamp}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    return results

def quick_evaluation(model, processor, eval_loader, device, max_batches=None):
    """
    Quick evaluation during training with memory optimizations.
    
    Args:
        model: Model to evaluate
        processor: Processor
        eval_loader: DataLoader for evaluation
        device: Device
        max_batches: Maximum number of batches to evaluate
    
    Returns:
        dict: Quick evaluation results
    """
    model.eval()
    total_samples = 0
    successful_generations = 0
    empty_generations = 0
    
    sample_predictions = []
    sample_ground_truths = []
    
    # Get tokenizer
    tokenizer = getattr(processor, 'tokenizer', processor)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Quick Eval", leave=False)):
            if max_batches and batch_idx >= max_batches:
                break
                
            try:
                # Clear cache periodically
                if batch_idx % 3 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Prepare inputs - only essential ones
                gen_inputs = {}
                for k, v in batch.items():
                    if k == "ground_truth":
                        continue
                    if k in ["input_ids", "attention_mask", "pixel_values_videos"] and isinstance(v, torch.Tensor):
                        gen_inputs[k] = v.to(device)
                
                if "input_ids" not in gen_inputs:
                    continue
                
                # Generate with conservative settings
                generated_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=32,  # Short for quick eval
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
                
                # Decode
                prompt_length = batch["input_ids"].shape[1]
                new_tokens = generated_ids[:, prompt_length:]
                predictions = tokenizer.batch_decode(
                    new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                # Count statistics
                batch_size_actual = len(predictions)
                total_samples += batch_size_actual
                
                for pred in predictions:
                    pred = pred.strip()
                    if len(pred) > 0:
                        successful_generations += 1
                    else:
                        empty_generations += 1
                
                # Save samples
                if len(sample_predictions) < 3:
                    sample_predictions.extend(predictions[:3-len(sample_predictions)])
                    sample_ground_truths.extend(
                        batch.get("ground_truth", ["N/A"] * batch_size_actual)[:3-len(sample_ground_truths)]
                    )
                
            except Exception as e:
                print(f"Quick eval error at batch {batch_idx}: {e}")
                continue
    
    model.train()
    
    # Calculate metrics
    success_rate = successful_generations / total_samples if total_samples > 0 else 0
    empty_rate = empty_generations / total_samples if total_samples > 0 else 0
    
    results = {
        "total_samples": total_samples,
        "successful_generations": successful_generations,
        "success_rate": success_rate,
        "empty_rate": empty_rate,
        "sample_predictions": sample_predictions,
        "sample_ground_truths": sample_ground_truths
    }
    
    print(f"\nQuick Evaluation Results:")
    print(f"Samples processed: {total_samples}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Empty generation rate: {empty_rate:.2%}")
    
    if sample_predictions:
        print(f"Sample Generation:")
        print(f"Generated: '{sample_predictions[0][:50]}{'...' if len(sample_predictions[0]) > 50 else ''}'")
        print(f"Expected: '{sample_ground_truths[0][:50]}{'...' if len(sample_ground_truths[0]) > 50 else ''}'")
    
    return results

def compare_models(checkpoint_paths, eval_csv=None, eval_video_dir=None, max_samples=25):
    """
    Compare multiple model checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        eval_csv: Evaluation CSV file
        eval_video_dir: Evaluation video directory
        max_samples: Maximum samples for quick comparison
    
    Returns:
        dict: Comparison results
    """
    results = {}
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n=== Evaluating Model {i+1}: {checkpoint_path} ===")
        
        try:
            model_results = run_evaluation(
                checkpoint_path=checkpoint_path,
                eval_csv=eval_csv,
                eval_video_dir=eval_video_dir,
                max_eval_samples=max_samples
            )
            results[f"model_{i+1}"] = model_results["evaluation_summary"]
            results[f"model_{i+1}"]["checkpoint"] = checkpoint_path
            
        except Exception as e:
            print(f"Error evaluating {checkpoint_path}: {e}")
            results[f"model_{i+1}"] = {"error": str(e), "checkpoint": checkpoint_path}
    
    # Print comparison
    print("\n=== Model Comparison ===")
    for model_name, model_results in results.items():
        if "error" not in model_results:
            print(f"{model_name}:")
            print(f"  Total samples: {model_results.get('total_samples', 'N/A')}")
            print(f"  Valid predictions: {model_results.get('valid_predictions', 'N/A')}")
            print(f"  Success rate: {model_results.get('success_rate', 'N/A'):.2%}")
            print(f"  Avg pred length: {model_results.get('avg_pred_length', 'N/A'):.1f}")
        else:
            print(f"{model_name}: ERROR - {model_results['error']}")
    
    return results

def evaluate_single_video(model, processor, video_path, question, ground_truth=None):
    """
    Evaluate a single video-question pair.
    
    Args:
        model: Model for inference
        processor: Processor
        video_path: Path to video file
        question: Question to ask
        ground_truth: Expected answer (optional)
    
    Returns:
        dict: Evaluation result
    """
    from scripts.inference import memory_safe_inference
    
    device = next(model.parameters()).device
    tokenizer = getattr(processor, 'tokenizer', processor)
    
    print(f"Evaluating single video: {video_path}")
    print(f"Question: {question}")
    
    try:
        # Run inference
        prediction = memory_safe_inference(
            model=model,
            processor=processor,
            video_path=video_path,
            question=question,
            device=device,
            tokenizer=tokenizer
        )
        
        result = {
            "video_path": video_path,
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "success": not prediction.startswith("Error:")
        }
        
        if ground_truth:
            # Simple word overlap calculation
            pred_words = set(prediction.lower().split())
            gt_words = set(ground_truth.lower().split())
            if pred_words and gt_words:
                overlap = len(pred_words & gt_words) / len(pred_words | gt_words)
                result["word_overlap"] = overlap
        
        print(f"Prediction: {prediction}")
        if ground_truth:
            print(f"Ground Truth: {ground_truth}")
            if "word_overlap" in result:
                print(f"Word Overlap: {result['word_overlap']:.2%}")
        
        return result
        
    except Exception as e:
        print(f"Single video evaluation failed: {e}")
        return {
            "video_path": video_path,
            "question": question,
            "prediction": f"Error: {e}",
            "ground_truth": ground_truth,
            "success": False
        }