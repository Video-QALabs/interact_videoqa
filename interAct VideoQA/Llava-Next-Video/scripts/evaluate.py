import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from dataset import LlavaVideoQADataset
from model_setup import setup_llava_model
from inference import batch_inference, evaluate_predictions
from config import DATA_PATHS, TRAINING_CONFIG

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
    model, processor = setup_llava_model(checkpoint_path=checkpoint_path)
    
    # Get device
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # Create evaluation dataset
    print(f"Loading evaluation dataset from: {eval_csv}")
    eval_dataset = LlavaVideoQADataset(
        csv_file=eval_csv,
        video_dir=eval_video_dir,
        processor=processor
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
        batch_size=batch_size
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
    
    print(f"✓ Results saved to: {results_file}")
    
    return results

def quick_evaluation(model, processor, eval_loader, device, max_batches=None):
    """
    Quick evaluation during training.
    
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
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Quick Eval")):
            if max_batches and batch_idx >= max_batches:
                break
                
            try:
                # Prepare inputs
                gen_inputs = {k: v for k, v in batch.items() 
                            if k in ["input_ids", "attention_mask", "pixel_values_videos"]}
                gen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in gen_inputs.items()}
                
                # Generate
                generated_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=64,  # Shorter for quick eval
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
                
                # Decode
                prompt_length = batch["input_ids"].shape[1]
                new_tokens = generated_ids[:, prompt_length:]
                predictions = processor.batch_decode(
                    new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                # Count statistics
                batch_size = len(predictions)
                total_samples += batch_size
                
                for pred in predictions:
                    pred = pred.strip()
                    if len(pred) > 0:
                        successful_generations += 1
                    else:
                        empty_generations += 1
                
                # Save samples
                if len(sample_predictions) < 5:
                    sample_predictions.extend(predictions[:5-len(sample_predictions)])
                    sample_ground_truths.extend(
                        batch.get("ground_truth", ["N/A"] * batch_size)[:5-len(sample_ground_truths)]
                    )
                
            except Exception as e:
                print(f"Quick eval error: {e}")
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
        print(f"\nSample Prediction:")
        print(f"Generated: {sample_predictions[0]}")
        print(f"Expected: {sample_ground_truths[0]}")
    
    return results

def compare_models(checkpoint_paths, eval_csv=None, eval_video_dir=None):
    """
    Compare multiple model checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        eval_csv: Evaluation CSV file
        eval_video_dir: Evaluation video directory
    
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
                max_eval_samples=50  # Quick comparison
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
            print(f"  Empty predictions: {model_results.get('empty_predictions', 'N/A')}")
            print(f"  Error predictions: {model_results.get('error_predictions', 'N/A')}")
            print(f"  Avg pred length: {model_results.get('avg_pred_length', 'N/A'):.1f}")
        else:
            print(f"{model_name}: ERROR - {model_results['error']}")
    
    return results