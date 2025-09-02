import torch
import pandas as pd
import json
import os
from scripts.inference import batch_inference
from scripts.model_setup import load_model_for_inference
from scripts.config import DATA_PATHS, GENERATION_CONFIG

def evaluate_on_dataset(
    eval_csv: str = None,
    eval_video_dir: str = None,
    model_path: str = None,
    output_file: str = "evaluation_results.json",
    max_samples: int = None
):
    """
    Evaluate VideoLLaMA3 on a test dataset.
    
    Args:
        eval_csv: Path to evaluation CSV file
        eval_video_dir: Directory containing evaluation videos
        model_path: Path to model checkpoint
        output_file: Path to save evaluation results
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        Evaluation results dictionary
    """
    # Use default paths if not provided
    if eval_csv is None:
        eval_csv = DATA_PATHS.get("eval_csv")
    if eval_video_dir is None:
        eval_video_dir = DATA_PATHS.get("eval_video_dir")
    
    print(f"Evaluating on dataset: {eval_csv}")
    print(f"Video directory: {eval_video_dir}")
    
    # Load evaluation data
    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"Evaluation CSV not found: {eval_csv}")
    
    df = pd.read_csv(eval_csv, dtype={
        "Video_File_Path": str, 
        "Questions": str, 
        "Answers": str
    })
    
    # Limit samples if specified
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)
        print(f"Limited evaluation to {max_samples} samples")
    
    print(f"Evaluating on {len(df)} samples")
    
    # Prepare video paths and questions
    video_paths = []
    questions = []
    ground_truth_answers = []
    
    for idx, row in df.iterrows():
        video_file = row["Video_File_Path"].strip()
        question = row["Questions"].strip()
        answer = row["Answers"].strip()
        
        video_path = os.path.join(eval_video_dir, video_file)
        if not os.path.splitext(video_path)[1]:
            video_path += ".mp4"  # Add extension if missing
        
        video_paths.append(video_path)
        questions.append(question)
        ground_truth_answers.append(answer)
    
    # Run batch inference
    results = batch_inference(
        video_paths=video_paths,
        questions=questions,
        model_path=model_path,
        output_file=None,  # We'll save our own format
        **GENERATION_CONFIG
    )
    
    # Add ground truth and compute metrics
    evaluation_results = {
        "total_samples": len(results),
        "successful_samples": 0,
        "failed_samples": 0,
        "samples": [],
        "metrics": {}
    }
    
    for i, result in enumerate(results):
        sample_result = {
            "index": i,
            "video_path": result["video_path"],
            "question": result["question"],
            "ground_truth": ground_truth_answers[i],
            "generated_response": result["response"],
            "status": result["status"]
        }
        
        if result["status"] == "success":
            evaluation_results["successful_samples"] += 1
            # Add basic metrics (can be extended)
            sample_result["response_length"] = len(result["response"])
        else:
            evaluation_results["failed_samples"] += 1
        
        evaluation_results["samples"].append(sample_result)
    
    # Compute overall metrics
    success_rate = evaluation_results["successful_samples"] / evaluation_results["total_samples"]
    evaluation_results["metrics"]["success_rate"] = success_rate
    evaluation_results["metrics"]["failure_rate"] = 1 - success_rate
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Evaluation completed!")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Results saved to: {output_file}")
    
    return evaluation_results

def compute_text_metrics(generated_responses, ground_truth_answers):
    """
    Compute text-based metrics for evaluation.
    
    Args:
        generated_responses: List of generated responses
        ground_truth_answers: List of ground truth answers
    
    Returns:
        Dictionary of metrics
    """
    try:
        # Try to import advanced NLP metrics
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu
        import nltk
        nltk.download('punkt', quiet=True)
        
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        
        for gen_resp, gt_answer in zip(generated_responses, ground_truth_answers):
            # ROUGE scores
            rouge_result = scorer.score(gt_answer, gen_resp)
            for rouge_type in rouge_scores:
                rouge_scores[rouge_type].append(rouge_result[rouge_type].fmeasure)
            
            # BLEU score
            bleu_score = sentence_bleu([gt_answer.split()], gen_resp.split())
            bleu_scores.append(bleu_score)
        
        # Average scores
        metrics = {}
        for rouge_type in rouge_scores:
            metrics[f'{rouge_type}_f1'] = sum(rouge_scores[rouge_type]) / len(rouge_scores[rouge_type])
        
        metrics['bleu'] = sum(bleu_scores) / len(bleu_scores)
        
        return metrics
        
    except ImportError:
        print("Advanced NLP metrics not available. Install rouge-score and nltk for detailed metrics.")
        return {"note": "Advanced metrics not available"}

def evaluate_with_metrics(
    eval_csv: str = None,
    eval_video_dir: str = None,
    model_path: str = None,
    output_file: str = "detailed_evaluation_results.json",
    max_samples: int = None,
    compute_nlp_metrics: bool = False
):
    """
    Evaluate with additional NLP metrics.
    
    Args:
        eval_csv: Path to evaluation CSV file
        eval_video_dir: Directory containing evaluation videos
        model_path: Path to model checkpoint
        output_file: Path to save evaluation results
        max_samples: Maximum number of samples to evaluate
        compute_nlp_metrics: Whether to compute NLP metrics (ROUGE, BLEU)
    
    Returns:
        Detailed evaluation results
    """
    # Run basic evaluation
    results = evaluate_on_dataset(
        eval_csv=eval_csv,
        eval_video_dir=eval_video_dir,
        model_path=model_path,
        output_file=None,  # Don't save yet
        max_samples=max_samples
    )
    
    if compute_nlp_metrics and results["successful_samples"] > 0:
        # Extract successful responses and ground truth
        successful_samples = [s for s in results["samples"] if s["status"] == "success"]
        generated_responses = [s["generated_response"] for s in successful_samples]
        ground_truth_answers = [s["ground_truth"] for s in successful_samples]
        
        # Compute NLP metrics
        nlp_metrics = compute_text_metrics(generated_responses, ground_truth_answers)
        results["metrics"].update(nlp_metrics)
        
        print("NLP Metrics:")
        for metric, value in nlp_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
    
    # Save detailed results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed evaluation results saved to: {output_file}")
    return results
