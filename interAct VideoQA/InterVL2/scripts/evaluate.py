#!/usr/bin/env python3
"""
Evaluation script for InterVL3 Video QA model
Supports both base model and LoRA fine-tuned model evaluation
"""

import os
import json
import csv
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scripts.model_setup import setup_model
from scripts.inference import process_video_conversation, batch_process_videos
from scripts.config import DATA_DIR

def load_test_data(csv_path):
    """Load test data from CSV file"""
    test_data = []
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_data.append({
                'index': row['Index'],
                'video_file': row['Video File Path'],
                'question': row['Question'],
                'category': row['Category'],
                'ground_truth': row['Answer']
            })
    
    return test_data

def evaluate_model(model, tokenizer, test_data, video_dir, output_file=None):
    """Evaluate model on test data"""
    results = []
    
    generation_config = dict(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    print(f"Evaluating on {len(test_data)} samples...")
    
    for sample in tqdm(test_data, desc="Processing videos"):
        video_path = os.path.join(video_dir, sample['video_file'])
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            results.append({
                **sample,
                'predicted_answer': 'ERROR: Video not found',
                'success': False
            })
            continue
        
        try:
            # Generate prediction
            response, _ = process_video_conversation(
                model, tokenizer, video_path, sample['question'], 
                generation_config, num_segments=8, max_num=1
            )
            
            results.append({
                **sample,
                'predicted_answer': response,
                'success': True
            })
            
        except Exception as e:
            print(f"Error processing {sample['video_file']}: {e}")
            results.append({
                **sample,
                'predicted_answer': f'ERROR: {str(e)}',
                'success': False
            })
    
    # Save results
    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    
    return results

def calculate_metrics(results):
    """Calculate evaluation metrics"""
    total_samples = len(results)
    successful_samples = sum(1 for r in results if r['success'])
    success_rate = successful_samples / total_samples * 100
    
    # Count by category
    category_stats = {}
    for result in results:
        category = result['category']
        if category not in category_stats:
            category_stats[category] = {'total': 0, 'success': 0}
        
        category_stats[category]['total'] += 1
        if result['success']:
            category_stats[category]['success'] += 1
    
    # Calculate category success rates
    for category in category_stats:
        stats = category_stats[category]
        stats['success_rate'] = stats['success'] / stats['total'] * 100
    
    return {
        'total_samples': total_samples,
        'successful_samples': successful_samples,
        'overall_success_rate': success_rate,
        'category_stats': category_stats
    }

def print_evaluation_report(metrics, results):
    """Print detailed evaluation report"""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Successful predictions: {metrics['successful_samples']}")
    print(f"Overall success rate: {metrics['overall_success_rate']:.2f}%")
    
    print(f"\nCategory breakdown:")
    print("-" * 40)
    for category, stats in metrics['category_stats'].items():
        print(f"{category}:")
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Success rate: {stats['success_rate']:.2f}%")
    
    # Show some example predictions
    print(f"\nExample predictions:")
    print("-" * 40)
    successful_results = [r for r in results if r['success']][:3]
    
    for i, result in enumerate(successful_results, 1):
        print(f"\nExample {i}:")
        print(f"Video: {result['video_file']}")
        print(f"Question: {result['question']}")
        print(f"Ground truth: {result['ground_truth']}")
        print(f"Prediction: {result['predicted_answer']}")
        print(f"Category: {result['category']}")

def run_evaluation(model_path="OpenGVLab/InternVL3-38B", 
                   test_csv=None, 
                   video_dir=None, 
                   output_file=None, 
                   cache_dir=None):
    """Run evaluation - called from main.py"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    
    if test_csv is None:
        test_csv = base_dir / "data" / "Video_Annotations_raw - 1.33pm_10.1mins_clip_60.csv"
    elif not os.path.isabs(test_csv):
        test_csv = base_dir / test_csv
        
    if video_dir is None:
        video_dir = base_dir / "data"
    elif not os.path.isabs(video_dir):
        video_dir = base_dir / video_dir
        
    # Default output file
    if output_file is None:
        model_name = Path(model_path).name.replace("/", "_")
        output_file = base_dir / "outputs" / f"evaluation_{model_name}.csv"
        os.makedirs(output_file.parent, exist_ok=True)
    
    print(f"Loading model: {model_path}")
    print(f"Test data: {test_csv}")
    print(f"Video directory: {video_dir}")
    print(f"Output file: {output_file}")
    
    # Load model
    try:
        if cache_dir:
            os.environ['MODELSCOPE_CACHE'] = cache_dir
        
        from scripts.model_setup import setup_model
        model, tokenizer = setup_model(
            model_path=model_path,
            cache_dir=cache_dir
        )
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Load test data
    try:
        test_data = load_test_data(test_csv)
        print(f"Loaded {len(test_data)} test samples")
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return False
    
    # Run evaluation
    try:
        results = evaluate_model(model, tokenizer, test_data, video_dir, output_file)
        
        # Calculate and print metrics
        metrics = calculate_metrics(results)
        print_evaluation_report(metrics, results)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False
    
    print(f"\nEvaluation completed successfully!")
    return True