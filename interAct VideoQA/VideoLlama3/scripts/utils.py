import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import List, Dict, Any

def reduce_tensor(tensor, spatial_factor=2):
    """
    Reduce tensor spatial dimensions by a factor.
    Note: This is mainly for backward compatibility. VideoLLaMA3 handles
    frame sizing internally and may not need this function.
    
    Args:
        tensor: Input tensor
        spatial_factor: Factor to reduce spatial dimensions
    
    Returns:
        Reduced tensor
    """
    if tensor.dtype != torch.half and tensor.dtype != torch.bfloat16:
        tensor = tensor.half()
    
    if tensor.dim() >= 3:
        *other_dims, height, width = tensor.shape
        new_height = max(1, int(height) // spatial_factor)
        new_width = max(1, int(width) // spatial_factor)
        tensor = F.interpolate(
            tensor.unsqueeze(0), 
            size=(new_height, new_width),
            mode="bilinear", 
            align_corners=False
        ).squeeze(0)
    
    return tensor

def fetch_video(video_path: str, num_frames: int = 8):
    """
    Fetch video frames using decord.
    Note: VideoLLaMA3 processor handles video loading internally,
    but this is useful for preprocessing or debugging.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
    
    Returns:
        Tensor of video frames
    """
    try:
        from decord import VideoReader, cpu
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Sample frames evenly
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = vr.get_batch(indices)
        
        return frames.to(dtype=torch.float)
        
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        # Return dummy frames
        return torch.zeros((num_frames, 224, 224, 3), dtype=torch.float)

def convert_csv_to_jsonl(
    csv_file: str, 
    video_dir: str, 
    output_jsonl: str,
    system_message: str = "You are a helpful assistant."
):
    """
    Convert CSV dataset to JSONL format for VideoLLaMA3 training.
    
    Args:
        csv_file: Path to CSV file with columns: Video_File_Path, Questions, Answers
        video_dir: Directory containing video files
        output_jsonl: Output JSONL file path
        system_message: System message for conversations
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file, dtype={
        "Video_File_Path": str, 
        "Questions": str, 
        "Answers": str
    })
    
    jsonl_data = []
    
    for idx, row in df.iterrows():
        video_file = row["Video_File_Path"].strip()
        question = row["Questions"].strip()
        answer = row["Answers"].strip()
        
        # Create relative path from video_dir
        video_path = os.path.join(video_dir, video_file)
        if not os.path.splitext(video_path)[1]:
            video_path += ".mp4"
        
        # Create JSONL entry
        entry = {
            "video": [video_path],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<video>\n{question}"
                },
                {
                    "from": "gpt", 
                    "value": answer
                }
            ]
        }
        
        jsonl_data.append(entry)
    
    # Save JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(jsonl_data)} entries to {output_jsonl}")

def validate_dataset(csv_file: str, video_dir: str):
    """
    Validate dataset by checking if all videos exist.
    
    Args:
        csv_file: Path to CSV file
        video_dir: Directory containing videos
    
    Returns:
        Dictionary with validation results
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file, dtype={
        "Video_File_Path": str, 
        "Questions": str, 
        "Answers": str
    })
    
    results = {
        "total_samples": len(df),
        "valid_samples": 0,
        "missing_videos": [],
        "invalid_entries": []
    }
    
    for idx, row in df.iterrows():
        try:
            video_file = row["Video_File_Path"].strip()
            question = row["Questions"].strip()
            answer = row["Answers"].strip()
            
            if not video_file or not question or not answer:
                results["invalid_entries"].append({
                    "index": idx,
                    "reason": "Empty fields",
                    "video_file": video_file
                })
                continue
            
            video_path = os.path.join(video_dir, video_file)
            if not os.path.splitext(video_path)[1]:
                video_path += ".mp4"
            
            if not os.path.exists(video_path):
                results["missing_videos"].append({
                    "index": idx,
                    "video_file": video_file,
                    "full_path": video_path
                })
                continue
            
            results["valid_samples"] += 1
            
        except Exception as e:
            results["invalid_entries"].append({
                "index": idx,
                "reason": str(e),
                "video_file": row.get("Video_File_Path", "N/A")
            })
    
    # Print summary
    print(f"Dataset Validation Results:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Valid samples: {results['valid_samples']}")
    print(f"  Missing videos: {len(results['missing_videos'])}")
    print(f"  Invalid entries: {len(results['invalid_entries'])}")
    
    if results['missing_videos']:
        print(f"\nFirst 5 missing videos:")
        for missing in results['missing_videos'][:5]:
            print(f"  - Index {missing['index']}: {missing['video_file']}")
    
    return results

def create_sample_dataset(output_dir: str, num_samples: int = 10):
    """
    Create a sample dataset for testing.
    
    Args:
        output_dir: Directory to create sample dataset
        num_samples: Number of sample entries to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample CSV
    import pandas as pd
    
    sample_data = []
    for i in range(num_samples):
        sample_data.append({
            "Video_File_Path": f"sample_video_{i:03d}.mp4",
            "Questions": f"What is happening in video {i}?",
            "Answers": f"This is a sample answer for video {i}."
        })
    
    df = pd.DataFrame(sample_data)
    csv_path = os.path.join(output_dir, "sample_qa.csv")
    df.to_csv(csv_path, index=False)
    
    # Create sample JSONL
    jsonl_data = []
    for i in range(num_samples):
        entry = {
            "video": [f"videos/sample_video_{i:03d}.mp4"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<video>\nWhat is happening in video {i}?"
                },
                {
                    "from": "gpt",
                    "value": f"This is a sample answer for video {i}."
                }
            ]
        }
        jsonl_data.append(entry)
    
    jsonl_path = os.path.join(output_dir, "sample_annotations.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created sample dataset in {output_dir}:")
    print(f"  - CSV file: {csv_path}")
    print(f"  - JSONL file: {jsonl_path}")
    print(f"  - Samples: {num_samples}")

def setup_environment():
    """Setup environment for VideoLLaMA3 training."""
    
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_CACHE"] = "./cache"
    
    # Set CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("Environment setup completed")

def print_model_info(model):
    """Print model information including parameter counts."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")

def save_training_config(output_dir: str, config_dict: Dict[str, Any]):
    """Save training configuration to file."""
    
    config_path = os.path.join(output_dir, "training_config.json")
    
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config_dict.items():
        try:
            json.dumps(value)
            serializable_config[key] = value
        except TypeError:
            serializable_config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Training configuration saved to: {config_path}")