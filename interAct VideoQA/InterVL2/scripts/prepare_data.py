import json
import csv
import os
from pathlib import Path

def convert_csv_to_jsonl(csv_path, video_dir, output_path):
    """
    Convert CSV annotation file to JSONL format for InterVL3 training
    
    Args:
        csv_path: Path to the CSV annotation file
        video_dir: Directory containing video files
        output_path: Output JSONL file path
    """
    
    jsonl_data = []
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            video_file = row['Video File Path']
            question = row['Question']
            answer = row['Answer']
            category = row['Category']
            
            # Create full video path
            video_path = os.path.join(video_dir, video_file)
            
            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            # Create conversation format expected by InterVL3
            conversation = [
                {
                    "from": "human",
                    "value": f"<video>\n{question}"
                },
                {
                    "from": "gpt", 
                    "value": answer
                }
            ]
            
            # Create JSONL entry
            jsonl_entry = {
                "id": f"video_{row['Index']}",
                "video": video_file,
                "conversations": conversation,
                "category": category
            }
            
            jsonl_data.append(jsonl_entry)
    
    # Write JSONL file
    with open(output_path, 'w', encoding='utf-8') as jsonlfile:
        for entry in jsonl_data:
            jsonlfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created JSONL file with {len(jsonl_data)} entries: {output_path}")
    return len(jsonl_data)

def create_meta_json(jsonl_path, video_dir, output_path, dataset_name="custom_video_qa"):
    """
    Create meta JSON file for InterVL3 training configuration
    
    Args:
        jsonl_path: Path to the JSONL annotation file
        video_dir: Directory containing video files  
        output_path: Output meta JSON file path
        dataset_name: Name for the dataset
    """
    
    # Count entries in JSONL
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        length = sum(1 for _ in f)
    
    # Ensure all paths are strings for JSON serialization
    meta_config = {
        dataset_name: {
            "root": str(video_dir),
            "annotation": str(jsonl_path),
            "data_augment": False,
            "max_dynamic_patch": 6,  # Reduced for video processing
            "repeat_time": 1,
            "length": length
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta_config, f, indent=2, ensure_ascii=False)
    
    print(f"Created meta JSON file: {output_path}")
    return meta_config

def prepare_training_data():
    """Prepare data for training - called from main.py"""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "Video_Annotations_raw - 1.33pm_10.1mins_clip_60.csv"
    video_dir = str(base_dir / "data")
    
    # Output paths
    output_dir = base_dir / "training_data"
    output_dir.mkdir(exist_ok=True)
    
    jsonl_path = str(output_dir / "video_qa_annotations.jsonl")
    meta_path = str(output_dir / "meta_config.json")
    
    # Convert CSV to JSONL (convert paths to strings)
    print("Converting CSV to JSONL format...")
    num_entries = convert_csv_to_jsonl(str(csv_path), video_dir, jsonl_path)
    
    # Create meta configuration
    print("Creating meta configuration...")
    meta_config = create_meta_json(jsonl_path, video_dir, meta_path)
    
    print(f"\nData preparation complete!")
    print(f"JSONL file: {jsonl_path}")
    print(f"Meta config: {meta_path}")
    print(f"Total samples: {num_entries}")
    
    return jsonl_path, meta_path