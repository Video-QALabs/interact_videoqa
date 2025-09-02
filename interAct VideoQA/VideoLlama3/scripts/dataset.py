import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from scripts.config import config_dict
import json

class VideoLLaMA3Dataset(Dataset):
    """Dataset class for VideoLLaMA3 training."""
    
    def __init__(self, csv_file: str, video_dir: str, processor, 
                 num_frames: int = 8, extension: str = ".mp4"):
        """
        Initialize VideoLLaMA3 dataset.
        
        Args:
            csv_file: Path to CSV file with video paths, questions, and answers
            video_dir: Directory containing video files
            processor: VideoLLaMA3 processor
            num_frames: Number of frames to sample from each video
            extension: Video file extension
        """
        self.df = pd.read_csv(csv_file, dtype={
            "Video_File_Path": str, 
            "Questions": str, 
            "Answers": str
        })
        self.video_dir = video_dir
        self.processor = processor
        self.num_frames = num_frames
        self.extension = extension
        
        print(f"Loaded dataset with {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def create_conversation(self, video_path: str, question: str, answer: str):
        """Create conversation format for VideoLLaMA3."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": {
                            "video_path": video_path, 
                            "fps": 1, 
                            "max_frames": self.num_frames
                        }
                    },
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        return conversation

    def __getitem__(self, idx: int):
        """Get item from dataset."""
        try:
            row = self.df.iloc[idx]
            video_file = row["Video_File_Path"].strip()
            question = row["Questions"].strip()
            answer = row["Answers"].strip()
            
            # Construct full video path
            video_path = os.path.join(self.video_dir, video_file)
            if not os.path.splitext(video_path)[1]:
                video_path += self.extension
            
            # Verify video exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Create conversation
            conversation = self.create_conversation(video_path, question, answer)
            
            # Process with VideoLLaMA3 processor
            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # Prepare inputs for training
            processed_inputs = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 1 else v
                for k, v in inputs.items()
            }
            
            # Convert pixel values to bfloat16 if present
            if "pixel_values" in processed_inputs and isinstance(processed_inputs["pixel_values"], torch.Tensor):
                processed_inputs["pixel_values"] = processed_inputs["pixel_values"].to(torch.bfloat16)
            
            # Create labels for training (copy of input_ids)
            if "input_ids" in processed_inputs:
                processed_inputs["labels"] = processed_inputs["input_ids"].clone()
            
            return processed_inputs
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a dummy item to avoid training interruption
            return self._get_dummy_item()
    
    def _get_dummy_item(self):
        """Create dummy item for error cases."""
        return {
            "input_ids": torch.tensor([1, 2, 3]),  # dummy tokens
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
        }

class VideoLLaMA3JSONLDataset(Dataset):
    """Dataset class for VideoLLaMA3 using JSONL format (recommended)."""
    
    def __init__(self, jsonl_file: str, data_root: str, processor):
        """
        Initialize dataset with JSONL format.
        
        Args:
            jsonl_file: Path to JSONL annotation file
            data_root: Root directory for data
            processor: VideoLLaMA3 processor
        """
        self.data_root = data_root
        self.processor = processor
        
        # Load JSONL data
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        print(f"Loaded JSONL dataset with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """Get item from JSONL dataset."""
        try:
            item = self.data[idx]
            
            # Handle video or image data
            if "video" in item:
                media_paths = [os.path.join(self.data_root, path) for path in item["video"]]
                media_type = "video"
            elif "image" in item:
                media_paths = [os.path.join(self.data_root, path) for path in item["image"]]
                media_type = "image"
            else:
                raise ValueError("No video or image found in data item")
            
            # Convert conversations to VideoLLaMA3 format
            conversation = []
            for conv in item["conversations"]:
                role = "user" if conv["from"] == "human" else "assistant"
                content = conv["value"]
                
                if role == "user" and media_type in content:
                    # Parse content with media
                    if media_type == "video":
                        conversation.append({
                            "role": role,
                            "content": [
                                {"type": "video", "video": {"video_path": media_paths[0], "fps": 1, "max_frames": 8}},
                                {"type": "text", "text": content.replace("<video>", "").strip()}
                            ]
                        })
                    else:  # image
                        conversation.append({
                            "role": role,
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": content.replace("<image>", "").strip()}
                            ]
                        })
                else:
                    conversation.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    })
            
            # Process with VideoLLaMA3 processor
            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # Prepare inputs
            processed_inputs = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 1 else v
                for k, v in inputs.items()
            }
            
            if "pixel_values" in processed_inputs and isinstance(processed_inputs["pixel_values"], torch.Tensor):
                processed_inputs["pixel_values"] = processed_inputs["pixel_values"].to(torch.bfloat16)
            
            if "input_ids" in processed_inputs:
                processed_inputs["labels"] = processed_inputs["input_ids"].clone()
            
            return processed_inputs
            
        except Exception as e:
            print(f"Error processing JSONL item {idx}: {e}")
            return self._get_dummy_item()
    
    def _get_dummy_item(self):
        """Create dummy item for error cases."""
        return {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
        }

def videollama3_collate_fn(examples):
    """Collate function for VideoLLaMA3 dataset."""
    from torch.nn.utils.rnn import pad_sequence
    
    # Filter out None examples
    examples = [ex for ex in examples if ex is not None]
    
    if not examples:
        return {}
    
    batch = {}
    
    # Handle text inputs
    for key in ["input_ids", "attention_mask"]:
        if key in examples[0]:
            pad_value = 0 if key == "input_ids" else 0
            batch[key] = pad_sequence(
                [ex[key] for ex in examples if key in ex],
                batch_first=True,
                padding_value=pad_value
            )
    
    # Handle labels
    if "labels" in examples[0]:
        batch["labels"] = pad_sequence(
            [ex["labels"] for ex in examples if "labels" in ex],
            batch_first=True,
            padding_value=-100
        )
    
    # Handle visual inputs
    if "pixel_values" in examples[0]:
        pixel_values = [ex["pixel_values"] for ex in examples if "pixel_values" in ex]
        if pixel_values:
            # Stack pixel values
            batch["pixel_values"] = torch.stack(pixel_values, dim=0)
    
    return batch