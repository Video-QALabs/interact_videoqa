import os
import pandas as pd
import torch
import numpy as np
import av
from torch.utils.data import Dataset
from PIL import Image
from config import MODEL_CONFIG

class LlavaVideoQADataset(Dataset):
    """
    A dataset class for video-based question answering using LLaVA-Next.
    """
    def __init__(self, csv_file: str, video_dir: str, processor, num_frames: int = None, 
                 single_video_mode: bool = False, video_path: str = None, question: str = None):
        """
        Args:
            csv_file (str): Path to the CSV file containing video paths and QA pairs
            video_dir (str): Directory containing the video files
            processor: LlavaNextVideoProcessor instance
            num_frames (int): Number of frames to sample from each video
            single_video_mode (bool): Whether to use single video mode for inference
            video_path (str): Path to single video for inference
            question (str): Question for single video inference
        """
        self.video_dir = video_dir
        self.processor = processor
        self.num_frames = num_frames or MODEL_CONFIG["num_frames"]
        self.single_video_mode = single_video_mode
        
        if single_video_mode:
            # Single video inference mode
            self.data = pd.DataFrame([{
                "Video_File_Path": video_path,
                "Questions": question,
                "Answers": "N/A"
            }])
        else:
            # Training/evaluation mode
            self.data = pd.read_csv(csv_file)
        
        # Get video token from processor
        self.video_token_str = "<video>"

    def __len__(self):
        return len(self.data)

    def read_video_pyav(self, video_path: str, indices: list = None) -> list:
        """Extract frames from video using PyAV."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        container = av.open(video_path)
        # Get video stream
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        if indices is None:
            # Calculate frame indices to sample
            if total_frames <= self.num_frames:
                indices = np.arange(total_frames)
                # Pad with last frame if needed
                while len(indices) < self.num_frames:
                    indices = np.append(indices, indices[-1])
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                # Convert frame to PIL Image
                img = frame.to_image()
                frames.append(img)
            if len(frames) >= len(indices):
                break
        
        container.close()
        
        # Ensure we have exactly num_frames frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])  # Duplicate last frame
            
        return frames[:self.num_frames]

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        video_file = row["Video_File_Path"].strip()
        question = row["Questions"].strip()
        answer = row["Answers"].strip()
        
        # Handle video path
        video_path = os.path.join(self.video_dir, video_file)
        
        # Add extension if not present
        if not os.path.splitext(video_path)[1]:
            # Try common video extensions
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                if os.path.exists(video_path + ext):
                    video_path += ext
                    break
        
        try:
            # Read video frames
            clip = self.read_video_pyav(video_path)
            
            # Construct the prompt using the expected video token
            prompt = f"USER: <video>\n{question}\nASSISTANT:"
            
            # Process the inputs
            inputs = self.processor(
                text=prompt, 
                videos=[clip], 
                padding=True, 
                return_tensors="pt"
            )
            
            # Remove batch dimension added by processor
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            
            # Ensure correct key naming for video data
            if "pixel_values" in inputs and "pixel_values_videos" not in inputs:
                inputs["pixel_values_videos"] = inputs.pop("pixel_values")
            
            # Add ground truth answer
            inputs["ground_truth"] = answer
            
            return inputs
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return dummy data in case of error
            dummy_frames = [Image.new('RGB', (224, 224), color='black')] * self.num_frames
            prompt = f"USER: <video>\n{question}\nASSISTANT:"
            
            inputs = self.processor(
                text=prompt,
                videos=[dummy_frames],
                padding=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            if "pixel_values" in inputs and "pixel_values_videos" not in inputs:
                inputs["pixel_values_videos"] = inputs.pop("pixel_values")
            inputs["ground_truth"] = answer
            
            return inputs

    def collate_fn(self, batch):
        """Custom collate function for batching."""
        # Extract ground truth before collating
        ground_truths = [item.pop("ground_truth") for item in batch]
        
        # Get all keys from the first item
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                # Stack tensors
                collated[key] = torch.stack(values)
            else:
                # Keep as list for non-tensor values
                collated[key] = values
        
        # Add ground truth back
        collated["ground_truth"] = ground_truths
        
        return collated