import os
import pandas as pd
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from decord import VideoReader, cpu

class VideoLlama3Dataset(Dataset):
    def __init__(self, csv_file: str, video_dir: str, processor, num_frames: int = 4, 
                 extension: str = ".mp4", target_size: tuple = None):
        self.df = pd.read_csv(csv_file, dtype={"Video_File_Path": str, "Questions": str, "Answers": str})
        self.video_dir = video_dir
        self.processor = processor
        self.num_frames = num_frames
        self.extension = extension
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def read_video(self, video_path: str) -> torch.Tensor:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            raise RuntimeError(f"Error opening video file {video_path}: {e}")
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int)
        frames = vr.get_batch(indices)
        if self.target_size is not None:
            frames_np = frames.cpu().numpy()
            resized_frames = []
            for frame in frames_np:
                resized = cv2.resize(frame, self.target_size)
                resized_frames.append(torch.tensor(resized, dtype=torch.float))
            frames_tensor = torch.stack(resized_frames)
        else:
            frames_tensor = frames.to(dtype=torch.float)
        return frames_tensor

    @staticmethod
    def flatten_conversation(conversation):
        flattened = ""
        for msg in conversation:
            role = msg["role"].capitalize()
            content = msg["content"]
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "video":
                            parts.append("<|video|>")
                        elif item.get("type") == "text":
                            parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                content = " ".join(parts)
            flattened += f"{role}: {content}\n"
        return flattened.strip()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_file = row["Video_File_Path"].strip()
        question = row["Questions"].strip()
        answer = row["Answers"].strip()
        video_path = os.path.join(self.video_dir, video_file)
        if not os.path.splitext(video_path)[1]:
            video_path += self.extension
        clip = self.read_video(video_path)
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": self.num_frames}},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer}
            ]}
        ]
        conversation_text = self.flatten_conversation(conversation)
        inputs = self.processor(
            text=conversation_text,
            return_tensors="pt"
        )
        processed_inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in processed_inputs:
            from utils import reduce_tensor
            processed_inputs["pixel_values"] = reduce_tensor(processed_inputs["pixel_values"], spatial_factor=2)
            processed_inputs["pixel_values"] = processed_inputs["pixel_values"].to(torch.float16)
        processed_inputs["labels"] = processed_inputs["input_ids"].clone()
        return processed_inputs

def video_llama3_collate_fn(examples):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    for key in ["input_ids", "attention_mask"]:
        pad_value = examples[0].get("pad_token_id", 0)
        batch[key] = pad_sequence(
            [ex[key] for ex in examples],
            batch_first=True,
            padding_value=pad_value
        )
    if "labels" in examples[0]:
        batch["labels"] = pad_sequence(
            [ex["labels"] for ex in examples],
            batch_first=True,
            padding_value=-100
        )
    if "pixel_values" in examples[0]:
        from utils import reduce_tensor
        reduced = [reduce_tensor(ex["pixel_values"], spatial_factor=2) for ex in examples]
        batch["pixel_values"] = torch.stack(reduced, dim=0)
    return batch
