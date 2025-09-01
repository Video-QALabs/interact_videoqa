import os
import pandas as pd
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from scripts.config import MODEL_CONFIG

class MemoryOptimizedQwenVideoQADataset(Dataset):
    """
    Memory-optimized dataset for Qwen2.5-VL that returns chat-style `messages`
    suitable for processor.apply_chat_template + process_vision_info.
    Falls back to frames or text-only when videos are missing/unreadable.
    """

    def __init__(
        self,
        csv_file: str,
        video_dir: str,
        processor,
        tokenizer,
        single_video_mode: bool = False,
        video_path: str = None,
        question: str = None,
    ):
        self.video_dir = video_dir
        self.processor = processor
        self.tokenizer = tokenizer

        # memory-related knobs
        self.max_pixels = MODEL_CONFIG.get("max_pixels", 120 * 140)
        self.max_frames = MODEL_CONFIG.get("max_frames", 3)
        self.fps = MODEL_CONFIG.get("fps", 0.3)
        self.image_size = MODEL_CONFIG.get("image_size", 168)
        self.video_max_length = MODEL_CONFIG.get("video_max_length", 20)
        self.max_seq_length = MODEL_CONFIG.get("max_seq_length", 4096)

        self.single_video_mode = single_video_mode
        if single_video_mode:
            self.data = pd.DataFrame(
                [{
                    "video_file_path": video_path,
                    "question": question,
                    "answer": "N/A",
                }]
            )
        else:
            self.data = pd.read_csv(csv_file)

    # -------- utils --------

    def __len__(self):
        return len(self.data)

    def _safe_str(self, x, default=""):
        if isinstance(x, str):
            return x.strip()
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return str(x).strip()

    def _to_local_path(self, p: str) -> str:
        """Return a normalized absolute local path (no file://)."""
        return str(Path(p).expanduser().resolve())

    # -------- frame extraction --------

    def extract_and_resize_frames(self, video_path: str, max_frames: int = 3) -> list:
        """
        Extract up to `max_frames` frames from a local video file, resized to image_size.
        Returns a list of PIL Images (length guaranteed to be max_frames by padding).
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps and fps > 0 else 0

            # clamp duration
            if duration and duration > self.video_max_length and fps and fps > 0:
                max_frame_idx = int(fps * self.video_max_length)
                total_frames = min(total_frames, max_frame_idx)

            if total_frames <= 0:
                raise ValueError("Video appears to have 0 frames")

            # choose indices
            if total_frames <= max_frames:
                step = max(1, total_frames // max_frames)  # avoid step=0
                frame_indices = list(range(0, total_frames, step))[:max_frames]
            else:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))

            cap.release()

            # pad/ensure length
            while len(frames) < max_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(Image.new("RGB", (self.image_size, self.image_size), color="black"))

            return frames[:max_frames]

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            black = Image.new("RGB", (self.image_size, self.image_size), color="black")
            return [black] * max_frames

    # -------- main sample builder --------

    def __getitem__(self, idx: int):
        """
        Returns a dict:
          {
            "messages": [
              {
                "role": "user",
                "content": [
                  {"type": "video", "video": <abs local path>, "max_pixels": ..., "fps": ...},
                  {"type": "text", "text": <question>}
                ]
              }
            ],
            "ground_truth": <str>
          }
        If no video is available/readable, falls back to frames or text-only.
        """
        row = self.data.iloc[idx]
        video_file = self._safe_str(row.get("video_file_path", ""), "")
        question = self._safe_str(row.get("question", ""), "Please describe what happens in the video.")
        answer = self._safe_str(row.get("answer", ""), "N/A")

        # resolve path (probe extension if needed)
        video_path = os.path.join(self.video_dir, video_file) if video_file else ""
        if video_path and not os.path.splitext(video_path)[1]:
            for ext in (".mp4", ".avi", ".mov", ".mkv"):
                cand = video_path + ext
                if os.path.exists(cand):
                    video_path = cand
                    break

        # 1) Preferred: direct video (absolute local path, NO file://)
        if video_path and os.path.exists(video_path):
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": self._to_local_path(video_path),
                        "max_pixels": self.max_pixels,
                        "fps": self.fps,
                    },
                    {"type": "text", "text": question or "Please answer the question about this video."},
                ],
            }]
            return {"messages": messages, "ground_truth": answer}

        # 2) Fallback: a few frames → images
        frames = []
        try:
            if video_path:
                frames = self.extract_and_resize_frames(video_path, max_frames=self.max_frames)
                frames = [f for f in frames if isinstance(f, Image.Image)]
        except Exception:
            frames = []

        if frames:
            messages = [{
                "role": "user",
                "content": (
                    [{"type": "text", "text": f"These {len(frames)} frames are from a video. {question}"}] +
                    [{"type": "image", "image": frame} for frame in frames]
                ),
            }]
            return {"messages": messages, "ground_truth": answer}

        # 3) Last resort: text-only (keeps the pipeline alive)
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": f"(No video available) {question}"}],
        }]
        return {"messages": messages, "ground_truth": answer}

    # -------- collate --------

    def collate_fn_fixed(self, batch):
        """
        Collate for messages-based batches.
        We just pack lists; the processor will tokenize/build tensors later.
        """
        # Expect dicts with "messages" and "ground_truth"
        messages = []
        gts = []
        for item in batch:
            messages.append(item.get("messages", []))
            gts.append(item.get("ground_truth", "N/A"))
        return {"messages": messages, "ground_truth": gts}


# -------- dataloader helper --------

def create_memory_efficient_dataloader(dataset, batch_size=1, shuffle=True):
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn_fixed,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
    )


# -------- optional: tiny utility to clear cache --------

def clear_memory_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_resv = torch.cuda.memory_reserved() / (1024 ** 3)
        return mem_alloc, mem_resv
    return 0.0, 0.0
