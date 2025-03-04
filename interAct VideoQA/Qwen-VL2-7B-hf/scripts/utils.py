import torch
import torch.nn.functional as F

def reduce_tensor(tensor, spatial_factor=2):
    if tensor.dtype != torch.half:
        tensor = tensor.half()
    if tensor.dim() >= 3:
        *other_dims, height, width = tensor.shape
        new_height = max(1, int(height) // spatial_factor)
        new_width = max(1, int(width) // spatial_factor)
        tensor = F.interpolate(tensor.unsqueeze(0), size=(new_height, new_width),
                               mode="bilinear", align_corners=False).squeeze(0)
    return tensor

def fetch_video(video_path, num_frames=4):
    import numpy as np
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices)
    return frames.to(dtype=torch.float)
