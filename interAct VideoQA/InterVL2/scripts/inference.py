import math 
import torch 
from scripts.dataset import load_video
from scripts.model_setup import setup_model

def process_video_conversation(model, tokenizer, video_path, question, generation_config=None, history=None, num_segments=8, max_num=1):
    """Process a video and generate a conversation response with eager execution"""
    if generation_config is None:
        generation_config = dict(
            max_new_tokens=1024, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Load and process video
    pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=max_num)
    pixel_values = pixel_values.to(torch.bfloat16)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
    
    # Create video prefix for multi-frame input
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    full_question = video_prefix + question
    
    # Ensure model is in eval mode for inference
    model.eval()
    
    # Generate response with no gradient computation
    with torch.no_grad():
        try:
            response, new_history = model.chat(
                tokenizer, pixel_values, full_question, generation_config,
                num_patches_list=num_patches_list, history=history, return_history=True
            )
        except Exception as e:
            print(f"Error during model inference: {e}")
            return "Error occurred during inference", history
    
    return response, new_history

def batch_process_videos(model, tokenizer, video_paths, questions, generation_config=None, num_segments=8, max_num=1):
    """Process multiple videos in batch for efficiency"""
    results = []
    
    for video_path, question in zip(video_paths, questions):
        try:
            response, history = process_video_conversation(
                model, tokenizer, video_path, question, generation_config, 
                None, num_segments, max_num
            )
            results.append({
                'video_path': video_path,
                'question': question,
                'response': response,
                'success': True
            })
        except Exception as e:
            results.append({
                'video_path': video_path,
                'question': question,
                'response': f"Error: {str(e)}",
                'success': False
            })
    
    return results



