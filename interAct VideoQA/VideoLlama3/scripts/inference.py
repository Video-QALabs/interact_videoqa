import torch
import os
from scripts.model_setup import load_model_for_inference
from scripts.config import MODEL_CONFIG, GENERATION_CONFIG

def run_inference(
    video_path: str,
    question: str,
    model_path: str = None,
    max_new_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
):
    """
    Run inference on a single video with VideoLLaMA3.
    
    Args:
        video_path: Path to video file
        question: Question to ask about the video
        model_path: Path to model checkpoint (if None, uses base model)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        Generated response text
    """
    print(f"Running inference on: {video_path}")
    print(f"Question: {question}")
    
    # Load model and processor
    if model_path:
        model, processor = load_model_for_inference(model_path)
    else:
        model, processor = load_model_for_inference()
    
    # Verify video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create conversation
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
                        "max_frames": MODEL_CONFIG.get("num_frames", 8)
                    }
                },
                {"type": "text", "text": question}
            ]
        },
    ]
    
    # Process inputs
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Move inputs to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Convert pixel values to bfloat16 if present
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    # Set generation parameters
    gen_kwargs = {
        "max_new_tokens": max_new_tokens or GENERATION_CONFIG["max_new_tokens"],
        "do_sample": GENERATION_CONFIG.get("do_sample", True),
        "temperature": temperature or GENERATION_CONFIG.get("temperature", 0.7),
        "top_p": top_p or GENERATION_CONFIG.get("top_p", 0.9),
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    
    # Decode response
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # Extract only the assistant's response (remove the conversation prefix)
    if "assistant\n" in generated_text:
        response = generated_text.split("assistant\n")[-1].strip()
    else:
        response = generated_text
    
    print(f"Response: {response}")
    return response

def batch_inference(
    video_paths: list,
    questions: list,
    model_path: str = None,
    output_file: str = None,
    **gen_kwargs
):
    """
    Run inference on multiple videos.
    
    Args:
        video_paths: List of video file paths
        questions: List of questions (same length as video_paths)
        model_path: Path to model checkpoint
        output_file: Path to save results
        **gen_kwargs: Generation parameters
    
    Returns:
        List of responses
    """
    if len(video_paths) != len(questions):
        raise ValueError("video_paths and questions must have the same length")
    
    results = []
    
    # Load model once for batch processing
    if model_path:
        model, processor = load_model_for_inference(model_path)
    else:
        model, processor = load_model_for_inference()
    
    for i, (video_path, question) in enumerate(zip(video_paths, questions)):
        print(f"\nProcessing {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
        
        try:
            response = run_single_inference(model, processor, video_path, question, **gen_kwargs)
            results.append({
                "video_path": video_path,
                "question": question,
                "response": response,
                "status": "success"
            })
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            results.append({
                "video_path": video_path,
                "question": question,
                "response": f"Error: {str(e)}",
                "status": "error"
            })
    
    # Save results if output file specified
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return results

def run_single_inference(model, processor, video_path: str, question: str, **gen_kwargs):
    """Helper function for single inference with pre-loaded model."""
    # Verify video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create conversation
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
                        "max_frames": MODEL_CONFIG.get("num_frames", 8)
                    }
                },
                {"type": "text", "text": question}
            ]
        },
    ]
    
    # Process inputs
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Move inputs to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Convert pixel values to bfloat16 if present
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    # Set generation parameters
    generation_params = {
        "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
        "do_sample": GENERATION_CONFIG.get("do_sample", True),
        "temperature": GENERATION_CONFIG.get("temperature", 0.7),
        "top_p": GENERATION_CONFIG.get("top_p", 0.9),
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    generation_params.update(gen_kwargs)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_params)
    
    # Decode response
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # Extract only the assistant's response
    if "assistant\n" in generated_text:
        response = generated_text.split("assistant\n")[-1].strip()
    else:
        response = generated_text
    
    return response

def interactive_inference(model_path: str = None):
    """Interactive inference mode."""
    print("=== VideoLLaMA3 Interactive Inference ===")
    print("Enter 'quit' to exit")
    
    # Load model
    if model_path:
        model, processor = load_model_for_inference(model_path)
    else:
        model, processor = load_model_for_inference()
    
    while True:
        # Get video path
        video_path = input("\nEnter video path: ").strip()
        if video_path.lower() == 'quit':
            break
        
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        
        # Get question
        question = input("Enter question: ").strip()
        if question.lower() == 'quit':
            break
        
        try:
            response = run_single_inference(model, processor, video_path, question)
            print(f"\nResponse: {response}")
        except Exception as e:
            print(f"Error: {e}")
