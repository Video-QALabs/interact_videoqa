import argparse
import os
import sys

def setup_model(model_name):
    model_name = model_name.lower()
    if model_name == "llava":
        from Llava_Next_Video.main import main as llava_main
        return llava_main
    elif model_name == "qwen":
        from Qwen_VL2_7B_hf.main import main as qwen_main
        return qwen_main
    elif model_name == "videollama":
        from VideoLlama2.main import main as videollama_main
        return videollama_main
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: llava, qwen, videollama")

def main():
    parser = argparse.ArgumentParser(description="Run VideoQA models")
    parser.add_argument("--model", type=str, required=True, 
                      help="Model to use (llava, qwen, videollama)")
    parser.add_argument("--config", type=str, default=None,
                      help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to model checkpoint")
    
    args, remaining_args = parser.parse_known_args()
    
    # Get the appropriate main function for the selected model
    model_main = setup_model(args.model)
    
    # Call the model-specific main function with the remaining arguments
    model_main(remaining_args)

if __name__ == "__main__":
    main()
