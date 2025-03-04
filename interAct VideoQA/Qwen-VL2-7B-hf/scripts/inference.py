import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu, bridge
from utils import fetch_video

# Set up decord to return torch tensors.
bridge.set_bridge("torch")

def main():
    save_dir = "./saved_qwen2vl_model"
    # Load the model, tokenizer, and processor from disk.
    # Optionally, you can use 8-bit mode if supported:
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     save_dir,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     load_in_8bit=True,
    #     attn_implementation="flash_attention_2",
    # )
    model = Qwen2VLForConditionalGeneration.from_pretrained(save_dir, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    processor = AutoProcessor.from_pretrained(save_dir)
    print("Model, tokenizer, and processor loaded from", save_dir)

    video_path = "/scratch/jnolas77/Videos/clip_videos_44.mp4"  # Adjust as needed.
    video = fetch_video(video_path, num_frames=4)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Is there a cycle in the video?."}
            ]
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        videos=[video],
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=128)
    decoded_output = processor.batch_decode(output_ids, skip_special_tokens=True)
    print("Inference Output:")
    print(decoded_output)

if __name__ == "__main__":
    main()
