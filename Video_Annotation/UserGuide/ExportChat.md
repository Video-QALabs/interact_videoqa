# Export Questions Based on Chat Templates

Converts accepted Q&A annotations into training-ready formats for different video language models.

### Supported Models

1. **VideoLLaMA2** - Conversation-based format
2. **LLaVA-Next-Video** - Prompt-answer format
3. **Qwen-VL2-7b-hf** - Qwen video model format
4. **All Templates** - Exports all formats simultaneously

### Export Process

#### Step 1: Accept Questions

- Review Q&A pairs in video interface
- Click **\"Accept\"** for quality questions
- Use **\"Reject\"** for poor quality pairs
- Edit rejected questions by double-clicking (auto-accepts)

#### Step 2: Choose Template Format

Select desired format:

- 🔘 VideoLLaMA2
- 🔘 LLaVA-Next-Video
- 🔘 Qwen-VL2-7b-hf
- 🔘 All (generates all three)

#### Step 3: Save Template Selection

- Click **\"Save Template Selection\"**
- System generates templates in memory
- Continue with additional videos

#### Step 4: Final Export

- Click **\"Finish and Export\"**
- Choose output directory
- System creates CSV backup and JSONL files

### Template Formats

#### VideoLLaMA2 Format

```
{"prompt": "USER: How many lanes are visible in the video?\n<|video|>\nASSISTANT:", "answer": "Three lanes are visible", "video_path": "clip_012.mp4"}
{"prompt": "USER: Did the bus stop at the intersection?\n<|video|>\nASSISTANT:", "answer": "No the bus did not stop at the intersection", "video_path": "clip_012.mp4"}
{"prompt": "USER: Does any vehicle change lanes during the clip?\n<|video|>\nASSISTANT:", "answer": "Yes one vehicle changes lanes", "video_path":"clip_014.mp4"}
{"prompt": "USER: How many traffic signs are visible in the frame?\n<|video|>\nASSISTANT:", "answer": "Two traffic signs are visible", "video_path": "clip_017.mp4"}
{"prompt": "USER: What is the main color of the rightmost vehicle?\n<|video|>\nASSISTANT:", "answer": "The rightmost vehicle is blue", "video_path": "clip_017.mp4"}
{"prompt": "USER: Does any car stop at a red light?\n<|video|>\nASSISTANT:", "answer": "Yes a car stops at a red light", "video_path": "clip_017.mp4"}

```

#### LLaVA-Next-Video Format

```
{"prompt": "USER: How many lanes are visible in the video?\n<|video|>\nASSISTANT:", "answer": "Three lanes are visible", "video_path": "clip_012.mp4"}
{"prompt": "USER: Did the bus stop at the intersection?\n<|video|>\nASSISTANT:", "answer": "No the bus did not stop at the intersection", "video_path": "clip_012.mp4"}
{"prompt": "USER: Does any vehicle change lanes during the clip?\n<|video|>\nASSISTANT:", "answer": "Yes one vehicle changes lanes", "video_path": "clip_014.mp4"}
{"prompt": "USER: How many traffic signs are visible in the frame?\n<|video|>\nASSISTANT:", "answer": "Two traffic signs are visible", "video_path": "clip_017.mp4"}
{"prompt": "USER: What is the main color of the rightmost vehicle?\n<|video|>\nASSISTANT:", "answer": "The rightmost vehicle is blue", "video_path": "clip_017.mp4"}
{"prompt": "USER: Does any car stop at a red light?\n<|video|>\nASSISTANT:", "answer": "Yes a car stops at a red light", "video_path": "clip_017.mp4"}

```

#### Qwen-VL2 Format

```
{"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "video", "video": {"video_path": "clip_012.mp4", "fps": 1, "max_frames": 4}}, {"type": "text", "text": "How many lanes are visible in the video?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Three lanes are visible"}]}]}
{"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "video", "video": {"video_path": "clip_012.mp4", "fps": 1, "max_frames": 4}}, {"type": "text", "text": "Did the bus stop at the intersection?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "No the bus did not stop at the intersection"}]}]}
{"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "video", "video": {"video_path": "clip_014.mp4", "fps": 1, "max_frames": 4}}, {"type": "text", "text": "Does any vehicle change lanes during the clip?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Yes one vehicle changes lanes"}]}]}
{"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "video", "video": {"video_path": "clip_017.mp4", "fps": 1, "max_frames": 4}}, {"type": "text", "text": "How many traffic signs are visible in the frame?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Two traffic signs are visible"}]}]}
{"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "video", "video": {"video_path": "clip_017.mp4", "fps": 1, "max_frames": 4}}, {"type": "text", "text": "What is the main color of the rightmost vehicle?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "The rightmost vehicle is blue"}]}]}
{"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "video", "video": {"video_path": "clip_017.mp4", "fps": 1, "max_frames": 4}}, {"type": "text", "text": "Does any car stop at a red light?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Yes a car stops at a red light"}]}]}

```

### Output Files

#### Single Template Export

- Format: `model_train_{template_suffix}.jsonl`
- Examples:
  - `model_train_videollama2.jsonl`
  - `model_train_llava_next.jsonl`
  - `model_train_qwen.jsonl`

#### All Templates Export

Creates three separate files with all formats.


## Quality Control Features

### Data Validation

- Ensures required fields are present (video_path, question, answer)
- Skips incomplete records with logging
- Validates video file paths exist

### Backup System

- Automatic timestamped CSV backups before export
- Format: `{original_name}_backup_{YYYYMMDD_HHMMSS}.csv`
- Preserves all original data and status information

## Best Practices

### Before Export

1. Review all Q&A pairs for quality and accuracy
2. Accept only high-quality, relevant questions
3. Edit and improve rejected questions when possible
4. Ensure video files are accessible at specified paths

### During Export

1. Choose appropriate template format for your target model
2. Select organized output directory structure
3. Verify sufficient disk space for export files
4. Monitor export progress through status messages

### After Export

1. Verify generated JSONL files contain expected data
2. Check file sizes are reasonable for dataset size
3. Validate JSON formatting using external tools if needed
4. Backup exported files before using in training pipelines
