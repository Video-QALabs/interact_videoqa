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

```json
{
  \"conversations\": [
    {
      \"role\": \"system\",
      \"content\": \"You are a helpful assistant.\"
    },
    {
      \"role\": \"user\",
      \"content\": [
        {
          \"type\": \"video\",
          \"video\": {
            \"video_path\": \"/path/to/video.mp4\",
            \"fps\": 1,
            \"max_frames\": 4
          }
        },
        {
          \"type\": \"text\",
          \"text\": \"What vehicles are visible in this traffic scene?\"
        }
      ]
    },
    {
      \"role\": \"assistant\",
      \"content\": [
        {
          \"type\": \"text\",
          \"text\": \"I can see several cars, a bus, and a motorcycle in this busy intersection.\"
        }
      ]
    }
  ]
}
```

#### LLaVA-Next-Video Format

```json
{
  \"prompt\": \"USER: What vehicles are visible in this traffic scene?\
<|video|>\
ASSISTANT:\",
  \"answer\": \"I can see several cars, a bus, and a motorcycle in this busy intersection.\",
  \"video_path\": \"/path/to/video.mp4\"
}
```

#### Qwen-VL2 Format

```json
{
  \"conversations\": [
    {
      \"role\": \"system\",
      \"content\": \"You are a helpful assistant.\"
    },
    {
      \"role\": \"user\",
      \"content\": [
        {
          \"type\": \"video\",
          \"video\": {
            \"video_path\": \"/path/to/video.mp4\",
            \"fps\": 1,
            \"max_frames\": 4
          }
        },
        {
          \"type\": \"text\",
          \"text\": \"What vehicles are visible in this traffic scene?\"
        }
      ]
    },
    {
      \"role\": \"assistant\",
      \"content\": [
        {
          \"type\": \"text\",
          \"text\": \"I can see several cars, a bus, and a motorcycle in this busy intersection.\"
        }
      ]
    }
  ]
}
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

### Export Log Example

```
EXPORT PROCESS LOG
==================

Step 1: Template Selection
- Selected Format: All Templates
- Current Video: traffic_scene_001.mp4
- Accepted Q&A Pairs: 12
- Templates Generated: 36 (12 × 3 formats)
- Status: Stored in memory ✓

Step 2: Processing Additional Videos
- Video: intersection_view.mp4
- Accepted Q&A Pairs: 8
- Templates Generated: 24 (8 × 3 formats)
- Total in Memory: 60 templates
- Status: Ready for export ✓

Step 3: Final Export Process
- Export Directory: /path/to/training_data/
- CSV Backup: annotations_backup_20241201_143052.csv
- CSV Updated: annotations.csv (with status columns)

File Generation:
✓ model_train_videollama2.jsonl
  - Records: 20
  - Size: 45.2 KB

✓ model_train_llava_next.jsonl
  - Records: 20
  - Size: 38.7 KB

✓ model_train_qwen.jsonl
  - Records: 20
  - Size: 47.1 KB

EXPORT SUMMARY:
==============
Total Processed Q&As: 20
Total Template Records: 60
Output Files Created: 3
Processing Time: 2.3 seconds
Status: Export Complete ✓
```

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
