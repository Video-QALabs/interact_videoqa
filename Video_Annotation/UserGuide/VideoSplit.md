# Video Splitting Functionality

Automatically segments long videos into 10-second clips for easier annotation and analysis.

### Process Overview

1. **Select Source Directory** - Choose folder with source videos
2. **Choose Output Directory** - Select destination for clips
3. **Automatic Processing** - System processes all supported formats

### Technical Specifications

#### Supported Formats

- **Primary**: `.mp4`, `.avi`, `.mov`, `.mkv`
- **Additional**: `.wmv`, `.flv`, `.moov`

#### Clip Parameters

- **Duration**: 10 seconds per clip
- **Quality**: Preserves original resolution and codec
- **FPS**: Maintains original frame rate
- **Naming**: Sequential (`clip_001.mp4`, `clip_002.mp4`, etc.)

### Output Structure

```
Output_Directory/
├── VideoName1_clips/
│   ├── clip_001.mp4    (0:00-0:10)
│   ├── clip_002.mp4    (0:10-0:20)
│   ├── clip_003.mp4    (0:20-0:30)
│   └── ...
├── VideoName2_clips/
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
└── ...
```

### Sample Processing Output

```
Processing Video: sample_traffic.mp4
- Duration: 2:30 (150 seconds)
- FPS: 30
- Resolution: 1920x1080
- Expected clips: 15

Clip Generation Progress:
clip_001.mp4 saved (0:00-0:10)
clip_002.mp4 saved (0:10-0:20)
clip_003.mp4 saved (0:20-0:30)
...
clip_015.mp4 saved (2:20-2:30)

Output Directory: /path/to/output/sample_traffic_clips/
Total clips generated: 15
Processing time: 45.2 seconds
Status: Complete
```
