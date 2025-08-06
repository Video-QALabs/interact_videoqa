# Video Analysis

<img width="1205" height="830" alt="image" src="../assets/VideoAnalysis.png" />

Comprehensive video playback system supporting both continuous playback and frame-by-frame analysis.

### Playback Controls

| Control             | Function         | Behavior                                      |
| ------------------- | ---------------- | --------------------------------------------- |
| ▶️ **Play/Pause**   | Toggle playback  | Disables auto-play, enables questions         |
| ⏹️ **Stop**         | Stop and reset   | Returns to frame 0, shows first frame         |
| ⏩ **Fast-Forward** | Skip +10 seconds | Calculates based on FPS (default: 300 frames) |
| ⏪ **Rewind**       | Skip -10 seconds | Calculates based on FPS (default: 300 frames) |

### Analysis Modes

#### Video Playback Mode

- Continuous video playback
- Standard media controls
- Auto-play on load
- Questions disabled during auto-play

#### Frame Analysis Mode

- Frame-by-frame navigation
- Previous/Next frame controls
- All frames loaded in memory
- Questions always enabled

### Auto-Play Behavior

```
1. Video loads → Auto-play starts
2. Questions disabled → \"Auto-playing video - questions disabled\"
3. Video completes → Questions enabled
4. Manual control → Auto-play disabled, questions enabled
```

### Status Messages

- `Loading video...` - Video initialization
- `Playing video...` - Active playback
- `Video paused` - Playback paused
- `Fast forwarded to X.X seconds` - Forward skip
- `Rewound to X.X seconds` - Backward skip
- `Video completed - Questions now enabled` - Playback finished
