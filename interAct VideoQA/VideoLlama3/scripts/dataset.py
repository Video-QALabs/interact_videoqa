import os
import json
import shutil
import pandas as pd
from pathlib import Path

def build_json(
    annotations_csv: str,
    source_video_dir: str,
    out_root: str,
    status_filter=('accepted',),
    copy_missing_only: bool = True,
    use_absolute_paths: bool = True,
    separate_conversations: bool = True
):
    """
    Build JSONL dataset with flexible formatting options:
      - out_root/videos/ : copied videos (all .mp4 from source_video_dir)
      - out_root/custom.jsonl : training JSONL
    
    Args:
        annotations_csv: Path to CSV with video annotations
        source_video_dir: Directory containing source video files
        out_root: Output root directory
        status_filter: Tuple of accepted status values
        copy_missing_only: Only copy videos that don't exist in destination
        use_absolute_paths: Use absolute paths in JSONL (True) or relative "videos/" paths (False)
        separate_conversations: Create separate JSON object for each Q&A pair (True) or group by video (False)
    """

    out_root = Path(out_root)
    videos_dir = out_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(annotations_csv)

    needed_cols = {'video_file_path', 'question', 'answer'}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Filter by status
    if 'status' in df.columns and status_filter:
        df = df[df['status'].astype(str).str.lower().isin([s.lower() for s in status_filter])]

    # Clean strings
    for col in ['video_file_path', 'question', 'answer']:
        df[col] = df[col].astype(str).str.strip()

    # Remove rows with empty questions or answers
    df = df[(df['question'] != '') & (df['answer'] != '') & 
            (df['question'] != 'nan') & (df['answer'] != 'nan')]

    items = []
    copied_videos = set()

    if separate_conversations:
        # Create separate JSON object for each Q&A pair
        item_id = 0
        for _, row in df.iterrows():
            video_rel = row['video_file_path']
            src = Path(source_video_dir) / Path(video_rel).name
            
            if not src.exists():
                print(f"[WARN] Missing video {video_rel}, skipping...")
                continue

            dst = videos_dir / src.name
            if dst.name not in copied_videos:
                if not dst.exists() or not copy_missing_only:
                    shutil.copy2(src, dst)
                    print(f"[INFO] Copied {src.name}")
                copied_videos.add(dst.name)

            # Determine video path format
            if use_absolute_paths:
                video_path = [str(dst.absolute())]
            else:
                video_path = f"videos/{dst.name}"

            # Create individual conversation item
            item = {
                "id": f"{item_id:x}",  # Use hex format like "0a", "0b", etc.
                "video": video_path,
                "conversations": [
                    {"from": "human", "value": f"<video>\n {row['question']}"},
                    {"from": "gpt", "value": row['answer']}
                ]
            }
            items.append(item)
            item_id += 1

    else:
        # Group conversations by video (original behavior)
        grouped = df.groupby('video_file_path', sort=False)
        next_id = 0
        
        for video_rel, gdf in grouped:
            src = Path(source_video_dir) / Path(video_rel).name
            if not src.exists():
                print(f"[WARN] Missing video {video_rel}, skipping JSON entry...")
                continue

            dst = videos_dir / src.name
            if not dst.exists() or not copy_missing_only:
                shutil.copy2(src, dst)

            conversations = []
            for _, row in gdf.iterrows():
                q, a = row['question'], row['answer']
                if not q or not a:
                    continue
                conversations.append({"from": "human", "value": "<video>\n " + q})
                conversations.append({"from": "gpt", "value": a})

            if conversations:
                if use_absolute_paths:
                    video_path = [str(dst.absolute())]
                else:
                    video_path = f"videos/{dst.name}"
                    
                items.append({
                    "id": next_id,
                    "video": video_path,
                    "conversations": conversations
                })
                next_id += 1

    # Copy any extra .mp4s not in CSV
    for mp4 in Path(source_video_dir).glob("*.mp4"):
        dst = videos_dir / mp4.name
        if not dst.exists() or not copy_missing_only:
            shutil.copy2(mp4, dst)

    # Write JSONL file
    out_jsonl = out_root / "custom.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] Wrote {out_jsonl} with {len(items)} items.")
    print(f"[INFO] Total {len(list(videos_dir.glob('*.mp4')))} mp4 files in {videos_dir}")

