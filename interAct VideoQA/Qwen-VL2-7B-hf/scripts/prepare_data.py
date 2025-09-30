import os
import pandas as pd
import json
from pathlib import Path
import argparse
import shutil
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("⚠️ Warning: decord not available. Will skip video validation.")

def can_read_video(video_path):
    """
    Test if a video file can be read by decord (used by qwen-vl-utils).
    Returns True if readable, False otherwise.
    """
    if not DECORD_AVAILABLE:
        return True  # Skip validation if decord not available
    
    try:
        vr = decord.VideoReader(video_path)
        # Try to read at least one frame
        if len(vr) > 0:
            _ = vr[0]  # Read first frame
            return True
        return False
    except Exception as e:
        print(f"⚠️ Cannot read video {video_path}: {e}")
        return False

def backup_csv_files(root_dir):
    """
    Find all .csv files in the directory tree and move them to a backup location.
    Returns the backup directory path.
    """
    backup_dir = os.path.join(os.path.dirname(root_dir), "csv_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    csv_files_moved = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                src_path = os.path.join(dirpath, filename)
                # Create relative path structure in backup
                rel_path = os.path.relpath(dirpath, root_dir)
                backup_subdir = os.path.join(backup_dir, rel_path)
                os.makedirs(backup_subdir, exist_ok=True)
                
                dst_path = os.path.join(backup_subdir, filename)
                shutil.move(src_path, dst_path)
                print(f"📁 Moved CSV: {src_path} -> {dst_path}")
                csv_files_moved += 1
    
    if csv_files_moved > 0:
        print(f"✅ Moved {csv_files_moved} CSV files to {backup_dir}")
    else:
        print("ℹ️ No CSV files found to move")
    
    return backup_dir

def prepare_data(root_dir='data', output_file='training_data/meta_config.json'):
    """
    Prepares video QA data from multiple subdirectories containing annotations.csv
    and converts it into a single JSON file for training (Qwen-VL format).
    """
    print(f"🔍 Starting data preparation from root directory: {root_dir}")
    
    # First, backup and remove CSV files from the data directory
    backup_dir = backup_csv_files(root_dir)
    
    all_annotations = []
    video_validation_stats = {"total": 0, "readable": 0, "unreadable": 0}
    
    # Ensure the output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process CSV files from backup directory instead
    print(f"📖 Processing CSV files from backup directory: {backup_dir}")
    
    # Walk through all subdirectories in backup to find annotations.csv
    for dirpath, _, filenames in os.walk(backup_dir):
        if 'annotations.csv' in filenames:
            annotations_file = os.path.join(dirpath, 'annotations.csv')
            print(f"Processing annotations from: {annotations_file}")
            
            try:
                df = pd.read_csv(annotations_file)
                
                # Get the corresponding video directory in the original data
                rel_path = os.path.relpath(dirpath, backup_dir)
                original_video_dir = os.path.join(root_dir, rel_path)
                
                # Process each row in the CSV
                for _, row in df.iterrows():
                    video_file = row.get('video_filename') or row.get('filename') or row.get('video_file_path')
                    question = row.get('question')
                    answer = row.get('answer')

                    if not all([video_file, question, answer]):
                        print(f"⚠️ Warning: Skipping row with missing data in {annotations_file}: {row}")
                        continue

                    # Skip files that don't have the correct extension
                    if not isinstance(video_file, str) or not video_file.endswith('.mp4'):
                        print(f"⚠️ Warning: Skipping malformed video filename: {video_file}")
                        continue
                    
                    # Get the absolute path of the video file from original directory
                    absolute_video_path = os.path.abspath(os.path.join(original_video_dir, video_file))
                    
                    # Check if video file exists
                    if not os.path.exists(absolute_video_path):
                        print(f"❌ Video file not found: {absolute_video_path}")
                        video_validation_stats["unreadable"] += 1
                        continue
                    
                    # Test if video can be read by decord
                    video_validation_stats["total"] += 1
                    if not can_read_video(absolute_video_path):
                        print(f"❌ Cannot read video: {absolute_video_path}")
                        video_validation_stats["unreadable"] += 1
                        continue
                    
                    video_validation_stats["readable"] += 1
                    print(f"✅ Video readable: {absolute_video_path}")

                    # Create the JSON structure for Qwen-VL format
                    json_record = {
                        "id": f"{Path(original_video_dir).name}_{video_file.replace('.mp4', '')}",
                        "video": absolute_video_path.replace('\\', '/'),  # Use forward slashes for consistency
                        "conversations": [
                            {"from": "user", "value": f"<video>\n{question}"},
                            {"from": "assistant", "value": answer}
                        ]
                    }
                    all_annotations.append(json_record)
                    
            except Exception as e:
                print(f"❌ Error processing {annotations_file}: {e}")

    # Print validation statistics
    print(f"\n📊 Video Validation Results:")
    print(f"   Total videos found: {video_validation_stats['total']}")
    print(f"   Readable videos: {video_validation_stats['readable']}")
    print(f"   Unreadable videos: {video_validation_stats['unreadable']}")
    
    # Write all annotations to the output file as a JSON array (Qwen-VL format)
    if all_annotations:
        print(f"\n💾 Writing {len(all_annotations)} valid records to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, indent=2, ensure_ascii=False)
        print("✅ Data preparation complete!")
        print(f"📁 CSV backup location: {backup_dir}")
    else:
        print("🤷 No valid annotations found. The output file was not created.")
        print(f"📁 CSV backup location: {backup_dir}")

if __name__ == '__main__':
    # This allows the script to be run directly
    parser = argparse.ArgumentParser(description="Prepare video QA data for Qwen-VL.")
    parser.add_argument('--root_dir', default='data', help='Root directory of the dataset')
    parser.add_argument('--output_file', default='training_data/meta_config.json', help='Output JSON file for training config')
    args = parser.parse_args()
    
    prepare_data(args.root_dir, args.output_file)