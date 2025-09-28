import os
import pandas as pd
import json
from pathlib import Path
import argparse

def prepare_data(root_dir='data', output_file='training_data/meta_config.json'):
    """
    Prepares video QA data from multiple subdirectories containing annotations.csv
    and converts it into a single JSONL file for training.
    """
    print(f"🔍 Starting data preparation from root directory: {root_dir}")
    
    all_annotations = []
    
    # Ensure the output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Walk through all subdirectories to find annotations.csv
    for dirpath, _, filenames in os.walk(root_dir):
        if 'annotations.csv' in filenames:
            annotations_file = os.path.join(dirpath, 'annotations.csv')
            print(f"Processing annotations from: {annotations_file}")
            
            try:
                df = pd.read_csv(annotations_file)
                
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
                    
                    # Get the relative path of the video from the root_dir
                    # This assumes the video is in the same directory as the annotations.csv
                    relative_video_path = os.path.relpath(os.path.join(dirpath, video_file), root_dir)

                    # Create the JSONL structure
                    json_record = {
                        "id": f"{Path(dirpath).name}_{video_file.replace('.mp4', '')}",
                        "video": relative_video_path.replace('\\', '/'),  # Use forward slashes for consistency
                        "conversations": [
                            {"from": "human", "value": f"<video>\n{question}"},
                            {"from": "gpt", "value": answer}
                        ]
                    }
                    all_annotations.append(json_record)
                    
            except Exception as e:
                print(f"❌ Error processing {annotations_file}: {e}")

    # Write all annotations to the output file
    if all_annotations:
        print(f"Writing {len(all_annotations)} records to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            # First, write the metadata line that the training script expects
            meta_data = {
                "custom_video_qa": True,
                "root": root_dir,
                "annotation": output_file
            }
            f.write(json.dumps(meta_data) + '\n')

            # Then write all the annotation records
            for record in all_annotations:
                f.write(json.dumps(record) + '\n')
        print("✅ Data preparation complete!")
    else:
        print("🤷 No annotations found. The output file was not created.")

if __name__ == '__main__':
    # This allows the script to be run directly
    parser = argparse.ArgumentParser(description="Prepare video QA data.")
    parser.add_argument('--root_dir', default='data', help='Root directory of the dataset')
    parser.add_argument('--output_file', default='training_data/meta_config.json', help='Output JSONL file for training config')
    args = parser.parse_args()
    
    prepare_data(args.root_dir, args.output_file)