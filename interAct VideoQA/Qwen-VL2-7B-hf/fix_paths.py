import json
import os

# --- Configuration ---
# This script reads 'meta_config.json', corrects the video paths, and saves a new 'meta_config_final.json'.

# The folder where your videos are stored, relative to the project root.
video_base_folder = 'interAct VideoQA/data'

# The input and output JSON files
input_json_path = 'interAct VideoQA/Qwen-VL2-7B-hf/training_data/meta_config.json'
output_json_path = 'interAct VideoQA/Qwen-VL2-7B-hf/training_data/meta_config_final.json'

# --- Mappings from incorrect old paths to correct new paths ---
# This is the critical part. You must fill this out.
# Map the incorrect directory prefix from the JSON to the real folder name.
# I have listed all the real folder names based on your 'ls' output.
# You need to find the corresponding incorrect name from 'meta_config.json' and create the mapping.
path_mappings = {
    # EXAMPLE: 'Incorrect_Path_From_JSON': 'Correct_Folder_Name_From_ls',
    'Evening1-8min_clips': 'Evening1-8min_clips', # This one seems correct already
    
    # --- TODO: Add the rest of your mappings here ---
    # 'some_other_path_in_json': '1.33pm_10.1mins_clips_60',
    # 'another_path_in_json': '1.43pm_10.1mins_clips_60',
    # 'yet_another_path': '1.59pm_10.1mins_clips_60',
    # 'and_so_on': '11.30am_13.32mins_clips_67',
    # 'etc': '2.26pm_10.1mins_clips_60',
    # 'oops': 'Evening 1_21 mins_clips_126',
    # 'almost_done': 'Morning6-9min_clips',
    # 'getting_there': 'day_4.38mins_clips',
    # 'last_one_1': 'final_Set_2',
    # 'last_one_2': 'final_Set_5',
    # 'last_one_3': 'final_Set_6',
}

# --- Script Logic ---
print(f"Reading from: {input_json_path}")
try:
    with open(input_json_path, 'r') as f:
        data = json.load(f)
except (json.JSONDecodeError, FileNotFoundError) as e:
    print(f"Error reading input file: {e}")
    exit()

print("Processing video paths...")
corrected_count = 0
unmapped_paths = set()

for item in data:
    if 'video' in item:
        original_path = item['video']
        
        # Ensure value is a string before processing
        if 'conversations' in item:
            for conv in item['conversations']:
                if 'value' in conv:
                    conv['value'] = str(conv['value'])

        old_dir = os.path.dirname(original_path)
        filename = os.path.basename(original_path)
        
        if old_dir in path_mappings:
            new_dir = path_mappings[old_dir]
            if old_dir != new_dir:
                # Use forward slashes for consistency
                item['video'] = f"{new_dir}/{filename}"
                corrected_count += 1
        else:
            unmapped_paths.add(old_dir)

print(f"Total items processed: {len(data)}")
print(f"Paths corrected based on mapping: {corrected_count}")

if unmapped_paths:
    print("\n--- WARNING: The following path prefixes from the JSON were not found in the mapping dictionary ---")
    for path in sorted(list(unmapped_paths)):
        print(f"- '{path}'")
    print("Please add these to the `path_mappings` dictionary in the fix_paths.py script and run it again.")

# Write the corrected data to the final output file
with open(output_json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Successfully created final data file at: {output_json_path}")
