"""
Converts the MultiCaRe image_metadata.json (JSON Lines format) into a standard 
LLaVA-style conversational JSON array.
"""

import os
import json

def main():
    # Setup paths relative to the script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "whole_multicare_dataset", "vlm_mri_subset")
    input_file = os.path.join(data_dir, "image_metadata.json")
    output_file = os.path.join(base_dir, "data", "vlm_conversational_dataset.json")

    print(f"Reading metadata from: {input_file}")

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return

    conversational_dataset = []
    skipped_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON at line {line_num}. Skipping.")
                skipped_count += 1
                continue
            
            file_id = record.get("file_id")
            caption = record.get("caption")
            
            raw_path = record.get("file_path", "")
            
            if "images/" in raw_path:
                relative_suffix = raw_path.split("images/")[-1]
                # Canonical path relative to the repo root
                actual_rel_path = os.path.join("data", "whole_multicare_dataset", "vlm_mri_subset", "images", relative_suffix)
                abs_path = os.path.join(base_dir, actual_rel_path)
            else:
                print(f"Warning: Unexpected file_path format '{raw_path}' at line {line_num}. Skipping.")
                skipped_count += 1
                continue

            if not file_id or not caption:
                skipped_count += 1
                continue

            # Verify the image file actually exists to avoid dataset corruption
            if not os.path.exists(abs_path):
                skipped_count += 1
                continue

            # Construct LLaVA-style conversation schema
            conv_entry = {
                "id": file_id,
                "image": actual_rel_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nDescribe the findings in this medical image."
                    },
                    {
                        "from": "gpt",
                        "value": caption
                    }
                ]
            }
            conversational_dataset.append(conv_entry)

    print(f"Processed {len(conversational_dataset)} valid entries.")
    print(f"Skipped {skipped_count} malformed or missing entries.")

    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(conversational_dataset, out_f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved conversational dataset to: {output_file}")

if __name__ == "__main__":
    main()
