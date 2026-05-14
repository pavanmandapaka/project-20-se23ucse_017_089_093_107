import os
import glob
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

from database import log_inference

def main():
    # 1. Setup Directories
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, "data", "whole_multicare_dataset", "vlm_mri_subset", "images")
    results_dir = os.path.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Find ALL .webp images recursively
    # This looks through PMC1, PMC2, etc., and all subfolders automatically
    search_pattern = os.path.join(data_dir, "**", "*.webp")
    images = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(images)} images to process.")
    
    if not images:
        print(f"No .webp images found in {data_dir}. Check your path!")
        return

    # 3. Load Model (CPU mode for stability)
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cpu" 
    model.to(device)
    
    # 4. Process and Save
    results_file = os.path.join(results_dir, "preliminary_results.txt")
    with open(results_file, "w") as f:
        for img_path in images:
            try:
                raw_image = Image.open(img_path).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                # Use the filename for the log
                img_name = os.path.basename(img_path)
                f.write(f"File: {img_name} | Caption: {caption}\n")
                f.flush() # Saves as it goes so you don't lose data if it crashes
            except Exception as e:
                continue 

    print(f"Done! Results saved to {results_file}")

if __name__ == "__main__":
    main()