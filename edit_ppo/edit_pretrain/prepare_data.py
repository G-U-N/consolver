# prepare_data.py
# This script downloads the first 1000 samples from the OmniEdit dataset, processes the source images 
# (center crop to square and resize to 1024x1024), and saves the reference images and editing instructions.

from datasets import load_dataset
from PIL import Image
import os

# Create directories
os.makedirs("data/ref_images", exist_ok=True)
os.makedirs("data/prompts", exist_ok=True)

# Load the dataset in streaming mode to handle large size
ds = load_dataset("TIGER-Lab/OmniEdit-Filtered-1.2M", split="dev", streaming=True)

i = 0
for sample in ds:
    if i >= 2000:
        break
    
    img = sample["src_img"]
    
    # Center crop to square
    w, h = img.size
    if w > h:
        left = (w - h) // 2
        img = img.crop((left, 0, left + h, h))
    elif h > w:
        top = (h - w) // 2
        img = img.crop((0, top, w, top + w))
    
    # Resize to 1024x1024
    img = img.resize((1024, 1024), Image.LANCZOS)
    
    # Save reference image
    img.save(f"data/ref_images/{i}.png")
    
    # Save editing instruction (take the first prompt if multiple)
    prompt = sample["edited_prompt_list"][0] if sample["edited_prompt_list"] else ""
    with open(f"data/prompts/{i}.txt", "w") as f:
        f.write(prompt)
    
    i += 1

print("Prepared 1000 reference images and prompts.")