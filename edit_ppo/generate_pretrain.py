import os
import json
import shutil
import re
from pipeline import FluxKontextPipeline
from diffusers.utils import load_image
import torch
import logging
from pathlib import Path
import torch.multiprocessing as mp
from scheduler_fm import FlowMatchGeneralDiscreteScheduler
from math import ceil

# Set up global logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

scheduler_type = "euler"  # "heun", "dpm-solver", "dpm-solver-multistep"
# Configuration
JSON_FILE = "kontext-bench/test/metadata.jsonl"  # Path to the JSONL file
IMAGE_DIR = "kontext-bench/test/images/"  # Directory containing reference images
OUTPUT_DIR = f"kontext-{scheduler_type}-5step/"  # Target directory for results
NUM_INFERENCE_STEPS = 5  # Number of steps for Flux-Kontext
GUIDANCE_SCALE = 2.5  # CFG scale for Flux-Kontext
NUM_DEVICES = 8  # Number of GPUs to use (will be capped by actual available)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_folder_name(name):
    """Convert category name to a valid folder name."""
    if not name:
        return "Unknown"
    return re.sub(r'[^a-zA-Z0-9]', '_', name.strip())

def ensure_unique_path(filepath):
    """Append a suffix to filepath if it exists to avoid overwriting."""
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath
    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{ext}"
        counter += 1
    return new_filepath

def process_instruction(entry, pipe, device_id):
    """Process a single JSON entry and generate/save results using the provided pipeline."""
    logger = logging.getLogger(f"GPU_{device_id}")
    
    try:
        # Extract fields
        file_name = entry.get("file_name")
        instruction = entry.get("instruction")
        category = entry.get("category")
        key = entry.get("key")
        
        # Validate required fields
        if not all([file_name, instruction, key]):
            logger.warning(f"Skipping entry with missing fields: {entry}")
            return
        
        # Check if image exists
        full_image_path = os.path.join(IMAGE_DIR, os.path.basename(file_name))
        if not os.path.exists(full_image_path):
            logger.error(f"Image not found: {full_image_path}")
            return
        
        # Create output subfolder based on category and key
        category_folder = sanitize_folder_name(category)
        output_subfolder = os.path.join(OUTPUT_DIR, category_folder, key)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Copy reference image
        ref_image_path = ensure_unique_path(os.path.join(output_subfolder, "ref_image.jpg"))
        shutil.copy(full_image_path, ref_image_path)
        logger.info(f"Copied reference image to {ref_image_path}")
        
        # Save instruction as text file
        instruction_path = ensure_unique_path(os.path.join(output_subfolder, "instruction.txt"))
        with open(instruction_path, "w") as f:
            f.write(instruction)
        logger.info(f"Saved instruction to {instruction_path}")
        
        # Load reference image
        ref_image = load_image(full_image_path)
        
        # Generate edited image
        try:
            edited_image = pipe(
                image=ref_image,
                prompt=instruction,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=torch.manual_seed(0),
            ).images[0]
        except Exception as e:
            logger.error(f"Failed to generate edited image for {key}: {e}")
            return
        
        # Save edited image
        edited_image_path = ensure_unique_path(os.path.join(output_subfolder, "edited_image.jpg"))
        edited_image.save(edited_image_path)
        logger.info(f"Saved edited image to {edited_image_path}")
        
    except Exception as e:
        logger.error(f"Error processing entry {key}: {e}")

def worker(entries, device_id):
    """Worker function for each process, handling a subset of entries on a specific GPU."""
    # Set CUDA device for this process
    torch.cuda.set_device(device_id)
    
    # Set up process-specific logging
    logger = logging.getLogger(f"GPU_{device_id}")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f'[GPU {device_id}] %(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    scheduler = FlowMatchGeneralDiscreteScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="scheduler",
        type = scheduler_type,
    )
    
    # Load model once per process on the assigned GPU
    try:
        logger.info(f"Loading Flux-Kontext model on GPU {device_id}...")
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16
        )
        pipe.to(f"cuda:{device_id}")
        logger.info(f"Model loaded successfully on GPU {device_id}")
    except Exception as e:
        logger.error(f"Failed to load Flux-Kontext model on GPU {device_id}: {e}")
        return
    
    # Process each entry in the chunk
    for entry in entries:
        process_instruction(entry, pipe, device_id)

def main():
    # Load JSONL file
    try:
        data = []
        with open(JSON_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line: {line}")
        logging.info(f"Loaded {len(data)} JSON entries")
    except Exception as e:
        logging.error(f"Failed to load JSONL file {JSON_FILE}: {e}")
        return
    
    if len(data) == 0:
        logging.warning("No data to process. Exiting.")
        return
    
    # Determine number of GPUs to use
    num_gpus = min(torch.cuda.device_count(), NUM_DEVICES)
    if num_gpus == 0:
        logging.error("No CUDA devices available. Exiting.")
        return
    if num_gpus < NUM_DEVICES:
        logging.warning(f"Requested {NUM_DEVICES} GPUs, but only {num_gpus} available. Using {num_gpus} GPUs.")
    
    # Split data into chunks
    chunk_size = ceil(len(data) / num_gpus)
    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    data_chunks = data_chunks[:num_gpus]  # Trim if more chunks than GPUs
    
    # Start multiprocessing with 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for i in range(num_gpus):
        p = mp.Process(target=worker, args=(data_chunks[i], i))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    logging.info("All processes completed.")

if __name__ == "__main__":
    main()