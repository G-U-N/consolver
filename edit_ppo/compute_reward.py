assert 0, "git clone the repo of EditReward before running this script"

import os
import sys
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing
import math
from collections import defaultdict

# Add project root to Python path
sys.path.append("./EditReward")

from EditReward import EditRewardInferencer
# Initialize EditReward model parameters
CHECKPOINT_PATH = "EditReward-MiMo-VL-7B-SFT-2508"
CONFIG_PATH = "EditReward/EditReward/config/EditReward-MiMo-VL-7B-SFT-2508.yaml"

def read_instruction(file_path):
    """Read instruction from a text file."""
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading instruction file {file_path}: {e}")
        return None

def process_subfolder(inferencer, subfolder_path):
    """Process a single subfolder and compute its EditReward score."""
    ref_image_path = os.path.join(subfolder_path, "ref_image.jpg")
    edited_image_path = os.path.join(subfolder_path, "edited_image.jpg")
    instruction_path = os.path.join(subfolder_path, "instruction.txt")

    # Check if all required files exist
    if not all(os.path.exists(p) for p in [ref_image_path, edited_image_path, instruction_path]):
        print(f"Missing files in {subfolder_path}")
        return None

    try:
        # Load instruction
        instruction = read_instruction(instruction_path)
        if instruction is None:
            print(f"Skipping {subfolder_path} due to instruction read error")
            return None

        # Compute EditReward
        with torch.no_grad():
            rewards = inferencer.reward(
                prompts=[instruction],
                image_src=[ref_image_path],
                image_paths=[edited_image_path]
            )
        score = rewards[0][0].item()  # Extract the score from the reward tensor
        return score
    except Exception as e:
        print(f"Error processing {subfolder_path}: {e}")
        return None

def worker(chunk, gpu_id, queue, inferencer):
    
    results = []
    for task in tqdm(chunk, desc=f"Worker {gpu_id} processing"):
        root_folder, parent_folder, subfolder_path = task
        score = process_subfolder(inferencer, subfolder_path)
        if score is not None:
            results.append((root_folder, parent_folder, score))
    
    queue.put(results)

def batch_process_rewards(root_folders, models):
    """Collect tasks, divide into chunks, process in parallel, and compute rewards."""
    # Collect all tasks
    tasks = []
    for root_folder in root_folders:
        for root, dirs, files in os.walk(root_folder):
            if all(f in files for f in ["ref_image.jpg", "edited_image.jpg", "instruction.txt"]):
                parent_folder = os.path.basename(os.path.dirname(root))
                tasks.append((root_folder, parent_folder, root))
    
    if not tasks:
        print("No valid subfolders found.")
        return
    
    # Divide tasks into 8 chunks
    num_tasks = len(tasks)
    chunk_size = math.ceil(num_tasks / 8)
    chunks = [tasks[i:i + chunk_size] for i in range(0, num_tasks, chunk_size)]
    
    # Pad chunks if fewer than 8
    while len(chunks) < 8:
        chunks.append([])
    
    
    # Set up multiprocessing
    queue = multiprocessing.Queue()
    processes = []
    for i in range(8):
        p = multiprocessing.Process(target=worker, args=(chunks[i], i, queue, models[i]))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Collect all results
    all_results = []
    while not queue.empty():
        all_results.extend(queue.get())
    
    # Group results by root_folder and parent_folder
    scores_per_root = defaultdict(lambda: defaultdict(list))
    for root, parent, score in tqdm(all_results):
        scores_per_root[root][parent].append(score)
    
    # Calculate and print results for each root_folder
    for root_folder in root_folders:
        print(f"\n=== EditReward Results for {root_folder} ===")
        subfolder_scores = scores_per_root[root_folder]
        all_scores = [s for parents in subfolder_scores.values() for s in parents]
        
        for folder, scores in subfolder_scores.items():
            avg_score = np.mean(scores) if scores else 0
            print(f"Average EditReward for {folder}: {avg_score:.2f} (from {len(scores)} samples)")
        
        overall_avg = np.mean(all_scores) if all_scores else 0
        print(f"Overall Average EditReward: {overall_avg:.2f} (from {len(all_scores)} total samples)")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    root_folders = [
        
        "kontext-dpm++-3step",
        "kontext-dpm++-4step",
        "kontext-dpm++-5step",
    ]
    
    models = []
    

    for gpu_id in range(8):    
        inferencer = EditRewardInferencer(
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            device=f"cuda:{gpu_id}",
            reward_dim="overall_detail",
            rm_head_type="ranknet_multi_head"
        )
        models.append(inferencer)
    for root_folder in root_folders:
        batch_process_rewards([root_folder], models)

