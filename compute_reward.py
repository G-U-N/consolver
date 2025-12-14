import os
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import json
import argparse
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process

# Import your reward model functions
from reward_model import load_reward_model, calculate_reward


# def find_image_pairs(root_dir1, root_dir2, image_name="edited_image.jpg"):
#     """
#     Recursively find image pairs with the same relative path in two root directories
    
#     Args:
#         root_dir1: First root directory path
#         root_dir2: Second root directory path
#         image_name: Image filename to search for
    
#     Returns:
#         image_pairs: [(path1, path2), ...] List of image pair paths
#     """
#     root1 = Path(root_dir1)
#     root2 = Path(root_dir2)
    
#     image_pairs = []
    
#     # Recursively find all images with the specified name in the first directory
#     for img_path1 in root1.rglob(image_name):
#         # Calculate relative path
#         rel_path = img_path1.relative_to(root1)
        
#         # Construct corresponding path in the second directory
#         img_path2 = root2 / rel_path
        
#         # Check if the second image exists
#         if img_path2.exists():
#             image_pairs.append((str(img_path1), str(img_path2)))
#         else:
#             print(f"Warning: Cannot find paired image {img_path2}")
    
#     print(f"Found {len(image_pairs)} image pairs in total")
#     return image_pairs


def find_image_pairs(root_dir1: str, root_dir2: str, *args):
    """
    Recursively find pairs of PNG files with the same relative path in two root directories.

    Args:
        root_dir1: Path to the first root directory
        root_dir2: Path to the second root directory

    Returns:
        List[Tuple[str, str]]: List of tuples containing paired file paths (path1, path2)
    """
    root1 = Path(root_dir1)
    root2 = Path(root_dir2)

    # Collect all PNG files in root1: {relative_path: absolute_path}
    png_in_root1 = {p.relative_to(root1): p for p in root1.rglob("*.png")}

    pairs = []
    for rel_path, path1 in png_in_root1.items():
        path2 = root2 / rel_path
        if path2.exists():  # Ensure the corresponding file exists in root2
            pairs.append((str(path1), str(path2)))
        else:
            print(f"Warning: Paired file not found: {path2}")

    print(f"Found {len(pairs)} PNG pairs in total")
    return pairs


def load_image_tensor(image_path, device):
    """
    Load image and convert to tensor in [0,1] range
    
    Args:
        image_path: Image file path
        device: torch device
    
    Returns:
        image_tensor: [C, H, W] tensor in [0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).to(device)
    return img_tensor


def process_gpu_worker(gpu_id, image_pairs_chunk, reward_type, batch_size, result_queue):
    """
    Worker function for processing image pairs on a single GPU
    
    Args:
        gpu_id: GPU device ID
        image_pairs_chunk: Chunk of image pairs assigned to this GPU
        reward_type: Type of reward to calculate
        batch_size: Batch size for processing
        result_queue: Queue to put results in
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"GPU {gpu_id}: Processing {len(image_pairs_chunk)} pairs for {reward_type}")
        
        # Load reward model on this GPU
        model, processor = load_reward_model(reward_type, device)
        
        # Process image pairs
        scores = []
        pred_tensors = []
        target_tensors = []
        
        # Load all images for this chunk
        for path1, path2 in image_pairs_chunk:
            try:
                pred_tensor = load_image_tensor(path1, device)
                target_tensor = load_image_tensor(path2, device)
                pred_tensors.append(pred_tensor)
                target_tensors.append(target_tensor)
            except Exception as e:
                print(f"GPU {gpu_id}: Failed to load image {path1} or {path2}: {e}")
                continue
        
        # Process in batches
        for i in range(0, len(pred_tensors), batch_size):
            batch_pred = torch.stack(pred_tensors[i:i+batch_size])
            batch_target = torch.stack(target_tensors[i:i+batch_size])
            
            try:
                # Calculate reward
                rewards = calculate_reward(
                    reward_type,
                    model,
                    processor,
                    batch_pred,
                    batch_target,
                    device
                )
                
                scores.extend(rewards.flatten().cpu().numpy().tolist())
                
            except Exception as e:
                print(f"GPU {gpu_id}: Failed to calculate reward: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clean up
        del model
        del processor
        torch.cuda.empty_cache()
        
        # Put results in queue
        result_queue.put({
            'gpu_id': gpu_id,
            'reward_type': reward_type,
            'scores': scores,
            'num_processed': len(scores)
        })
        
        print(f"GPU {gpu_id}: Completed {len(scores)} samples for {reward_type}")
        
    except Exception as e:
        print(f"GPU {gpu_id}: Worker failed with error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'gpu_id': gpu_id,
            'reward_type': reward_type,
            'scores': [],
            'num_processed': 0,
            'error': str(e)
        })


def calculate_rewards_multigpu(image_pairs, reward_types, num_gpus=8, batch_size=8):
    """
    Calculate rewards for all image pairs using multiple GPUs
    
    Args:
        image_pairs: [(path1, path2), ...] List of image pair paths
        reward_types: List of reward types to calculate
        num_gpus: Number of GPUs to use
        batch_size: Batch size per GPU
    
    Returns:
        results: {reward_type: [scores]} Dictionary of results
    """
    results = {r_type: [] for r_type in reward_types}
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No GPUs available, falling back to single CPU processing")
        return calculate_rewards_single_device(image_pairs, reward_types, 'cpu', batch_size)
    
    num_gpus = min(num_gpus, available_gpus)
    print(f"Using {num_gpus} GPUs for parallel processing")
    
    for reward_type in reward_types:
        print(f"\n{'='*50}")
        print(f"Calculating {reward_type} reward with {num_gpus} GPUs...")
        print(f"{'='*50}")
        
        # Split image pairs across GPUs
        chunk_size = len(image_pairs) // num_gpus
        image_chunks = []
        
        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(image_pairs)
            image_chunks.append(image_pairs[start_idx:end_idx])
            print(f"GPU {i}: Assigned {len(image_chunks[i])} image pairs")
        
        # Create result queue
        result_queue = mp.Queue()
        
        # Start worker processes
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=process_gpu_worker,
                args=(gpu_id, image_chunks[gpu_id], reward_type, batch_size, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results from queue
        all_scores = []
        for _ in range(num_gpus):
            result = result_queue.get()
            if 'error' in result:
                print(f"GPU {result['gpu_id']} encountered error: {result['error']}")
            else:
                all_scores.extend(result['scores'])
                print(f"GPU {result['gpu_id']}: Collected {result['num_processed']} scores")
        
        results[reward_type] = all_scores
        print(f"{reward_type} completed: {len(all_scores)} total samples")
    
    return results


def calculate_rewards_single_device(image_pairs, reward_types, device, batch_size=8):
    """
    Calculate rewards using a single device (fallback method)
    
    Args:
        image_pairs: [(path1, path2), ...] List of image pair paths
        reward_types: List of reward types
        device: Device to use ('cpu' or 'cuda:X')
        batch_size: Batch size
    
    Returns:
        results: {reward_type: [scores]} Dictionary of results
    """
    device = torch.device(device)
    results = {r_type: [] for r_type in reward_types}
    
    for reward_type in reward_types:
        print(f"\n{'='*50}")
        print(f"Calculating {reward_type} reward on {device}...")
        print(f"{'='*50}")
        
        # Load reward model
        model, processor = load_reward_model(reward_type, device)
        
        # Prepare all image pairs
        pred_tensors = []
        target_tensors = []
        
        print("Loading images...")
        for path1, path2 in tqdm(image_pairs):
            try:
                pred_tensor = load_image_tensor(path1, device)
                target_tensor = load_image_tensor(path2, device)
                pred_tensors.append(pred_tensor)
                target_tensors.append(target_tensor)
            except Exception as e:
                print(f"Failed to load image {path1} or {path2}: {e}")
                continue
        
        # Process in batches
        print(f"Calculating rewards ({len(pred_tensors)} samples)...")
        for i in tqdm(range(0, len(pred_tensors), batch_size)):
            batch_pred = torch.stack(pred_tensors[i:i+batch_size])
            batch_target = torch.stack(target_tensors[i:i+batch_size])
            
            try:
                # Calculate reward
                rewards = calculate_reward(
                    reward_type,
                    model,
                    processor,
                    batch_pred,
                    batch_target,
                    device
                )
                
                # Save results
                results[reward_type].extend(rewards.flatten().cpu().numpy().tolist())
                
            except Exception as e:
                print(f"Failed to calculate reward: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clean up memory
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"{reward_type} completed: {len(results[reward_type])} samples")
    
    return results


def calculate_statistics(results):
    """
    Calculate statistics for results
    
    Args:
        results: {reward_type: [scores]} Dictionary of scores
    
    Returns:
        stats: {reward_type: {mean, std, min, max}} Dictionary of statistics
    """
    stats = {}
    
    for reward_type, scores in results.items():
        if len(scores) > 0:
            scores_array = np.array(scores)
            stats[reward_type] = {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'median': float(np.median(scores_array)),
                'count': len(scores)
            }
        else:
            stats[reward_type] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'count': 0
            }
    
    return stats


def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Calculate reward means for image pairs')
    parser.add_argument('--dir1', type=str, required=True,
                        help='First root directory path')
    parser.add_argument('--dir2', type=str, required=True,
                        help='Second root directory path')
    parser.add_argument('--image_name', type=str, default='edited_image.jpg',
                        help='Image filename to search for')
    parser.add_argument('--reward_types', type=str, nargs='+',
                        default=['depth', 'inception', 'segmentation',
                                'image_psnr', 'clip', 'dino'],
                        help='List of reward types to calculate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use for parallel processing')
    parser.add_argument('--output', type=str, default='reward_results.json',
                        help='Output JSON file path')
    parser.add_argument('--single_gpu', action='store_true',
                        help='Use single GPU mode instead of multi-GPU')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for single GPU mode')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Directory 1: {args.dir1}")
    print(f"  Directory 2: {args.dir2}")
    print(f"  Image name: {args.image_name}")
    print(f"  Reward types: {args.reward_types}")
    print(f"  Batch size: {args.batch_size}")
    
    # Find image pairs
    print(f"\nSearching for image pairs...")
    image_pairs = find_image_pairs(args.dir1, args.dir2, args.image_name)
    print(image_pairs)
    
    if len(image_pairs) == 0:
        print("No image pairs found, exiting.")
        return
    
    # Calculate rewards
    if args.single_gpu:
        print(f"\nUsing single device mode: {args.device}")
        results = calculate_rewards_single_device(
            image_pairs,
            args.reward_types,
            args.device,
            args.batch_size
        )
    else:
        print(f"\nUsing multi-GPU mode with {args.num_gpus} GPUs")
        results = calculate_rewards_multigpu(
            image_pairs,
            args.reward_types,
            args.num_gpus,
            args.batch_size
        )
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Print results
    print(f"\n{'='*60}")
    print("Reward Statistics:")
    print(f"{'='*60}")
    for reward_type, stat in stats.items():
        print(f"\n{reward_type}:")
        print(f"  Sample count: {stat['count']}")
        print(f"  Mean:         {stat['mean']:.4f}")
        print(f"  Std Dev:      {stat['std']:.4f}")
        print(f"  Min:          {stat['min']:.4f}")
        print(f"  Max:          {stat['max']:.4f}")
        print(f"  Median:       {stat['median']:.4f}")
    
    # Save results
    output_data = {
        'statistics': stats,
        'raw_scores': results,
        'config': {
            'dir1': args.dir1,
            'dir2': args.dir2,
            'image_name': args.image_name,
            'reward_types': args.reward_types,
            'batch_size': args.batch_size,
            'num_gpus': args.num_gpus if not args.single_gpu else 1,
            'num_pairs': len(image_pairs)
        }
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()