import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

def decode_latents(vae, latents, batch_size=1):
    """
    Decode latents batch by batch instead of all at once.

    Args:
        vae: The VAE model
        latents: Input latents tensor of shape [batch_size, C, H, W]
        batch_size: Number of samples to process at a time (default: 1)

    Returns:
        Decoded images tensor
    """
    latents = 1 / vae.config.scaling_factor * latents
    total_samples = latents.shape[0]
    decoded_images = []

    # Process in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_latents = latents[start_idx:end_idx]

        # Decode current batch
        batch_image = vae.decode(batch_latents, return_dict=False)[0]
        batch_image = (batch_image / 2 + 0.5).clamp(0, 1)
        decoded_images.append(batch_image)

    # Concatenate all decoded batches along batch dimension
    image = torch.cat(decoded_images, dim=0)
    return image


def tensor_to_pil(tensor):
    # Move to CPU, clamp values, convert to PIL (handles channel permute and scaling internally)
    tensor = tensor.cpu().clamp(0, 1)  # Ensure values in [0, 1]
    return transforms.ToPILImage()(tensor)  # Directly converts tensor to PIL Image

def concatenate_samples(samples, is_dict=False):
    """
    Concatenate a list of samples along the batch dimension (dim=0).
    
    Args:
        samples: List of samples to concatenate (each sample is either a tensor or a dict of tensors).
        is_dict: If True, treat samples as dictionaries; otherwise, treat as tensors.
    
    Returns:
        Concatenated result (either a tensor or a dict of tensors).
    """
    if is_dict:
        # Handle dictionary case: concatenate each key's tensor
        return {
            k: torch.cat([sample[k] for sample in samples], dim=0)
            for k in samples[0].keys()
        }
    else:
        # Handle tensor case: directly concatenate
        return torch.cat(samples, dim=0)

def is_dict_like(obj):
    """
    Check if the object is a dictionary.
    
    Args:
        obj: Object to check.
    
    Returns:
        bool: True if obj is a dict, False otherwise.
    """
    return isinstance(obj, dict)