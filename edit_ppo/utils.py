import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

def tensor_to_pil(tensor):
    # Move to CPU, clamp values, convert to PIL (handles channel permute and scaling internally)
    tensor = tensor.cpu().clamp(0, 1)  # Ensure values in [0, 1]
    return transforms.ToPILImage()(tensor)  # Directly converts tensor to PIL Image

def decode_latents(pipe, latents, batch_size=8):
    """
    Decode latents batch by batch instead of all at once.

    Args:
        vae: The VAE model
        latents: Input latents tensor of shape [batch_size, C, H, W]
        batch_size: Number of samples to process at a time (default: 1)

    Returns:
        Decoded images tensor
    """
    latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    pred_images = pipe.vae.decode(latents, return_dict=False)[0]
    pred_images = (pred_images / 2 + 0.5).clamp(0, 1)
    
    return pred_images

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