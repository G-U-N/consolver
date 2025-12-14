import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, sample_size):
        """
        Args:
            root_dir (string): Root directory containing subfolders: 'edited images', 'initial noises', 'obtained noises', 'prompts', 'ref_images'.
            sample_size (tuple): Desired sample size as (height, width).
        """
        self.root_dir = root_dir
        self.sample_size = sample_size
        
        self.edited_dir = os.path.join(root_dir, 'edited_images')
        self.initial_noise_dir = os.path.join(root_dir, 'initial_noises')
        self.obtained_noise_dir = os.path.join(root_dir, 'obtained_noises')  
        self.prompts_dir = os.path.join(root_dir, 'prompts')
        self.ref_dir = os.path.join(root_dir, 'ref_images')
        
        self.img_names = [
            f for f in os.listdir(self.edited_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.img_names.sort()
        
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.sample_size, interpolation=transforms.InterpolationMode.LANCZOS
                ), 
                transforms.CenterCrop(self.sample_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.ref_transform = transforms.Compose(
            [
                transforms.Resize(
                    self.sample_size, interpolation=transforms.InterpolationMode.LANCZOS
                ), 
                transforms.CenterCrop(self.sample_size),
                transforms.ToTensor(),

            ]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        while True:
            try:
                img_name = self.img_names[idx].strip()
                base_name = os.path.splitext(img_name)[0]  
                
                img_path = os.path.join(self.edited_dir, img_name)
                image = Image.open(img_path).convert("RGB")  
                image = self.transform(image)
                
                text_name = base_name + ".txt"  
                text_path = os.path.join(self.prompts_dir, text_name)
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                noise_name = base_name + ".pt" 
                noise_path = os.path.join(self.initial_noise_dir, noise_name)
                noise = torch.load(noise_path, map_location="cpu")[0]
                
                latent_name = base_name + ".pt"
                latent_path = os.path.join(self.obtained_noise_dir, latent_name)
                latent = torch.load(latent_path, map_location="cpu")[0]
                ref_path = os.path.join(self.ref_dir, img_name)
                ref = Image.open(ref_path).convert("RGB")
                ref = self.ref_transform(ref)
                
                if torch.isnan(latent).any() or torch.isnan(noise).any():
                    raise ValueError("NaN in tensors")
                
                break
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                idx = random.randint(0, len(self.img_names) - 1)
                continue

        return image, text, noise, latent, ref

def repeat_random_sample(batch):
    image, text, noise, tch_traj, ref = batch  

    batch_size = image.shape[0]
    random_idx = random.randint(0, batch_size - 1)

    image_out = image[random_idx:random_idx+1].repeat(batch_size, *[1 for _ in range(len(image.shape)-1)])
    text_out = [text[random_idx]] * batch_size
    noise_out = noise[random_idx:random_idx+1].repeat(batch_size, *[1 for _ in range(len(noise.shape)-1)])
    tch_traj_out = tch_traj[random_idx:random_idx+1].repeat(batch_size, *[1 for _ in range(len(tch_traj.shape)-1)])
    ref_out = ref[random_idx:random_idx+1].repeat(batch_size, *[1 for _ in range(len(ref.shape)-1)])
    return image_out, text_out, noise_out, tch_traj_out, ref_out