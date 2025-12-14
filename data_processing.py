import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, sample_size):
        """
        Args:
            img_dir (string): Directory with all the images and text files.
            sample_size (tuple): Desired sample size as (height, width).
        """
        self.img_dir = img_dir
        self.sample_size = sample_size
        self.img_names = [
            f.replace(".txt", ".png")
            for f in os.listdir(img_dir)
            if f.endswith((".txt"))
        ]
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

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        while True:
            try:
                img_name = self.img_names[idx].strip()
                img_path = os.path.join(self.img_dir, img_name)
                image = Image.open(img_path).convert("RGB")  
                image = self.transform(image)
                text_name = img_name.rsplit(".", 1)[0] + ".txt"
                text_path = os.path.join(self.img_dir, text_name)
                with open(text_path, "r") as f:
                    text = f.read().strip()
                latent_name = "latent_" + img_name.rsplit(".", 1)[0] + ".pth"
                noise_name = "noise_" + img_name.rsplit(".", 1)[0] + ".pth"
                noise_path = os.path.join(self.img_dir, noise_name)
                latent_path = os.path.join(self.img_dir, latent_name)
                noise = torch.load(noise_path, map_location="cpu")
                latent = torch.load(latent_path, map_location="cpu")
                if torch.isnan(latent).any():
                    raise FileNotFoundError
                break
            except:
                idx = random.randint(0, len(self.img_names) - 1)
                print("error")
                continue

        return image, text, noise, latent

def repeat_random_sample(batch):

    image, text, noise, tch_traj = batch

    batch_size = image.shape[0]

    random_idx = random.randint(0, batch_size - 1)

    image_out = image[random_idx:random_idx+1].repeat(batch_size, *[1 for _ in range(len(image.shape)-1)])

    text_out = [text[random_idx]] * batch_size

    noise_out = noise[random_idx:random_idx+1].repeat(batch_size, *[1 for _ in range(len(noise.shape)-1)])

    tch_traj_out = tch_traj[random_idx:random_idx+1].repeat(batch_size, *[1 for _ in range(len(tch_traj.shape)-1)])

    return image_out, text_out, noise_out, tch_traj_out