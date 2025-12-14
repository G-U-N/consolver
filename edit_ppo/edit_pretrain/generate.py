# generate.py
# This script loads the prepared reference images and prompts, uses FluxKontextPipeline to perform editing,
# and saves the initial latents (initial noise) and final latents (obtained "noise" after denoising).
# Note: Run this on a machine with a CUDA-enabled GPU and sufficient VRAM.

import torch
from pipeline import FluxKontextPipeline, calculate_shift, retrieve_timesteps
from diffusers.utils import load_image
import os
import numpy as np

# Load the pipeline
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

initial_noise_name = "initial_noises"
obtained_noise_name = "obtained_noises"
obtained_image_name = "edited_images"

# Create directories for saving latents
os.makedirs(f"data/{initial_noise_name}", exist_ok=True)
os.makedirs(f"data/{obtained_noise_name}", exist_ok=True)
os.makedirs(f"data/{obtained_image_name}", exist_ok=True)

# Parameters (matching the example)
height = 1024
width = 1024
num_inference_steps = 28
guidance_scale = 2.5
num_images_per_prompt = 1
max_sequence_length = 512
generator_seed = 42  # For reproducibility

with torch.no_grad():
    for i in range(2000):
        # Load reference image and prompt
        img_path = f"data/ref_images/{i}.png"
        if not os.path.exists(img_path):
            continue
        input_image = load_image(img_path).convert("RGB")
        
        prompt_path = f"data/prompts/{i}.txt"
        if not os.path.exists(prompt_path):
            continue
        with open(prompt_path, "r") as f:
            prompt = f.read().strip()
        
        # Manual execution to extract initial and final latents
        batch_size = 1
        device = "cuda"
        dtype = torch.bfloat16
        
        # Encode prompt
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        
        # Preprocess image
        image = pipe.image_processor.preprocess(input_image).to(device=device, dtype=dtype)
        
        # Prepare latents
        num_channels_latents = pipe.transformer.config.in_channels // 4
        initial_latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(
            image=image,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=torch.Generator(device).manual_seed(generator_seed),
            latents=None,
        )
        
        # Save initial latents (initial noise)
        torch.save(initial_latents, f"data/{initial_noise_name}/{i}.pt")
        
        # Prepare IDs
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)
        
        # Guidance
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None
        
        # Timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = initial_latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            base_seq_len=pipe.scheduler.config.get("base_image_seq_len", 256),
            max_seq_len=pipe.scheduler.config.get("max_image_seq_len", 4096),
            base_shift=pipe.scheduler.config.get("base_shift", 0.5),
            max_shift=pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )
        
        # Denoising loop
        latents = initial_latents
        for t in timesteps:
            latent_model_input = torch.cat([latents, image_latents], dim=1) if image_latents is not None else latents
            timestep = t.expand(batch_size).to(dtype)
            
            with torch.no_grad():
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
            noise_pred = noise_pred[:, :latents.size(1)]
            
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Save final latents (obtained "noise")
        torch.save(latents, f"data/{obtained_noise_name}/{i}.pt")
        
        # Decode latents to image
        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        # Save edited image
        image.save(f"data/{obtained_image_name}/{i}.png")

        print(f"Processed {i+1}/1000")

    print("Generation complete.")