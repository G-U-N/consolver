import torch
from typing import Union, List, Optional
from torch.utils.checkpoint import checkpoint
import contextlib
import numpy as np
from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps
from diffusers import FluxKontextPipeline
from PIL import Image

@torch.no_grad()
def denoise_diffusion(
    scheduler,
    pipe,
    noise: torch.FloatTensor,
    text: Union[str, List[str]],
    image: Union[torch.FloatTensor, List[Image.Image]],
    cfg: float = 2.5,  # guidance scale
    num_inference_steps: int = 28,
    gradient_checkpointing = False,
    use_naive_scheduler = False,
) -> torch.FloatTensor:


    if isinstance(text, str):
        batch_size = 1
        text = [text]
    else:
        batch_size = len(text)
    device = noise.device
    dtype = noise.dtype
    guidance_scale = cfg
    height = 1024
    width = 1024
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=text,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

    # Preprocess image
    image = pipe.image_processor.preprocess(image).to(device=device, dtype=dtype)
    
    # Prepare latents
    

    
    num_channels_latents = pipe.transformer.config.in_channels // 4
    noise = pipe._pack_latents(noise, batch_size, num_channels_latents, height // 8, width // 8)
    initial_latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(
        image=image,
        batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=None,
        latents=noise,
    )
    assert (initial_latents == noise).all()
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
        base_seq_len=scheduler.config.get("base_image_seq_len", 256),
        max_seq_len=scheduler.config.get("max_image_seq_len", 4096),
        base_shift=scheduler.config.get("base_shift", 0.5),
        max_shift=scheduler.config.get("max_shift", 1.15),
    )
    
    
    timesteps, _ = retrieve_timesteps(
        scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )
    
    conds_ = dict(x=[], epsilon=[])
    actions_ = []
    probs_ = []
    masks_ = []
    # Denoising loop
    latents = initial_latents
    t_next = None
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents, image_latents], dim=1) if image_latents is not None else latents
        timestep = t.expand(batch_size).to(dtype)
        
        if t_next is not None:
            timestep = t_next.expand(batch_size).to(dtype)
        ctx = torch.no_grad()
        
        with ctx:
            if gradient_checkpointing:
                def transformer_call(latent_model_input, timestep, guidance,  pooled_prompt_embeds, prompt_embeds, text_ids, latent_ids, *args, **kwargs):
                    return pipe.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_ids,
                            return_dict=False,
                        )[0]

                noise_pred = checkpoint(
                    transformer_call,
                    latent_model_input,
                    timestep,
                    guidance,
                    pooled_prompt_embeds,
                    prompt_embeds,
                    text_ids,
                    latent_ids,
                    use_reentrant=False
                )
            else:
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
        if use_naive_scheduler:
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        else:
            latents, actions, probs, conds, masks = scheduler.step(noise_pred, t, latents, return_dict=False)
        # latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if i > 0 and not use_naive_scheduler:
            conds_["x"].append(conds["x"].unsqueeze(1))
            conds_["epsilon"].append(conds["epsilon"].unsqueeze(1))
            probs_.append(probs.unsqueeze(1))
            actions_.append(actions.unsqueeze(1))
            masks_.append(masks.unsqueeze(1))

        latents = latents.detach() 
    latents_output = latents.detach()
    
    # Decode latents to image
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    pred_images = pipe.vae.decode(latents, return_dict=False)[0]
    pred_images = pipe.image_processor.postprocess(pred_images, output_type="pil")

    if not use_naive_scheduler:
        conds_ = {k: torch.cat(v, dim=1) for k,v in conds_.items()}
        probs_ = torch.cat(probs_, dim=1)
        actions_ = torch.cat(actions_, dim=1)
        masks_ = torch.cat(masks_, dim=1)
    if not use_naive_scheduler:
        return latents_output, pred_images, conds_, probs_, actions_, masks_
    else:
        return latents_output, pred_images
if __name__ == "__main__":

    from scheduler_fmppo import FMPPOScheduler
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to(device)
    
    noise = torch.randn(1, 16, 128, 128).to(device).to(torch.bfloat16)  # [batch_size, channels, height, width]
    text = "Make the image look like it's from an ancient Egyptian mural."
    image = Image.open("../edit_pretrain/data/ref_images/0.png")
    factor_net_kwargs = dict(hidden_dim=256, num_actions=10)
    scheduler = FMPPOScheduler.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", subfolder="scheduler", order_dim=2, scaler_dim=0, factor_net_kwargs=factor_net_kwargs)
    scheduler.factor_net.to(device)
    latents, pred_images, conds_, probs_, actions_, masks_ = denoise_diffusion(
        scheduler=scheduler,
        pipe=pipe,
        image = [image],
        noise=noise,
        text=[text],
        num_inference_steps=4
    )
    
    pred_images[0].save("denoised_latent-ppo-4step-mu2.0.jpg")
    print("Denoised latents shape:", latents.shape)