import torch
from typing import Union, List, Optional
from torch.utils.checkpoint import checkpoint
import contextlib

def denoise_diffusion(
    text_encoder,
    scheduler,
    unet,
    noise: torch.FloatTensor,
    text: Union[str, List[str]],
    tokenizer,
    cfg: float = 3,  # guidance scale
    num_inference_steps: int = 50,
    gradient_checkpointing = False,
) -> torch.FloatTensor:

    if isinstance(text, str):
        batch_size = 1
        text = [text]
    else:
        batch_size = len(text)
    device = noise.device

    with torch.no_grad():
        text_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = text_encoder(text_input_ids)[0]
    prompt_embeds_txt = prompt_embeds  

    do_classifier_free_guidance = cfg > 1.0
    if do_classifier_free_guidance:
        with torch.no_grad():
            uncond_tokens = [""] * batch_size
            uncond_inputs = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_inputs.input_ids.to(device)
            negative_prompt_embeds = text_encoder(uncond_input_ids)[0]
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    latents = noise.clone()
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps


    # conds_ = []
    conds_ = dict(x=[], epsilon=[])
    actions_ = []
    probs_ = []
    masks_ = []
    for i, t in enumerate(timesteps):
        unet_apply = unet
        ctx = contextlib.nullcontext()

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with ctx:
            if gradient_checkpointing:
                def unet_call(latent_model_input, t, encoder_hidden_states, *args, **kwargs):
                    return unet_apply(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                        *args,
                        **kwargs
                    )[0]

                noise_pred = checkpoint(
                    unet_call,
                    latent_model_input,
                    t,
                    prompt_embeds,
                    use_reentrant=False
                )
            else:
                noise_pred = unet_apply(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False
                )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond = noise_pred_uncond.detach()
            
            noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
        

        latents, actions, probs, conds, masks = scheduler.step(noise_pred, t, latents, return_dict=False)
        # conds_.append(conds.unsqueeze(1))
        if i > 0:

            conds_["x"].append(conds["x"].unsqueeze(1))
            conds_["epsilon"].append(conds["epsilon"].unsqueeze(1))
            probs_.append(probs.unsqueeze(1))
            actions_.append(actions.unsqueeze(1))
            masks_.append(masks.unsqueeze(1))

        latents = latents.detach() # detach() is very important
    # conds_ = torch.cat(conds_, dim=1)
    conds_ = {k: torch.cat(v, dim=1) for k,v in conds_.items()}
    probs_ = torch.cat(probs_, dim=1)
    actions_ = torch.cat(actions_, dim=1)
    masks_ = torch.cat(masks_, dim=1)

    return latents, conds_, probs_, actions_, masks_, prompt_embeds_txt

if __name__ == "__main__":
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import UNet2DConditionModel, DDIMScheduler

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    teacher_unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    teacher_unet = teacher_unet.to(device)
    scheduler = scheduler.to(device)

    noise = torch.randn(1, 4, 64, 64).to(device)  # [batch_size, channels, height, width]
    text = "a photo of a cat"

    latents = denoise_diffusion(
        text_encoder=text_encoder,
        scheduler=scheduler,
        teacher_unet=teacher_unet,
        unet=unet,
        noise=noise,
        text=text,
        tokenizer=tokenizer,
        cfg=3,
        num_inference_steps=50
    )
    print("Denoised latents shape:", latents.shape)