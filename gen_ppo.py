from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import argparse
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import torch
from scheduler_ppo import PPOScheduler
from diffusers import DEISMultistepScheduler, UniPCMultistepScheduler,DPMSolverMultistepScheduler, PNDMScheduler, DDIMScheduler

import json
from diffusers import StableDiffusionPipeline
import multiprocessing as mp



num_device = 8
num_processes = 8



# factors printed by AMED Solver weights
SCHEDULES = {
    4: {
        "amed":      [999, 694, 500, 110, 0],
        "grad_scale":[1.0, 0.991, 1.0, 0.9912, 1.0],
        "time_scale":[1.0, 1.0333, 1.0, 0.9861, 1.0],
    },
    6: {
        "amed":      [999, 758, 666, 495, 333, 107, 0],
        "grad_scale":[1.0, 0.9924, 1.0, 0.9916, 1.0, 0.9906, 1.0],
        "time_scale":[1.0, 1.052, 1.0, 0.9998, 1.0, 0.9781, 1.0],
    },
    8: {
        "amed":      [999, 831, 749, 623, 500, 394, 250, 88, 0],
        "grad_scale":[1.0, 0.9976, 1.0, 0.991, 1.0, 0.9907, 1.0, 0.9905, 1.0],
        "time_scale":[1.0, 1.0257, 1.0, 0.9989, 1.0, 1.0022, 1.0, 0.9747, 1.0],
    },
    10: {
        "amed":      [999, 885, 799, 705, 599, 492, 400, 329, 200, 73, 0],
        "grad_scale":[1.0, 0.9974, 1.0, 0.9904, 1.0, 0.991, 1.0, 0.9905, 1.0, 0.9904, 1.0],
        "time_scale":[1.0, 0.9872, 1.0, 1.0152, 1.0, 1.0186, 1.0, 0.9934, 1.0, 0.9731, 1.0],
    },
    14: {
        "amed":      [999, 924, 856, 790, 714, 623, 571, 494, 428, 374, 285, 241, 143, 55, 0],
        "grad_scale":[1.0, 0.9922, 1.0, 0.9909, 1.0, 0.9914, 1.0, 0.9908, 1.0, 0.9904,
                      1.0, 0.9903, 1.0, 0.9904, 1.0],
        "time_scale":[1.0, 0.9835, 1.0, 1.0293, 1.0, 1.0216, 1.0, 1.0241, 1.0, 1.0021,
                      1.0, 0.9844, 1.0, 0.9714, 1.0],
    },
}


def extract_image_caption_pairs(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    image_captions = {}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]

        if image_id not in image_captions:
            image_captions[image_id] = []

        image_captions[image_id].append(caption)

    image_files = {}
    for image in data["images"]:
        image_id = image["id"]
        file_name = image["file_name"]
        image_files[image_id] = file_name

    img_paths = []
    captions = []
    for image_id, caption_list in image_captions.items():
        if image_id in image_files:
            file_name = image_files[image_id]
            for caption in caption_list[:1]:
                img_paths.append(file_name)
                captions.append(caption)

    return img_paths, captions


def read_prompts(file_path):
    img_paths, captions = extract_image_caption_pairs(file_path)
    return captions, img_paths


def get_module_kohya_state_dict(
    module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"
):
    kohya_ss_state_dict = {}
    for peft_key, weight in module.items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(8).to(dtype)

    return kohya_ss_state_dict



def load_pipeline(
    pretrained_path, lcm_lora_path, personalized_path, weight_dtype, device, type="consistencysolver"
):


    factor_net_kwargs = dict(embedding_dim=args.factor_embedding_dim, hidden_dim=args.factor_hidden_dim, num_actions=args.factor_num_actions)


    if type == "consistencysolver":
        noise_scheduler = PPOScheduler(
                beta_end = 0.012,
                beta_schedule = "scaled_linear",
                beta_start = 0.00085,
                num_train_timesteps = 1000,
                steps_offset = 1,
                trained_betas = None,
                timestep_spacing = "trailing",
                order_dim=args.order_dim,
                scaler_dim=args.scaler_dim,
                use_conv=args.use_conv,
                factor_net_kwargs = factor_net_kwargs,

            )
    elif type == "unipc":    
        noise_scheduler = UniPCMultistepScheduler.from_pretrained(
            pretrained_path,
            subfolder="scheduler",
        )
        
    elif type == "deis":    
        noise_scheduler = DEISMultistepScheduler.from_pretrained(
            pretrained_path,
            subfolder="scheduler",
        )
    elif type == "ipndm":    
        noise_scheduler = PNDMScheduler.from_pretrained(
            pretrained_path,
            subfolder="scheduler",
        ) # the diffusers implementation is exactly iPNDM
    elif type == "multistep-dpmsolver":
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
            pretrained_path,
            subfolder="scheduler",
            algorithm_type="dpmsolver",
            final_sigmas_type="sigma_min",
        )
    
    elif type == "amed":

        from diffusers_amed_plugin_dpmpp import  DPMSolverMultistepScheduler
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
            pretrained_path,
            subfolder="scheduler",
        )
    elif type == "dmdv2":
        noise_scheduler = DDIMScheduler.from_pretrained(
            pretrained_path,
            subfolder="scheduler",
            timestep_spacing = "trailing",
        )



    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_path,
        scheduler=noise_scheduler,
        revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    if type == "dmdv2":
        # w/o gan https://huggingface.co/tianweiy/DMD2/resolve/main/model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000/pytorch_model.bin
        # w/ gan https://huggingface.co/tianweiy/DMD2/resolve/main/model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume_fid8.35_checkpoint_model_041000/pytorch_model.bin
        unet_weight = torch.load(os.path.join("dmd", "withoutgan", "pytorch_model.bin"), map_location="cpu")
        pipeline.unet.load_state_dict(unet_weight)
    
    
    pipeline.set_progress_bar_config(disable=True)
    if personalized_path:
        weight = torch.load(personalized_path, map_location="cpu")
        pipeline.scheduler.factor_net.load_state_dict(weight)
        del weight

    pipeline = pipeline.to(device, dtype=weight_dtype)
    if personalized_path and type=="consistencysolver":
        pipeline.scheduler.factor_net.to(device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_vae_slicing()

    return pipeline


from multiprocessing import Pool


def process_image(args):
    img_path, transform, resolution, validation_path = args
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    path = os.path.join(validation_path, f"{img_path.split('/')[-1].split('.')[0]}.png")
    img.save(path)


def prepare_validation_set(validation_path, img_paths, resolution):
    if isinstance(resolution, int):
        resolution = [resolution, resolution]

    print("## Prepare validation dataset")
    transform = transforms.Compose(
        [
            transforms.Resize(
                resolution[0], interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(resolution),
        ]
    )

    args_list = [
        (img_path, transform, resolution, validation_path) for img_path in img_paths
    ]

    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, args_list), total=len(args_list)))


def generate_batch_images(
    prompts,
    batch_size,
    resolution,
    pipeline,
    cfg,
    num_inference_steps,
    eta,
    device,
    device_id,
    weight_dtype,
    seed,
    generation_path,
    type = "consistencysolver"
):

    total_batches = len(prompts) // batch_size + (
        1 if len(prompts) % batch_size != 0 else 0
    )
    for batch_idx in tqdm(range(total_batches)):
        batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        generator = torch.Generator(device=device).manual_seed(
            seed + batch_idx
        )  # Ensure different seeds for different batches

        # 4 step 
        # AMED schedule:     [999, 694, 500, 110, 0]
        # Gradient scales:   [1.0, 0.991, 1.0, 0.9912, 1.0]
        # Time scales:       [1.0, 1.0333, 1.0, 0.9861, 1.0]

        # 6 step
        # AMED schedule:     [999, 758, 666, 495, 333, 107, 0]
        # Gradient scales:   [1.0, 0.9924, 1.0, 0.9916, 1.0, 0.9906, 1.0]
        # Time scales:       [1.0, 1.052, 1.0, 0.9998, 1.0, 0.9781, 1.0]
            
        # # 8 step
        # AMED schedule:     [999, 831, 749, 623, 500, 394, 250, 88, 0]
        # Gradient scales:   [1.0, 0.9976, 1.0, 0.991, 1.0, 0.9907, 1.0, 0.9905, 1.0]
        # Time scales:       [1.0, 1.0257, 1.0, 0.9989, 1.0, 1.0022, 1.0, 0.9747, 1.0]
            
        # 10 step
        # [999, 885, 799, 705, 599, 492, 400, 329, 200, 73, 0]
        # [1.0, 0.9974, 1.0, 0.9904, 1.0, 0.991, 1.0, 0.9905, 1.0, 0.9904, 1.0]
        # [1.0, 0.9872, 1.0, 1.0152, 1.0, 1.0186, 1.0, 0.9934, 1.0, 0.9731, 1.0]
        
        # 14 step
        # AMED schedule:     [999, 924, 856, 790, 714, 623, 571, 494, 428, 374, 285, 241, 143, 55, 0]
        # Gradient scales:   [1.0, 0.9922, 1.0, 0.9909, 1.0, 0.9914, 1.0, 0.9908, 1.0, 0.9904, 1.0, 0.9903, 1.0, 0.9904, 1.0]
        # Time scales:       [1.0, 0.9835, 1.0, 1.0293, 1.0, 1.0216, 1.0, 1.0241, 1.0, 1.0021, 1.0, 0.9844, 1.0, 0.9714, 1.0]
        
        # sched = SCHEDULES[num_inference_steps]

        if type == "amed":
            sched = SCHEDULES[num_inference_steps]
            timesteps_lst   = sched["amed"]                # AMED schedule
            scale_dirs_list = sched["grad_scale"]          # Gradient scales
            scale_times_list = sched["time_scale"]         # Time scales
            pipeline.scheduler.scale_dirs  = scale_dirs_list
            pipeline.scheduler.scale_times = scale_times_list

            with torch.autocast("cuda", weight_dtype):
                outputs = pipeline(
                    prompt=batch_prompts,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    guidance_scale=cfg,
                    height=resolution[0],
                    width=resolution[1],
                    timesteps=timesteps_lst,          # <-- 传入自定义的 AMED 步数
                )
            images = outputs.images
        else:
            with torch.autocast("cuda", weight_dtype):
                outputs = pipeline(
                    prompt=batch_prompts,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    guidance_scale=cfg,
                    height=resolution[0],
                    width=resolution[1],
                )
            images = outputs.images
        for img_idx, (img, prompt) in enumerate(zip(images, batch_prompts)):
            img_path = os.path.join(
                generation_path,
                f"{device_id}_{batch_idx * batch_size + img_idx:08d}.png",
            )
            img.save(img_path)
            text_path = os.path.join(
                generation_path,
                f"{device_id}_{batch_idx * batch_size + img_idx:08d}.txt",
            )
            with open(text_path, "w") as f:
                f.write(prompt)


def generate_imgs(
    generation_path,
    prompts,
    resolution,
    pipeline,
    cfg,
    num_inference_steps,
    eta,
    device_id,
    weight_dtype,
    seed,
):

    torch.cuda.set_device(f"cuda:{device_id%num_device}")
    device = torch.device(f"cuda:{device_id%num_device}")

    num_prompts_per_device = len(prompts) // num_processes
    start_idx = device_id * num_prompts_per_device
    end_idx = (
        start_idx + num_prompts_per_device
        if device_id != (num_processes - 1)
        else len(prompts)
    )

    device_prompts = prompts[start_idx:end_idx]

    print(f"Device {device} generating for prompts {start_idx} to {end_idx-1}")

    print("## Prepare generation dataset")
    if isinstance(resolution, int):
        resolution = [resolution, resolution]

    batch_size = 32
    generate_batch_images(
        device_prompts,
        batch_size,
        resolution,
        pipeline,
        cfg,
        num_inference_steps,
        eta,
        device,
        device_id,
        weight_dtype,
        seed,
        generation_path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_path", default="coco_5k")
    parser.add_argument("--generation_path", default="train_coco")
    parser.add_argument(
        "--pretrained_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5"
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cfg", default=1, type=float)
    parser.add_argument("--num_inference_steps", default=4, type=int)
    parser.add_argument("--eta", default=1, type=float)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--personalized_path", default="")
    parser.add_argument("--lcm_lora_path", default="")
    parser.add_argument("--use_conv", action="store_true")
    parser.add_argument("--type", default="consistencysolver", choices=["consistencysolver", "unipc", "deis", "ipndm", "multistep-dpmsolver", "amed", "dmdv2"])


    parser.add_argument("--order_dim", type=int, default=4)
    parser.add_argument("--scaler_dim", type=int, default=2)
    parser.add_argument("--factor_embedding_dim", type=int, default=64)
    parser.add_argument("--factor_hidden_dim", type=int, default=256)
    parser.add_argument("--factor_num_actions", type=int, default=81)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # must put this line here for multiprocessing to work properly
    mp.set_start_method('spawn')

    os.makedirs(args.generation_path, exist_ok=True)
    os.makedirs(args.validation_path, exist_ok=True)

    prompts, img_paths = read_prompts("coco/annotations/captions_val2017.json")

    prompts = [prompt.strip() for prompt in prompts]

    if "subset" in args.generation_path:
        prompts = prompts[:500]

    img_paths = [os.path.join("coco/val2017", img_path) for img_path in img_paths]

    pipelines = []
    for i in range(num_processes):
        pipelines.append(
            load_pipeline(
                args.pretrained_path,
                args.lcm_lora_path,
                args.personalized_path,
                torch.float16,
                f"cuda:{i%num_device}",
                args.type, 
            )
        )
    
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                generate_imgs,
                args.generation_path,
                prompts,
                args.resolution,
                pipelines[device_id],
                args.cfg,
                args.num_inference_steps,
                args.eta,
                device_id,
                torch.float16,
                args.seed,
            )
            for device_id in range(num_processes)
        ]

        for future in as_completed(futures):
            print(f"Task completed: {future.result()}")
            
            
