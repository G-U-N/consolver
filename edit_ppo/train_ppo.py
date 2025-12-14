import logging
import math
import os
import random
from pathlib import Path
import random
import accelerate
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from diffusers import FluxKontextPipeline
import datetime
from accelerate.utils import broadcast


import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

from scheduler_fmppo import FMPPOScheduler
from denoise_diffusion import denoise_diffusion

from data_processing import CustomImageDataset, repeat_random_sample
from utils import decode_latents, is_dict_like, concatenate_samples, tensor_to_pil
from reward_model import calculate_reward
from reward_model import load_reward_model
from config import parse_args 

MAX_SEQ_LENGTH = 77

if is_wandb_available():
    import wandb
    
    
os.environ["TOKENIZERS_PARALLELISM"] = "false"

check_min_version("0.18.0.dev0")

logger = get_logger(__name__)

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.ppo_type == "discrete":
        factor_net_kwargs = dict(embedding_dim=args.factor_embedding_dim, hidden_dim=args.factor_hidden_dim, num_actions=args.factor_num_actions)
    else:
        factor_net_kwargs = dict(embedding_dim=args.factor_embedding_dim, hidden_dim=args.factor_hidden_dim)

    noise_scheduler = FMPPOScheduler.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", subfolder="scheduler", order_dim=2, scaler_dim=0, mu_dim=0, factor_net_kwargs=factor_net_kwargs)

    factor_net = noise_scheduler.factor_net

    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    pipe.to(accelerator.device)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    
    reward_model, reward_model_processor = load_reward_model(args.reward_type, accelerator.device)


    factor_net.requires_grad_(True)
    if  not (args.reward_type == "llava" or  args.reward_type=="image_psnr"):
        reward_model.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    if  not (args.reward_type == "llava" or  args.reward_type=="image_psnr"):
        reward_model.to(accelerator.device, dtype=weight_dtype) 
        reward_model.eval()

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                factor_net_ = accelerator.unwrap_model(factor_net)
                torch.save(factor_net_.state_dict(), os.path.join(output_dir, "model.ckpt"))
                for _, model in enumerate(models):
                    weights.pop()

        def load_model_hook(models, input_dir):
            factor_net_ = accelerator.unwrap_model(factor_net)
            factor_net_.load_state_dict(
                torch.load(os.path.join(input_dir, "model.ckpt"), map_location="cpu")
            )
            for _ in range(len(models)):
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)



    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        factor_net.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = CustomImageDataset(
        "/home/ubuntu/DiffusionPreview/edit_pretrain/data", args.resolution
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    factor_net, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        factor_net, optimizer, lr_scheduler, train_dataloader
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    samples_dir = os.path.join(args.output_dir, f"samples_{timestamp}")
    if accelerator.is_main_process:
        os.makedirs(samples_dir, exist_ok=True)
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(factor_net):
                # Number of samples per batch
                num_samples_per_batch = 1
                all_conds = []
                all_actions = []
                all_probs = []
                all_masks = []
                all_advantages = []

                for _ in range(num_samples_per_batch):
                    # Generate a new sample for the current batch
                    image, text, noise, tch_traj, ref_image = repeat_random_sample(batch)
                    # print(noise.shape)
                    # print(tch_traj.shape)

                    # Move tensors to device
                    image = image.to(accelerator.device, non_blocking=True)
                    tch_traj = tch_traj.to(accelerator.device, non_blocking=True)
                    noise = noise.to(accelerator.device, non_blocking=True)

                    # Randomly select number of inference steps
                    # num_inference = random.choice(list(range(4)))
                    # num_inference = random.choice([4, 6, 8, 10])
                    # print(num_inference)


                    if accelerator.is_main_process:
                        # num_inference = random.choice([])
                        num_inference = random.choice(list(range(2, 6)))
                    else:
                        num_inference = 0

                    num_inference_tensor = torch.tensor(num_inference, dtype=torch.int32).to(accelerator.device)
                    num_inference_tensor = broadcast(num_inference_tensor)
                    num_inference = num_inference_tensor.item()
                    
                    target = tch_traj

                    # Collect trajectories using denoise_diffusion
                    with torch.no_grad():
                        with accelerator.autocast():
                            latents_base, pred_image_base = denoise_diffusion(
                                pipe.scheduler,
                                pipe,
                                noise[0:1],
                                text[0:1],
                                ref_image[0:1],
                                cfg=float(args.cfg),
                                num_inference_steps=num_inference,
                                gradient_checkpointing=args.gradient_checkpointing,
                                use_naive_scheduler = True
                            )
                            
                            latents, pred_image, conds, probs, actions, masks = denoise_diffusion(
                                noise_scheduler,
                                pipe,
                                noise,
                                text,
                                ref_image,
                                cfg=float(args.cfg),
                                num_inference_steps=num_inference,
                                gradient_checkpointing=args.gradient_checkpointing,
                            )
                            
                        pred_img = decode_latents(pipe, latents)
                        pred_img_base = decode_latents(pipe, latents_base)
                        target_img = decode_latents(pipe, target)
                        rewards = calculate_reward(
                            args.reward_type, reward_model, reward_model_processor,
                            pred_img.float(), target_img.float(), accelerator.device
                        )
                        
                        rewards_base = calculate_reward(
                            args.reward_type, reward_model, reward_model_processor,
                            pred_img_base.float(), target_img[0:1].float(), accelerator.device
                        )[0].detach().cpu().item()
                        # Compute advantages
                        advantages = (rewards - rewards.mean().clip(rewards_base, 100.0))  / (rewards.std() + 1e-8) # more varianced reward model leads to better training?
                        advantages_origin = advantages.detach().clone()
                        advantages = advantages.repeat(1, (num_inference - 1)).reshape(
                            advantages.shape[0] * (num_inference - 1), -1
                        )
                        # Reshape conds (handles both dict and tensor cases)
                        conds = (
                            {k: v.reshape(v.shape[0] * (num_inference - 1), *v.shape[2:]) for k, v in conds.items()}
                            if is_dict_like(conds)
                            else conds.reshape(conds.shape[0] * (num_inference - 1), *conds.shape[2:])
                        )
                        actions = actions.reshape(actions.shape[0] * (num_inference - 1), -1)
                        probs = probs.reshape(probs.shape[0] * (num_inference - 1), -1)
                        masks = masks.reshape(masks.shape[0] * (num_inference - 1), -1)
                        
                        advantages = advantages * masks # mask out the non-updated steps
                        # Collect data from this sample
                        all_conds.append(conds)
                        all_actions.append(actions)
                        all_probs.append(probs)
                        all_masks.append(masks)
                        all_advantages.append(advantages)

                # Concatenate all samples
                conds = concatenate_samples(all_conds, is_dict=is_dict_like(all_conds[0]))
                actions = concatenate_samples(all_actions)
                probs = concatenate_samples(all_probs)
                masks = concatenate_samples(all_masks)
                advantages = concatenate_samples(all_advantages)
                # PPO optimization using factor_net directly
                for _ in range(args.ppo_epochs):
                    # Get current policy distribution from factor_net
                    curr_probs, entropy = factor_net(conds, actions)

                    log_probs = (curr_probs + 1e-9).log().sum(dim=1).unsqueeze(1) # joint distribution
                    old_log_probs = (probs + 1e-9).log().sum(dim=1).unsqueeze(1) # joint distribution 
                    
                    # print(conds)
                    ratio = (log_probs - old_log_probs).exp()
                    # print(ratio)
                    clipped_ratio = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)

                    policy_loss = -torch.min(
                        advantages * ratio,
                        advantages * clipped_ratio
                    ) # * masks
                    policy_loss = policy_loss.mean()

                    # Add entropy bonus buggy implementation
                    # entropy = -curr_probs * log_probs * masks
                    entropy_loss = -args.entropy_coef * entropy.mean()

                    # Total loss
                    loss = policy_loss  + entropy_loss

                    # Optimization step
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        norm = accelerator.clip_grad_norm_(
                            factor_net.parameters(),
                            args.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()



            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")


                if (global_step - 1) % 10 == 0:
                    unwrapped_model = accelerator.unwrap_model(factor_net)
                    param_sum = sum(p.sum().item() for p in unwrapped_model.parameters())
                    print(f"Step {global_step}, Process {accelerator.process_index}, Param sum: {param_sum}")

                if (global_step -  1) % 1 == 0:
                    # if accelerator.is_main_process:
                    # Assuming pred_img is a torch.Tensor of shape (bs, 3, h, w) in [0, 1]
                    for i in range(min(10, len(pred_image))):
                        pred_image[i].save(os.path.join(samples_dir, f"{global_step}_{accelerator.process_index}_{i}_{advantages_origin[i].item(): .3f}_{num_inference}step.jpg"))
                    tensor_to_pil(target_img[0].detach().float()).save(os.path.join(samples_dir, f"target_{global_step}_{accelerator.process_index}_{i}.jpg"))
                    with open(os.path.join(samples_dir, f"conds_{global_step}_{accelerator.process_index}_{i}.txt"), "w") as f:
                        f.write(text[0])
                            

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "norm": norm, "reward": rewards.mean().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)



            if global_step >= args.max_train_steps:
                break

    if args.output_dir is not None and accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(factor_net)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

if __name__ == "__main__":
    args = parse_args()
    main(args)