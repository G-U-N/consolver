import logging
import math
import os
import random
from pathlib import Path
import random
import accelerate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from scheduler_ppo import PPOScheduler
from denoise_ppo import denoise_diffusion

from data_processing import CustomImageDataset, repeat_random_sample
from utils import decode_latents, is_dict_like, concatenate_samples
from reward_model import calculate_reward
from reward_model import load_reward_model
from config import parse_args 

MAX_SEQ_LENGTH = 77

if is_wandb_available():
    import wandb

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
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.ppo_type == "discrete":
        factor_net_kwargs = dict(embedding_dim=args.factor_embedding_dim, hidden_dim=args.factor_hidden_dim, num_actions=args.factor_num_actions)
    else:
        factor_net_kwargs = dict(embedding_dim=args.factor_embedding_dim, hidden_dim=args.factor_hidden_dim)

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
            ppo_type=args.ppo_type,
            factor_net_kwargs = factor_net_kwargs,

        )

    factor_net = noise_scheduler.factor_net

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.teacher_revision,
        use_fast=False,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder",
        revision=args.teacher_revision,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.teacher_revision,
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
    )
    target_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="unet",
        revision=args.teacher_revision,
    )

    unet.train()

    reward_model, reward_model_processor = load_reward_model(args.reward_type, accelerator.device)

    low_precision_error_string = (
            " Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training, copy of the weights should still be float32."
        )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )


    unet.requires_grad_(False)
    target_unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    factor_net.requires_grad_(True)
    if  not (args.reward_type == "llava" or  args.reward_type=="image_psnr"):
        reward_model.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    target_unet.to(accelerator.device, dtype=weight_dtype)
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

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            target_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: pip install bitsandbytes."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        factor_net.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = CustomImageDataset(
        "gen_pretrain/samples/laion_2b_en/2k", args.resolution
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

                # image, text, noise, tch_traj = repeat_random_sample(batch)

                for _ in range(num_samples_per_batch):
                    # Generate a new sample for the current batch
                    image, text, noise, tch_traj = repeat_random_sample(batch)

                    # Move tensors to device
                    image = image.to(accelerator.device, non_blocking=True)
                    tch_traj = tch_traj.to(accelerator.device, non_blocking=True)
                    noise = noise.to(accelerator.device, non_blocking=True)

                    # Randomly select number of inference steps
                    num_inference = random.choice(list(range(2, 16)))

                    # Extract target from trajectory
                    # target = tch_traj[:, -1]
                    target = tch_traj

                    # Collect trajectories using denoise_diffusion
                    with torch.no_grad():
                        with accelerator.autocast():
                            model_pred, conds, probs, actions, masks, text_prompts = denoise_diffusion(
                                text_encoder,
                                noise_scheduler,
                                unet,
                                noise,
                                text,
                                tokenizer,
                                cfg=float(args.cfg),
                                num_inference_steps=num_inference,
                                gradient_checkpointing=args.gradient_checkpointing,
                            )

                            # Decode predictions and targets
                            model_pred_decoded = decode_latents(vae, model_pred, batch_size=8)
                            target_decoded = decode_latents(vae, target, batch_size=8)

                            # Calculate rewards
                            rewards = calculate_reward(
                                args.reward_type, reward_model, reward_model_processor,
                                model_pred_decoded, target_decoded, accelerator.device
                            )
                            # Compute advantages
                            advantages = (rewards - rewards.mean())  / (rewards.std() + 1e-8) * 10 
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
                    
                    ratio = (log_probs - old_log_probs).exp()
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


                if global_step % 10 == 0:
                    unwrapped_model = accelerator.unwrap_model(factor_net)
                    param_sum = sum(p.sum().item() for p in unwrapped_model.parameters())
                    print(f"Step {global_step}, Process {accelerator.process_index}, Param sum: {param_sum}")

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