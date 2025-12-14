PREFIX=depth_4order_cfg3_prod_num_action_11_1e-4
MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR="outputs/$PREFIX"
PROJ_NAME="$PREFIX"
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29500  --num_processes=1 --config_file="accelerate_config.yaml" train_ppo.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_name=$PROJ_NAME \
    --enable_xformers_memory_efficient_attention \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=1e-4 --loss_type="huber" --adam_weight_decay=1e-3 \
    --max_train_steps=3001 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=16 \
    --validation_steps=100 \
    --checkpointing_steps=100 --checkpoints_total_limit=20 \
    --train_batch_size=80 \
    --gradient_accumulation_steps=1 \
    --cfg=3 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --order_dim=4 \
    --scaler_dim=0 \
    --ppo_epochs=1 \
    --factor_embedding_dim=1024 \
    --factor_hidden_dim=256 \
    --factor_num_actions=11 \
    --reward_type="depth" \
    --ppo_type="discrete" \
    --gradient_checkpointing 
