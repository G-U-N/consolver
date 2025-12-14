PREFIX=Kontext_2order_naction_11_1e-3_dino
MODEL_DIR="black-forest-labs/FLUX.1-Kontext-dev"
OUTPUT_DIR="outputs/$PREFIX"
PROJ_NAME="$PREFIX"
accelerate launch --main_process_port 29500  --num_processes=8 --config_file="accelerate_config.yaml" train_ppo.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_name=$PROJ_NAME \
    --mixed_precision=bf16 \
    --resolution=1024 \
    --learning_rate=1e-3 --loss_type="huber" --adam_weight_decay=1e-3 \
    --max_train_steps=1001 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=16 \
    --validation_steps=100 \
    --checkpointing_steps=100 --checkpoints_total_limit=20 \
    --train_batch_size=10 \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --order_dim=2 \
    --scaler_dim=0 \
    --ppo_epochs=4 \
    --cfg=2.5 \
    --factor_embedding_dim=1024 \
    --factor_hidden_dim=256 \
    --factor_num_actions=11 \
    --reward_type="dino" \
    --ppo_type="discrete" \
    --gradient_checkpointing 
