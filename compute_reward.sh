python compute_reward.py \
    --dir1 baselines/1101_subset/teacher/cfg3/40step/3000 \
    --dir2 baselines/1101_subset/4order_depth/cfg3/5step/3000 \
    --num_gpus 8 \
    --reward_types dino inception image_psnr clip depth segmentation \
    --batch_size 1 \
    --output ./statistics/ours/5step-subset.json


