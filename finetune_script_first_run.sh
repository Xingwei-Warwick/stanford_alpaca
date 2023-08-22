python permu_augmentation.py

torchrun --nproc_per_node=3 train.py \
    --model_name_or_path /scratch/prj/eventnlu/share/pretrained_models/alpaca_weights \
    --data_path data/NYT_des_train_alpaca_permu5.json \
    --output_dir alpaca_run \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 