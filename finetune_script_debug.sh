
git clone https://github.com/huggingface/transformers.git

python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir /scratch/prj/eventnlu/share/pretrained_models/llama --model_size 7B --output_dir llama_7b_hf

git lfs install

git clone https://huggingface.co/tatsu-lab/alpaca-7b-wdiff

python weight_diff.py recover --path_raw llama_7b_hf --path_diff alpaca-7b-wdiff --path_tuned alpaca_weights

torchrun --nproc_per_node=3 train.py \
    --model_name_or_path alpaca_weights \
    --data_path data/alpaca_formatted_train_debug.json \
    --fp16 True \
    --output_dir alpaca_run \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 