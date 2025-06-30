#!/bin/bash

# Step 1: Prepare data
echo "Preparing O2ARC data..."
python o2arc_data_preparation.py

# Step 2: Run SFT experiment with default settings
echo "Running SFT experiment..."
python o2arc_experiment.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --data_path "sft_data_74dd1130.jsonl" \
    --output_dir "./o2arc-llama3-sft" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --use_4bit true \
    --use_lora true \
    --lora_r 16 \
    --lora_alpha 32 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100

# Optional: Run with custom prompt template
# python o2arc_experiment.py \
#     --prompt_template "custom" \
#     --data_path "sft_data_74dd1130.jsonl"
