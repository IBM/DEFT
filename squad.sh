#!/bin/bash

seed=$2
echo "seed: $seed"

if [[ $1 == 'deft' ]]; then
    accelerate launch  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT  SQuAD/deft_t5.py \
    --model_name_or_path google/flan-t5-xl \
    --seed $seed \
    --cache_dir "/scratch/IAS/hf/" \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-5 \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /DEFT_Flan_T5_XL_$seed \
    --train_adapter \
    --adapter_config pfeiffer \
    --eps 1e-07 \
    --red_factor 16 \
    --sparse_obj \

elif [[ $1 == 'peft' ]]; then
    accelerate launch  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT  SQuAD/peft_t5.py \
    --model_name_or_path google/flan-t5-xl \
    --seed $seed \
    --cache_dir "/scratch/IAS/hf/" \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-5 \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /PEFT_Flan_T5_XL_$seed \
    --train_adapter \
    --adapter_config pfeiffer \
    --red_factor 16 \
else
    echo 'unknown argment 1'
fi



