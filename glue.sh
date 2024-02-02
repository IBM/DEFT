#!/bin/bash

# DEFT with Adapter

if [[ $1 == 'deft' ]]; then
    python adapter.py  --reduction_factor 16 --eps 1e-07 --seed $2 --batch_size 64 --lr 3e-4 --weight_l 1.0 --model_name_or_path "roberta-large" \
        --pad_to_max_length --max_seq_length 128  --task $3 --epochs 10  \
        --sparse_obj
elif [[ $1 == 'peft' ]]; then
    python adapter.py --reduction_factor 16 --eps 1e-07 --seed $2 --batch_size 64 --lr 3e-4 --weight_l 1.0 --model_name_or_path "roberta-large" \
        --pad_to_max_length --max_seq_length 128  --task $3 --epochs 10
else
    echo 'unknown argment 1'
fi
