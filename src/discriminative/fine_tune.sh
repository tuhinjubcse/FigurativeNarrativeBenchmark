#!/usr/bin/env bash

device=$1
lm="roberta-large"
declare -a knowledge_types=(context literal)

CUDA_VISIBLE_DEVICES=${device} python -m src.discriminative.run_multiple_choice \
    --data_dir data/idioms/ \
    --output_dir output/discriminative/${lm} \
    --task_name idiom \
    --model_name_or_path ${lm} \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --max_seq_length 390  \
    --overwrite_cache \
    --num_train_epochs 10 \
    --logging_steps 1000 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 ;


for knowledge_type in "${knowledge_types[@]}"
do
    CUDA_VISIBLE_DEVICES=${device} python -m src.discriminative.run_multiple_choice \
        --data_dir data/idioms_inferences_${knowledge_type}/ \
        --output_dir output/discriminative/${lm}_inferences_${knowledge_type} \
        --task_name idiom_inferences \
        --model_name_or_path ${lm} \
        --do_train \
        --do_eval \
        --save_total_limit 1 \
        --max_seq_length 390  \
        --overwrite_cache \
        --num_train_epochs 10 \
        --logging_steps 1000 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --learning_rate 1e-5 ;
done
