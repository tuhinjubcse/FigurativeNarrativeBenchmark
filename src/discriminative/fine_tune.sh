#!/usr/bin/env bash

device=$1
task=$2 # idiom / simile
lm="roberta-large"
declare -a knowledge_types=(literal context)

for knowledge_type in "${knowledge_types[@]}"
do
    CUDA_VISIBLE_DEVICES=${device} python -m src.discriminative.run_multiple_choice \
        --data_dir data/${task}_inferences_${knowledge_type}/ \
        --output_dir output/${task}/discriminative/${lm}_inferences_${knowledge_type} \
        --task_name ${task}_inferences \
        --model_name_or_path ${lm} \
        --do_train \
        --do_eval \
	--gradient_accumulation_steps 16 \
        --evaluation_strategy steps \
        --overwrite_cache \
        --num_train_epochs 10 \
        --logging_steps 200 \
        --save_steps 200 \
        --eval_steps 200 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --learning_rate 1e-5 ;
done
