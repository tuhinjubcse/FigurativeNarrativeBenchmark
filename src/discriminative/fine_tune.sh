#!/usr/bin/env bash
#use single gpu for literal

device=$1
lm="roberta-large"
declare -a knowledge_types1=(literal)
declare -a knowledge_types2=(context)

#Trained on 2 gpus

CUDA_VISIBLE_DEVICES=${device} python -m src.discriminative.run_multiple_choice \
    --data_dir data/idioms/ \
    --output_dir output/discriminative/${lm} \
    --task_name idiom \
    --model_name_or_path ${lm} \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --max_seq_length 370  \
    --overwrite_cache \
    --num_train_epochs 10 \
    --logging_steps 401 \
    --save_steps 401 \
    --eval_steps 401 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 ;


# Trained on 1 GPU the remaining two
# This is a strict requirement because of variable number of inferences


for knowledge_type in "${knowledge_types1[@]}"
do
    CUDA_VISIBLE_DEVICES=${device} python -m src.discriminative.run_multiple_choice \
        --data_dir data/idioms_inferences_${knowledge_type}/ \
        --output_dir output/discriminative/${lm}_inferences_${knowledge_type} \
        --task_name idiom_inferences \
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


# #Trained on 1 GPU

for knowledge_type in "${knowledge_types2[@]}"
do
    CUDA_VISIBLE_DEVICES=${device} python -m src.discriminative.run_multiple_choice \
        --data_dir data/idioms_inferences_${knowledge_type}/ \
        --output_dir output/discriminative/${lm}_inferences_${knowledge_type}_dropout \
        --task_name idiom_inferences \
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
