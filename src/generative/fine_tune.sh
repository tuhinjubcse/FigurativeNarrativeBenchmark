#!/usr/bin/env bash

device=$1
declare -a enc_dec_models=(facebook/bart-large t5-large)
declare -a lm_models=(gpt2-xl)
declare -a knowledge_types=(context literal)


for lm in "${enc_dec_models[@]}"
do
    lr=5e-6 && [[ $lm == *"t5"* ]] && lr=1e-3

    python -m src.generative.encoder_decoder \
        --train_file data/idioms/train.jsonl \
        --eval_data_file data/idioms/dev.jsonl \
        --out_dir output/generative/${lm/facebook\//} \
        --model_name_or_path ${lm} \
        --device ${device} \
        --do_train \
        --do_eval \
        --eval_during_train \
        --save_total_limit 1 \
        --overwrite_cache \
        --num_train_epochs 5 \
        --logging_steps 1000 \
        --train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate ${lr} \
        --overwrite_out_dir \
        --eval_batch_size 64 ;
done

for lm in "${lm_models[@]}"
do
    python -m src.generative.fine_tune_lm \
        --train_file data/idioms/train.jsonl \
        --eval_data_file data/idioms/dev.jsonl \
        --out_dir output/generative/${lm} \
        --model_name_or_path ${lm} \
        --device ${device} \
        --do_train \
        --do_eval \
        --eval_during_train \
        --save_total_limit 1 \
        --overwrite_cache \
        --num_train_epochs 2 \
        --logging_steps 1000 \
        --train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --overwrite_out_dir \
        --eval_batch_size 16 ;
done

for knowledge_type in "${knowledge_types[@]}"
do
    python -m src.generative.fine_tune_lm_with_inferences \
            --train_file data/idioms_inferences_${knowledge_type}/train.jsonl \
            --eval_data_file data/idioms_inferences_${knowledge_type}/dev.jsonl \
            --out_dir output/generative/gpt2-large_${knowledge_type} \
            --model_name_or_path gpt2-large \
            --device ${device} \
            --do_train \
            --do_eval \
            --eval_during_train \
            --save_total_limit 1 \
            --overwrite_cache \
            --num_train_epochs 2 \
            --logging_steps 1000 \
            --train_batch_size 8 \
            --gradient_accumulation_steps 8 \
            --overwrite_out_dir \
            --eval_batch_size 16 ;
done