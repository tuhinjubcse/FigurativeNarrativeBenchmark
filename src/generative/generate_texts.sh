#!/usr/bin/env bash

device=$1
set=$2
task=$3
declare -a enc_dec_models=(facebook/bart-large t5-large)
declare -a lm_models=(gpt2-xl)
declare -a knowledge_types=(context literal)


for lm in "${lm_models[@]}"
do
    python -m src.generative.generate_texts \
            --in_file data/${task}/${set}.jsonl \
            --out_file output/${task}/generative/${lm}_${set}_predictions_k5.jsonl \
            --model_name_or_path output/${task}/generative/${lm} \
            --k 5 \
            --temperature 0.7 \
            --device ${device};

    for knowledge_type in "${knowledge_types[@]}"
    do
        python -m src.generative.generate_texts_inferences \
            --in_file data/${task}_inferences_${knowledge_type}/${set}.jsonl \
            --out_file output/${task}/generative/${lm}_${knowledge_type}_${set}_predictions_k5.jsonl \
            --model_name_or_path output/${task}/generative/${lm}_${knowledge_type} \
            --k 5 \
            --temperature 0.7 \
            --device ${device};
    done
done

for lm in "${enc_dec_models[@]}"
do
    python -m src.generative.generate_texts \
        --in_file data/${task}/${set}.jsonl \
        --out_file output/${task}/generative/${lm/facebook\//}_${set}_predictions_k5.jsonl \
        --model_name_or_path output/${task}/generative/${lm/facebook\//} \
        --k 5 \
        --temperature 0.7 \
        --is_encoder_decoder \
        --device ${device};
done

# Zero-shot

for lm in "${lm_models[@]}"
do
    python -m src.generative.generate_texts \
            --in_file data/${task}/${set}.jsonl \
            --out_file output/${task}/generative/${lm}_zeroshot_${set}_predictions_k5.jsonl \
            --model_name_or_path ${lm} \
            --k 5 \
            --temperature 0.7 \
            --is_zero_shot \
            --device ${device};
done

for lm in "${enc_dec_models[@]}"
do
    python -m src.generative.generate_texts \
        --in_file data/${task}/${set}.jsonl \
        --out_file output/${task}/generative/${lm/facebook\//}_zeroshot_${set}_predictions_k5.jsonl \
        --model_name_or_path ${lm} \
        --k 5 \
        --temperature 0.7 \
        --is_encoder_decoder \
        --is_zero_shot \
        --device ${device};
done
