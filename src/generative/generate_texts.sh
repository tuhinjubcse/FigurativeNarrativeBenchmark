#!/usr/bin/env bash

device=$1
set=$2
declare -a enc_dec_models=(facebook/bart-large t5-large)
declare -a lm_models=(gpt2-xl)
declare -a knowledge_types=(context literal)

for lm in "${enc_dec_models[@]}"
do
    python -m src.generative.generate_texts \
        --in_file data/idioms/${set}.jsonl \
        --out_file output/generative/${lm/facebook\//}_${set}_predictions_k5.jsonl \
        --model_name_or_path output/generative/idioms/${lm/facebook\//} \
        --is_encoder_decoder \
        --k 5 \
        --temperature 0.7 \
        --device ${device};
 done

for lm in "${lm_models[@]}"
do
    python -m src.generative.generate_texts \
        --in_file data/idioms/${set}.jsonl \
        --out_file output/generative/${lm}_${set}_predictions_k5.jsonl \
        --model_name_or_path output/generative/idioms/${lm} \
        --k 5 \
        --temperature 0.7 \
        --device ${device};
done

for lm in "${lm_models[@]}"
do
    mkdir output/generative/idioms/${lm}_zeroshot/;
    python -m src.generative.generate_texts_zs \
        --in_file data/idioms/${set}.jsonl \
        --out_file output/generative_${lm}_zeroshot/${set}_predictions_k5.jsonl \
        --model_name_or_path ${lm} \
        --k 5 \
        --temperature 0.7 \
        --device ${device};
done

for knowledge_type in "${knowledge_types[@]}"
do
    python -m src.generative.generate_texts_inferences \
        --in_file data/idioms_inferences_${knowledge_type}/${set}.jsonl \
        --out_file output/generative/gpt2-medium_${knowledge_type}_${set}_predictions_k1.jsonl \
        --model_name_or_path output/generative/idioms/gpt2-medium_${knowledge_type} \
        --k 5 \
        --temperature 0.7 \
        --device ${device};
done