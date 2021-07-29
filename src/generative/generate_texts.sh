#!/usr/bin/env bash

device=$1
set=$2
task=$3
declare -a enc_dec_models=(facebook/bart-large t5-large)
declare -a lm_models=(gpt2-xl)
declare -a knowledge_types=(context literal)


for knowledge_type in "${knowledge_types[@]}"
do
    python -m src.generative.generate_texts_inferences \
        --in_file data/${task}_inferences_${knowledge_type}/${set}.jsonl \
        --out_file output/${task}/generative/gpt2-xl_${knowledge_type}_${set}_predictions_k5.jsonl \
        --model_name_or_path output/${task}/generative/gpt2-xl_${knowledge_type} \
        --k 5 \
        --temperature 0.7 \
        --device ${device};
done
