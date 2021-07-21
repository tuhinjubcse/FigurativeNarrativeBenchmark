#!/usr/bin/env bash

api_key=$1
set=$2

python -m src.generative.gpt3_zero_shot \
        --openai_api_key ${api_key} \
        --eval_file data/idioms/${set}.jsonl \
        --out_prediction_file output/generative/idioms/gpt3_zeroshot_${set}_predictions_p0.9.jsonl;
