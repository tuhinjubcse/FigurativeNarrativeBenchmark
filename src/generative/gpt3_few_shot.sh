#!/usr/bin/env bash

api_key=$1
set=$2

python -m src.generative.gpt3_few_shot \
        --openai_api_key ${api_key} \
        --train_file data/idioms/train.jsonl \
        --eval_file data/idioms/${set}.jsonl \
        --out_prediction_file output/generative/gpt3_fewshot_${set}_predictions_p0.9.jsonl;