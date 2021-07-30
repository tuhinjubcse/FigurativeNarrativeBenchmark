#!/usr/bin/env bash

api_key=$1
set=$2
task=$3

python -m src.generative.gpt3_few_shot \
        --openai_api_key ${api_key} \
        --train_file data/${task}/train.jsonl \
        --eval_file data/${task}/${set}.jsonl \
        --out_prediction_file output/${task}/generative/gpt3_fewshot_${set}_predictions_p0.9.jsonl;