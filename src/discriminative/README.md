# Discriminative Models

## Supervised baseline and knowledge-enhanced models
Comment the remaining before running one

```bash
bash src/discriminative/fine_tune.sh [device] [task]
```

where `task` is in {idiom, simile}.

## Combination

Find the best `alpha` weight to combine the outputs of the context and literal models using the dev set, and evaluate on the test set.

TBD: change the paths (e.g. change checkpoint-[num] to best)

```
python -m src.discriminative.combine_context_and_literal \
        --context_model_checkpoint [context_model_checkpoint] \
        --literal_model_checkpoint [literal_model_checkpoint] \
        --context_data_dir data/idioms_inferences_context/ \
        --literal_data_dir data/idioms_inferences_literal/ \        
        --device [device]
```

where `device` is a GPU device number or -1 for CPU.

To predict the dev/test set:

```
python -m src.discriminative.predict --data_dir [data_dir] \
                                     --model_dir [model_dir] \
                                     --out_prediction_dir [model_dir] \
                                     --task_name [idiom/idiom_inferences/simile/simile_inference]
```

## Zero-shot baseline 

### UnifiedQA

```
python -m src.discriminative.uniqa \
        --eval_file [test_file] \
        --device [device]
```

where `device` is a GPU device number or -1 for CPU.

### GPT-2

       Use https://github.com/peterwestuw/surface-form-competition and replace ROC stories with this data

### GPT-3

        Use https://github.com/peterwestuw/surface-form-competition and replace ROC stories with this data

## Few-shot baselines

### GPT-3

```
python -m src.discriminative.gpt3_few_shot \
        --openai_api_key [api_key] \
        --train_file data/idioms/train.jsonl \
        --eval_file data/idioms/[set].jsonl \
        --out_prediction_file output/discriminative/gpt3_fewshot_[set]_predictions_p0.9.jsonl;
```

where `set` is in {dev, test}.

### PET

                cd src/discriminative/pet
                
                CUDA_VISIBLE_DEVICES=6 python cli.py --method pet --pattern_ids 2 \
                --sc_max_seq_length 512 --pet_max_seq_length 512 --data_dir src/discriminative/pet/petdata/idioms \
                --model_type albert --model_name_or_path albert-xxlarge-v2 --task_name multirc\
                --output_dir output/simile/discriminative/petmodelsfewshotsimile/output/ --do_eval --do_train

