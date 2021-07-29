# Generative Models

## Training 

Fine-tuning a LM to predict the continuation of the narratives. 
To train the models, run:

```bash
bash src/generative/fine_tune.sh [device] [task]
```

where `device` is a GPU device number or -1 for CPU and `task` is in {idiom, simile}.

## Prediction 

To generate texts using the trained models, run:

```bash
bash src/generative/generate_texts.sh [device] [set] [task]
```

where `set` is in {dev, test}, `device` is a GPU device number or -1 for CPU, and `task` is in {idiom, simile}.

### GPT-3

To get zero-shot results for GPT-3 (the other models are included in `generate_texts.sh`), run:

```
```

For few-shot results, run:

```bash
bash src/generative/gpt3_few_shot.sh [api_key] [set]
```

where `set` is in {dev, test} and `api_key` is your API Key for OpenAI GPT-3.

### Evaluation

To compute automatic evaluation metric, run:

```
python -m src.generative.automatic_metrics --set [set] --device [device] --task [task]
```

where `set` is in {dev, test}, `device` is a GPU device number or -1 for CPU, and `task` is in {idiom, simile}.