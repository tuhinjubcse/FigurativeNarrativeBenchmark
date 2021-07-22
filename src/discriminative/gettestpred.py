import tqdm
import json
import torch
import logging
import argparse
import numpy as np
import torch.nn.functional as F
import os

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from transformers import AutoModelForMultipleChoice, AutoTokenizer
from src.discriminative.common import simple_accuracy, Split
from src.discriminative.utils_multiple_choice import MultipleChoiceDataset
# from src.discriminative.utils_multiple_choice_inferences import MultipleChoiceDataset


os.environ["CUDA_VISIBLE_DEVICES"]='1'

def main():
    
    baseline_roberta_datadir = "/home/tuhinc/FigurativeNarrativeBenchmark/data/idioms"
    baseline_roberta_best_checkpoint = "/home/tuhinc/FigurativeNarrativeBenchmark/output/discriminative/roberta-large/checkpoint-4010/"
    baseline_roberta_max_seq_len = 370

    context_roberta_datadir = "/home/tuhinc/FigurativeNarrativeBenchmark/data/idioms"
    context_roberta_best_checkpoint = "/home/tuhinc/FigurativeNarrativeBenchmark/output/discriminative/roberta-large/checkpoint-2406/"
    context_roberta_max_seq_len = 390

    literal_roberta_datadir = "/home/tuhinc/FigurativeNarrativeBenchmark/data/idioms"
    literal_roberta_best_checkpoint = "/home/tuhinc/FigurativeNarrativeBenchmark/output/discriminative/roberta-large/checkpoint-2406/"
    literal_roberta_max_seq_len = 390

    datadir = baseline_roberta_datadir
    best_checkpoint = baseline_roberta_best_checkpoint
    max_seq_len = baseline_roberta_max_seq_len

    tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
    model = AutoModelForMultipleChoice.from_pretrained(best_checkpoint)
    model = model.cuda()
    model = model.eval()
    
    # Load the datasets
    print("Loading the context datasets")


    dev = MultipleChoiceDataset(
        data_dir=datadir, tokenizer=tokenizer, task="idiom",
        mode=Split.dev, max_seq_length=max_seq_len)

    test = MultipleChoiceDataset(
        data_dir=datadir, tokenizer=tokenizer, task="idiom",
        mode=Split.test, max_seq_length=max_seq_len)
   
    print("Predicting the dev set")
    dev_logits = get_logits(model, dev)
    dev_gold = [int(json.loads(line)["correctanswer"][len("option"):]) - 1 for line in open(datadir+"/dev.jsonl")]
    dev_predictions = [np.argmax(ctx_p) for ctx_p in dev_logits]
    print(dev_predictions)
    print("=============")
    print(dev_gold)
    accuracy = simple_accuracy(dev_predictions, dev_gold)
    print("Dev accuracy ", accuracy)

    print("Predicting the test set")
    test_logits = get_logits(model, test)
    test_gold = [int(json.loads(line)["correctanswer"][len("option"):]) - 1 for line in open(datadir+"/test.jsonl")]
    test_predictions = [np.argmax(ctx_p) for ctx_p in test_logits]
    accuracy = simple_accuracy(test_predictions, test_gold)
    print("Test accuracy: ",accuracy)
    

def get_logits(model, examples):
    """
    Forward the examples in the model and return the logits
    :return: a list of logits
    """
    with torch.no_grad():
        logits1 = []
        for i in tqdm.tqdm(range(len(examples.features))):
            ip = torch.Tensor([examples.features[i].input_ids]).long().cuda()
            att = torch.Tensor([examples.features[i].attention_mask]).long().cuda()
            op = model(input_ids=ip,attention_mask=att)
            log = op[0]
            logits1.append(log.cpu().numpy())
    
    return logits1


if __name__ == "__main__":
    main()
