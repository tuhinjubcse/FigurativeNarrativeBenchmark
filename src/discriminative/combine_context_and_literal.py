import tqdm
import json
import torch
import logging
import argparse
import numpy as np
import torch.nn.functional as F

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from transformers import AutoModelForMultipleChoice, AutoTokenizer

from src.discriminative.common import simple_accuracy, Split
from src.discriminative.utils_multiple_choice_inferences import MultipleChoiceDataset, RobertaForMultipleChoiceWithInferences


def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--context_model_checkpoint",
        default=None,
        type=str,
        required=True,
        help="The torch checkpoint to the context model.",
    )
    parser.add_argument(
        "--literal_model_checkpoint",
        default=None,
        type=str,
        required=True,
        help="The torch checkpoint to the literal model.",
    )
    parser.add_argument(
        "--context_data_dir",
        default=None,
        type=str,
        required=True,
        help="The data directory containing the dev and test set json files for the context model.",
    )
    parser.add_argument(
        "--literal_data_dir",
        default=None,
        type=str,
        required=True,
        help="The data directory containing the dev and test set json files for the literal model.",
    )
    
    # Optional arguments
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--max_seq_len", default=390, type=int, required=False, help="maximum number of input tokens."
    )

    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    
    # Compute each model separately because of memory issues


   
    model_class = RobertaForMultipleChoiceWithInferences

    
    # Context model:
    
    # Load the model
    logger.info("Loading the context model")
    context_tokenizer = AutoTokenizer.from_pretrained(args.context_model_checkpoint)
    context_model =  model_class.from_pretrained(args.context_model_checkpoint)
    context_model = context_model.to(device)
    context_model = context_model.eval()
    
    # Load the datasets
    logger.info("Loading the context datasets")
    context_dev = MultipleChoiceDataset(
        data_dir=args.context_data_dir, tokenizer=context_tokenizer, task="idiom_inferences",
        mode=Split.dev, max_seq_length=args.max_seq_len)
    context_test = MultipleChoiceDataset(
        data_dir=args.context_data_dir, tokenizer=context_tokenizer, task="idiom_inferences",
        mode=Split.test, max_seq_length=args.max_seq_len)
   
    res = open("results.txt","a")
    # Solve the dev set
    logger.info("Predicting the dev set with the context model")
    context_dev_logits, context_dev_logits1 = get_logits(context_model, context_dev, device)
    
    dev_gold = [int(json.loads(line)["correctanswer"][len("option"):]) - 1 for line in open(f"{args.context_data_dir}/dev.jsonl")]
    dev_predictions = [np.argmax(ctx_p) for ctx_p in context_dev_logits1]
    
    accuracy = simple_accuracy(dev_predictions, dev_gold)
    logger.info(f"Context Dev accuracy: {accuracy*100:.1f}")
    res.write("Context Dev accuracy: "+str(accuracy)+'\n\n')

    logger.info("Predicting the test set with the context model")
    context_test_logits, context_test_logits1 = get_logits(context_model, context_test, device)
    
    context_model.to(torch.device("cpu"))
    torch.cuda.empty_cache()
    
    test_gold = [int(json.loads(line)["correctanswer"][len("option"):]) - 1 for line in open(f"{args.context_data_dir}/test.jsonl")]
    test_predictions = [np.argmax(ctx_p) for ctx_p in context_test_logits1]
    accuracy = simple_accuracy(test_predictions, test_gold)
    logger.info(f"Context Test accuracy: {accuracy*100:.1f}")
    res.write("Context Test accuracy: "+str(accuracy)+'\n\n')
    
    # Literal model:
    
    # Load the model
    logger.info("Loading the literal model")
    literal_tokenizer = AutoTokenizer.from_pretrained(args.literal_model_checkpoint)
    literal_model = model_class.from_pretrained(args.literal_model_checkpoint)
    literal_model = literal_model.to(device)
    literal_model = literal_model.eval()
    
    # Load the datasets
    logger.info("Loading the literal datasets")
    literal_dev = MultipleChoiceDataset(
        data_dir=args.literal_data_dir, tokenizer=literal_tokenizer, task="idiom_inferences",
        mode=Split.dev, max_seq_length=args.max_seq_len)
    literal_test = MultipleChoiceDataset(
        data_dir=args.literal_data_dir, tokenizer=literal_tokenizer, task="idiom_inferences",
        mode=Split.test, max_seq_length=args.max_seq_len)
    
    # Solve the dev set
    logger.info("Predicting the dev set with the literal model")
    literal_dev_logits, literal_dev_logits1 = get_logits(literal_model, literal_dev, device)
    logger.info("Predicting the test set with the literal model")
    literal_test_logits, literal_test_logits1 = get_logits(literal_model, literal_test, device)
    
    literal_model.to(torch.device("cpu"))
    torch.cuda.empty_cache()

    test_predictions = [np.argmax(ltr_p) for ltr_p in literal_test_logits1]
    accuracy = simple_accuracy(test_predictions, test_gold)
    logger.info(f"Literal Test accuracy: {accuracy*100:.1f}")
    res.write("Literal Test accuracy: "+str(accuracy)+'\n\n')

    # Tune the weight of each model
    alphas = [0.1 * i for i in range(11)]
    accuracies = []
    dev_gold = [int(json.loads(line)["correctanswer"][len("option"):]) - 1 for line in open(f"{args.context_data_dir}/dev.jsonl")]
    
    for alpha in alphas:
        dev_predictions = [np.argmax(alpha * ctx_p + (1 - alpha) * ltr_p) for ctx_p, ltr_p in zip(context_dev_logits, literal_dev_logits)]
        accuracies.append(simple_accuracy(dev_predictions, dev_gold))
        
    best_index = np.argmax(accuracies)
    best_alpha = alphas[best_index]
    logger.info(
        f"Best: context = {best_alpha:.1f}, literal = {1-best_alpha:.1f}, Dev accuracy: {accuracies[best_index]*100:.1f}")
    
    # Compute test predictions and report accuracy    
    test_predictions = [np.argmax(best_alpha * ctx_p + (1 - best_alpha) * ltr_p) 
                        for ctx_p, ltr_p in zip(context_test_logits, literal_test_logits)]
    test_gold = [int(json.loads(line)["correctanswer"][len("option"):]) - 1 for line in open(f"{args.context_data_dir}/test.jsonl")]
    accuracy = simple_accuracy(test_predictions, test_gold)
    logger.info(f"Test accuracy: {accuracy*100:.1f}")
    

def get_logits(model, examples, device):
    """
    Forward the examples in the model and return the logits
    :return: a list of logits
    """
    with torch.no_grad():
        logits = []
        logits1 = []
        for i in tqdm.tqdm(range(len(examples.features))):
            ip = torch.Tensor([examples.features[i].input_ids]).long().to(device)
            att = torch.Tensor([examples.features[i].attention_mask]).long().to(device) 
            op = model(input_ids=ip,attention_mask=att)
            log = op[0]
            soft = F.softmax(log,dim=1).cpu().numpy()
            logits.append(soft)
            logits1.append(log.cpu().numpy())
    
    return logits, logits1


if __name__ == "__main__":
    main()
