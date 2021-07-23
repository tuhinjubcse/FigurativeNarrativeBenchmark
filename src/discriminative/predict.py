import tqdm
import json
import torch
import logging
import argparse
import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from src.discriminative.common import simple_accuracy, Split
from transformers import AutoTokenizer, RobertaForMultipleChoice
from src.discriminative.utils_multiple_choice_inferences import RobertaForMultipleChoiceWithInferences


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The dataset directory with the dev.jsonl and test.jsonl files",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        required=True,
        help="The model directory.",
    )
    parser.add_argument(
        "--out_prediction_dir",
        default=None,
        type=str,
        required=True,
        help="Where to save the predictions.",
    )

    # Optional parameters
    parser.add_argument(
        "--task_name",
        default="idiom",
        type=str,
        required=True,
        help="Type task: idiom / idiom_inferences",
    )

    args = parser.parse_args()

    if args.task_name == "idiom":
        from src.discriminative.utils_multiple_choice import MultipleChoiceDataset
        model_class = RobertaForMultipleChoice
    else:
        from src.discriminative.utils_multiple_choice_inferences import MultipleChoiceDataset
        model_class = RobertaForMultipleChoiceWithInferences

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = model_class.from_pretrained(args.model_dir)
    model = model.cuda()
    model = model.eval()
    
    # Load the datasets
    logger.info("Loading the datasets")
    dev = MultipleChoiceDataset(
        data_dir=args.data_dir, tokenizer=tokenizer, task=args.task_name, mode=Split.dev)

    test = MultipleChoiceDataset(
        data_dir=args.data_dir, tokenizer=tokenizer, task=args.task_name, mode=Split.test)

    for dataset, name in zip([dev, test], ["dev", "test"]):
        logger.info(f"Predicting the {name} set")
        gold = [int(dataset.features[i].label) for i in range(len(dataset.features))]
        logits = get_logits(model, dataset)
        predictions = np.argmax(logits, axis=-1).squeeze().tolist()
        accuracy = simple_accuracy(predictions, gold)
        logger.info(f"{name[0].upper() + name[1:]} accuracy {accuracy:.1f}")

        # Save the predictions to a file
        with open(f"{args.out_prediction_dir}/{name}_predictions.jsonl", "w") as f_out:
            with open(f"{args.data_dir}/{name}.jsonl") as f_in:
                for line, pred in zip(f_in, predictions):
                    ex = json.loads(line)
                    ex["prediction"] = pred
                    f_out.write(json.dumps(ex) + "\n")


def get_logits(model, examples):
    """
    Forward the examples in the model and return the logits
    :return: a list of logits
    """
    with torch.no_grad():
        logits = [
            model(input_ids=torch.Tensor([examples.features[i].input_ids]).long().cuda(),
                  attention_mask=torch.Tensor([examples.features[i].attention_mask]).long().cuda())[0].cpu().numpy()
            for i in tqdm.tqdm(range(len(examples.features)))]

    return logits


if __name__ == "__main__":
    main()
