import json
import torch
import logging
import argparse
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

np.random.seed(0)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--eval_file",
        default=None,
        type=str,
        required=True,
        help="The dev/test set json file from which examples will be used for evaluation.",
    )

    # Optional parameters
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-3b")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-t5-3b")
    model = model.to(device)

    correct = 0
    random = 0
    incorrect = 0

    with open(args.eval_file) as f_in:
        for line in f_in:
            ex = json.loads(line.strip())
            narrative = ex["narrative"].replace('<b>', '').replace('</b>', '')
            options = [ex["option1"], ex["option2"]]
            correct_idx = int(json.loads(line)["correctanswer"][len("option"):]) - 1
            incorrect_idx = 1 - correct_idx

            inp = "Which is more plausible between the two based on the context ? \\n " + "(A) " + \
                  options[0] + " (B) " + options[1] + " \\n " + narrative

            prediction = run_model(tokenizer, model, inp)

            if prediction.lower() in options[correct_idx].lower():
                correct = correct + 1

            elif prediction.lower() in options[incorrect_idx].lower():
                incorrect = incorrect + 1

            else:
                random = random + 1

    accuracy = float(correct) / float(correct + incorrect + random)
    logging.info(f"Correct: {correct}, incorrect: {incorrect}, random: {random}, accuracy: {accuracy*100.0:.1f}")


def run_model(tokenizer, model, input_string, **generator_args):
    input_ids = tokenizer.encode(input_string.lower(), return_tensors="pt")
    input_ids = input_ids.to('cuda')
    res = model.generate(input_ids, **generator_args, max_length=128)
    return tokenizer.batch_decode(res, skip_special_tokens=True)[0].lower()


if __name__ == "__main__":
    main()
