import json
import tqdm
import torch
import logging
import argparse
import pickle
from nltk.corpus import words
from collections import defaultdict
from transformers.generation_utils import top_k_top_p_filtering


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

from src.generative.common import init_model, set_seed


def main() -> None:
    """
    Generate intensifiers and attenuators
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--in_file",
        default=None,
        type=str,
        required=True,
        help="The input jsonl file",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        required=True,
        help="out jsonl file with generations",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="LM checkpoint for initialization.",
    )

    # Optional
    parser.add_argument(
        "--is_encoder_decoder", default=False, action="store_true",
        required=False, help="True for Bart / T5"
    )
    parser.add_argument(
        "--is_zero_shot", default=False, action="store_true",
        required=False, help="If true, will not include special characters"
    )
    parser.add_argument(
        "--max_length", default=20, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--p", default=0, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    args = parser.parse_args()
    logger.debug(args)

    set_seed(args.seed)

    if (
        (args.k == args.p == 0)
        or (args.k != 0 and args.p != 0)
    ):
        raise ValueError(
            "Exactly one of p and k should be set to a non-zero value."
        )

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(
        args.model_name_or_path, device, is_trained=True, is_encoder_decoder=args.is_encoder_decoder)

    # If using off-the-shelf GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    examples = [json.loads(line.strip()) for line in open(args.in_file)]

    # Group continuations and inferences of the same narrative
    gold = defaultdict(list)
    [gold[ex["narrative"]].append(ex[ex["correctanswer"]] + " <eos>") for ex in examples]
    inferences = {ex["narrative"]: ex["inferences"] for ex in examples}

    with open(args.out_file, "w") as f_out:
        for narrative, curr_golds in tqdm.tqdm(gold.items()):
            curr_inferences = inferences[narrative]
            torch.cuda.empty_cache()
            preds = generate_from_multiple_inferences(tokenizer, model, args, narrative, curr_inferences, device)
            f_out.write(json.dumps({"input": narrative, "gold": curr_golds, "predictions": preds}) + "\n")


def generate_from_multiple_inferences(tokenizer, model, args, narrative, inferences, device):
    """
    Generate a text by averaging the logits of all inputs
    """
    # Embed all inputs (narrative + inference) and average them
    arr = []
    for inference in inferences:
        value = inference+' @@@ '+narrative.replace("<b>", "").replace("</b>", "")+' ====== '
        arr.append(value)

    inputs = tokenizer(arr,padding=True, return_tensors="pt").to(device)
    curr_input = inputs.input_ids
    attention_mask = inputs.attention_mask
    position_ids = inputs.attention_mask.cumsum(dim=1) - 1
    generated_tokens = []
    num_inferences = curr_input.shape[0]

    with torch.no_grad():
        for step in range(args.max_length):
            # Compute the logits
            logits = model(curr_input, attention_mask=attention_mask, position_ids=position_ids).logits

            # Average the logits
            ensemble_logits = logits.sum(dim=0).unsqueeze(0)

            # Get the last non padded token
            if step == 0:
                last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                last_non_masked_idx = last_non_masked_idx.max()
                next_token_logits = ensemble_logits[range(1), last_non_masked_idx, :]
            else:
                next_token_logits = ensemble_logits[:, -1, :]

            # Predict next token and concatenate it to each of the inferences.
            next_token_logits /= args.temperature
            filtered_next_token_logits = top_k_top_p_filtering(
                next_token_logits,
                top_k=args.k if args.k > 0 else -1,
                top_p=args.p if args.p > 0 else -1)

            probs = torch.nn.functional.softmax(filtered_next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token)

            # Update the inputs for the next step
            curr_input = torch.cat([curr_input, next_token.repeat(num_inferences, 1)], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((num_inferences, 1))], dim=1)
            position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

    # Decode
    generated_tokens = torch.cat(generated_tokens, dim=-1).squeeze()
    pred = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Remove periods and spaces in the beginning
    pred = [p.strip() for p in pred.split(".") if len(p) > 0]
    pred = pred[0] if len(pred) > 0 else ""
    if '?' in pred:
        pred = pred.split('?')[0]
    pred = pred.replace(' i ',' I ').capitalize()

    l_words = pred.split()
    last_word = l_words[-1]
    if last_word[-1].isalpha() and last_word not in words.words():
        while last_word not in words.words():
            last_word  = last_word[:-1]
    pred = pred.replace(l_words[-1],last_word)
    return [pred]


if __name__ == "__main__":
    main()
