import json
import tqdm
import torch
import logging
import argparse

from collections import defaultdict

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
        "--beams", default=0, type=int, required=False, help="beams for beam search"
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

    if (
        (args.k == args.p == args.beams == 0)
        or (args.k != 0 and args.p != 0)
        or (args.beams != 0 and args.p != 0)
        or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    set_seed(args.seed)

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(
        args.model_name_or_path, device, is_trained=True, is_encoder_decoder=args.is_encoder_decoder)
    examples = [json.loads(line.strip()) for line in open(args.in_file)]

    generate = generate_conditional if args.is_encoder_decoder else generate_regular

    # Group continuations of the same narrative
    gold = defaultdict(list)
    [gold[ex["narrative"]].append(ex[ex["correctanswer"]] + " <eos>") for ex in examples]

    with open(args.out_file, "w") as f_out:
        for narrative, curr_golds in tqdm.tqdm(gold.items()):

            input_text = narrative
            if args.is_zero_shot:
                input_text = input_text.replace("<b>", "").replace("</b>", "").replace("<eos>", "").strip()

            try:
                preds = generate(
                    tokenizer,
                    model,
                    args,
                    input_text,
                    device,
                )

            except Exception as exp:
                logger.info(exp)
                preds = []

            f_out.write(
                json.dumps({"input": narrative, "gold": curr_golds, "predictions": preds})
                + "\n"
            )


def generate_conditional(tokenizer, model, args, input, device):
    """
    Generate a sequence with models like Bart and T5
    """
    input_ids = tokenizer(input, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    preds = model.generate(
        input_ids=input_ids,
        do_sample=args.beams == 0,
        max_length=args.max_length,
        min_length=2,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=max(1, args.beams)
    ).tolist()

    preds = [pred[:pred.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in pred else pred for pred in preds]
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    return preds


def generate_regular(tokenizer, model, args, input, device):
    """
    Generate a sequence with models like GPT, GPT2, or XLNet
    """
    context_tokens = tokenizer.encode(input)
    max_length = args.max_length + len(context_tokens)
    input_ids = torch.tensor(context_tokens, device=device).unsqueeze(0)

    outputs = model.generate(
        input_ids=input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=max(1, args.beams)
    )

    preds = [tokenizer.decode(output)[len(input):].strip() for output in outputs]
    preds = [pred.split(".")[0] for pred in preds]

    return preds


if __name__ == "__main__":
    main()