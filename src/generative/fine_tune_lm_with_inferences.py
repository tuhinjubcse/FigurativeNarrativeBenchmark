import os
import json
import torch
import logging
import argparse

from src.generative.common import init_model, set_seed
from src.generative.fine_tune_lm import train, evaluate
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Saves examples with the current format, tokenized and indexed according to the LM vocabulary
    Output: target <eos>
    """
    def __init__(self, tokenizer, args, file_path="train", block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        filename = f"{args.model_name_or_path}_cached_{block_size}_{filename}"

        cached_features_file = os.path.join(directory, filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Converting to token IDs")
            examples = load_data(file_path)
            logger.info(examples[:5])

            sequences = tokenizer.batch_encode_plus(
                [inp for inp in examples], add_special_tokens=False,
                padding=True, truncation=True,
                max_length=args.max_input_length)

            self.examples = {
                "examples": sequences["input_ids"],
                "mask": sequences["attention_mask"]
            }

        logger.info(f"Saving features into cached file {cached_features_file}")
        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples["examples"])

    def __getitem__(self, item):
        return {
            "examples": torch.tensor(self.examples["examples"][item]),
            "input_mask": torch.tensor(self.examples["mask"][item])
        }



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        required=True,
        help="Out directory for checkpoints.",
    )

    # Other parameters
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--eval_batch_size", default=64, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_data_file",
        type=str,
        required=True,
        help="The input CSV validation file."
    )
    parser.add_argument(
        "--eval_during_train",
        default=False,
        action="store_true",
        help="Evaluate at each train logging step.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Steps before backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-6,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10000, help="Log every X updates steps."
    )
    parser.add_argument(
        "--max_input_length",
        default=512,
        type=int,
        help="Maximum input event length in words.",
    )
    parser.add_argument(
        "--max_output_length",
        default=50,
        type=int,
        help="Maximum output event length in words.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: total number of training steps to perform.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="openai-gpt",
        type=str,
        help="LM checkpoint for initialization.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Number of training epochs to perform.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached data."
    )
    parser.add_argument(
        "--overwrite_out_dir",
        action="store_true",
        help="Overwrite the output directory.",
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from the last checkpoint.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--train_batch_size", default=64, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=False,
        help="The input CSV train file."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
            os.path.exists(args.out_dir)
            and os.listdir(args.out_dir)
            and args.do_train
            and not args.overwrite_out_dir
            and not args.continue_training
    ):
        raise ValueError(
            f"Output directory {args.out_dir} already exists and is not empty. "
            f"Use --overwrite_out_dir or --continue_training."
        )

    # Setup device
    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    args.device = device

    # Set seed
    set_seed(args.seed)

    # Load the data
    logger.info(f"Creating features from dataset file at {args.train_file}")

    # Load the models
    if args.continue_training:
        args.model_name_or_path = args.out_dir

    tokenizer, model = init_model(
        args.model_name_or_path, device=args.device, do_lower_case=args.do_lower_case,
        is_trained=(not args.do_train) or args.continue_training
    )
    args.block_size = tokenizer.max_len_single_sentence
    model.to(args.device)
    logger.info(f"Training/evaluation parameters {args}")

    args.pad_token_id = tokenizer.pad_token_id

    if args.do_eval or args.eval_during_train:
        eval_dataset = load_and_cache_examples(args.eval_data_file, args, tokenizer)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args.train_file, args, tokenizer)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, eval_dataset=eval_dataset)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

        # Create output directory if needed
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        logger.info(f"Saving model checkpoint to {args.out_dir}")

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.out_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        tokenizer, model = init_model(
            args.out_dir, device=args.device, do_lower_case=args.do_lower_case, is_trained=True
        )
        args.block_size = tokenizer.max_len_single_sentence
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint = args.out_dir
        logger.info(f"Evaluate the following checkpoint: {checkpoint}")
        prefix = (
            checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        )
        _, model = init_model(checkpoint, device=args.device, is_trained=True)
        model.to(args.device)
        result = evaluate(eval_dataset, args, model, prefix=prefix)
        results.update(result)

    return results


def load_and_cache_examples(file_path, args, tokenizer):
    """
    Load the dataset from the cache or from the CSV file
    """
    return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def load_data(in_file):
    """
    Loads the dataset file

    in_file: json file with "narrative", "option1", "option2", "correctanswer", "inferences" fields

    Returns a list of tuples (input, output)
    """
    examples = [json.loads(line.strip()) for line in open(in_file)]
    arr = []
    for ex in examples:
        for inference in ex["inferences"]:
            value = inference+' @@@ '+ex["narrative"].replace("<b>", "").replace("</b>", "")+' ====== '+ex[ex["correctanswer"]]
            arr.append(ex)
    return arr


if __name__ == "__main__":
    main()