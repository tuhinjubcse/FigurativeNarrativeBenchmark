from enum import Enum
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: str = field(metadata={"help": "The name of the task to train on"})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=390,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def simple_accuracy(preds, labels):
    return 100.0 * sum([1 if p == g else 0 for p, g in zip(preds, labels)]) / len(preds)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
