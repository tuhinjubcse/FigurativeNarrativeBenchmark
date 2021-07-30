# Based on transformers/examples/multiple_choice/run_multiple_choice.py
import os
import json
import tqdm
import torch
import logging

from filelock import FileLock
from typing import List, Optional
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from transformers.data.processors.utils import DataProcessor

from src.discriminative.common import Split

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class MultipleChoiceDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        processor = processors[task]()

        cached_features_file = os.path.join(
            data_dir,
            f"cached_{mode.value}_{tokenizer.__class__.__name__, str(max_seq_length)}_{task}"
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                label_list = processor.get_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)
                logger.info("Training examples: %s", len(examples))
                self.features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,)
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class IdiomProcessor(DataProcessor):
    """Processor for the idiom data set."""
    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        examples = [json.loads(line) for line in open(os.path.join(data_dir, f"train.jsonl"))]
        return self._create_examples(examples)

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        examples = [json.loads(line) for line in open(os.path.join(data_dir, f"dev.jsonl"))]
        return self._create_examples(examples)

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        examples = [json.loads(line) for line in open(os.path.join(data_dir, f"test.jsonl"))]
        return self._create_examples(examples)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, examples):
        """Creates examples for the training and dev sets."""
        examples = [InputExample(
            example_id=str(ex_id+1),
            question="",
            contexts=[ex["narrative"].replace('<b>','').replace('</b>',''),ex["narrative"].replace('<b>','').replace('</b>','')],
            endings=[ex["option1"], ex["option2"]],
            label=str(int(ex["correctanswer"][len("option"):]) - 1),
        ) for ex_id, ex in enumerate(examples)]

        return examples

class SimileProcessor(DataProcessor):
    """Processor for the idiom data set."""
    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        examples = [json.loads(line) for line in open(os.path.join(data_dir, f"train.jsonl"))]
        return self._create_examples(examples)

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        examples = [json.loads(line) for line in open(os.path.join(data_dir, f"dev.jsonl"))]
        return self._create_examples(examples)

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        examples = [json.loads(line) for line in open(os.path.join(data_dir, f"test.jsonl"))]
        return self._create_examples(examples)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, examples):
        """Creates examples for the training and dev sets."""
        examples = [InputExample(
            example_id=str(ex_id+1),
            question="",
            contexts=[ex["narrative"].replace('<b>','').replace('</b>',''),ex["narrative"].replace('<b>','').replace('</b>','')],
            endings=[ex["option1"], ex["option2"]],
            label=str(int(ex["correctanswer"][len("option"):]) - 1),
        ) for ex_id, ex in enumerate(examples)]

        return examples


def convert_examples_to_features(
    examples: List[InputExample], label_list: List[str], max_length: int, tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            text_b = ending
            print(text_a, '||', text_b)
            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


processors = {"idiom": IdiomProcessor,"simile": SimileProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"idiom", 2, "simile", 2}
