import os
import tqdm
import json
import torch
import logging

from filelock import FileLock
from typing import List, Optional
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataset import Dataset
from transformers import RobertaForMultipleChoice
from transformers.data.processors.utils import DataProcessor

from src.discriminative.common import Split
from src.discriminative.utils_multiple_choice import InputFeatures


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        context: str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    example_id: str
    inferences: List[str]
    context: str
    endings: List[str]
    label: Optional[str]


class MultipleChoiceDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer,
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


class IdiomWithInferencesProcessor(DataProcessor):
    """Processor for the Idiom data set with inferences."""
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
            context=ex["narrative"].replace('<b>', '').replace('</b>', ''),
            endings=[ex["option1"], ex["option2"]],
            label=str(int(ex["correctanswer"][len("option"):]) - 1),
            inferences=ex["inferences"]
        ) for ex_id, ex in enumerate(examples)]

        return examples


def convert_examples_to_features(examples: List[InputExample],
                                 label_list: List[str],
                                 max_length: int,
                                 tokenizer) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        label = label_map[example.label]

        # First, tokenize all sequences together, so we can pad according to the longest sequence
        tokenized = tokenizer.batch_encode_plus(
            [(" </s> ".join((example.context, inference.replace('personx', '').strip())), choice)
             for choice in example.endings
             for inference in example.inferences],
            add_special_tokens=True, padding=True, truncation=True, return_overflowing_tokens=True)

        # List of num_choices * Tokenizer outputs of num_inferences
        num_sequences = len(tokenized["input_ids"])
        choice_split = int(num_sequences/2)
        input_ids = [tokenized["input_ids"][:choice_split], tokenized["input_ids"][choice_split:]]
        attention_mask = (
            [tokenized["attention_mask"][:choice_split], tokenized["attention_mask"][choice_split:]]
            if "attention_mask" in tokenized else None)
        token_type_ids = (
            [tokenized["token_type_ids"][:choice_split], tokenized["token_type_ids"][choice_split:]]
            if "token_type_ids" in tokenized else None)

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


class RobertaForMultipleChoiceWithInferences(RobertaForMultipleChoice):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # Inputs are [batch_size=1, num_choices, num_inferences, max_len]

        # Compute the embeddings of each choice: list of num_choices [num_inferences, emb_dim]
        embeddings = [self.roberta(input_ids[0, i], attention_mask=attention_mask[0, i]) for i in range(2)]

        # Predict the logits for each choice -> list of num_choices [1]
        scores = [self.classifier(self.dropout(emb.pooler_output)).sum(0) for emb in embeddings]

        # Concatenate all the scores to get the score per choice -> [1, num_choices]
        reshaped_logits = torch.cat(scores, dim=0).unsqueeze(0)

        outputs = (reshaped_logits,) + embeddings[1][2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs


processors = {"idiom_inferences": IdiomWithInferencesProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"idiom_inferences", 2}
