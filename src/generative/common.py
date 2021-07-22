import json
import torch
import random
import numpy as np

from tokenizers import AddedToken
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          T5ForConditionalGeneration, BartForConditionalGeneration)

ENC_DEC_CLASSES = {"t5": T5ForConditionalGeneration,
                   "bart": BartForConditionalGeneration}


def init_model(model_name, device, do_lower_case=False, is_trained=False, is_encoder_decoder=False):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :param do_lower_case: whether the model is lower cased or not
    :param is_trained: is it already a trained model or a pre-trained LM (in which case we need to
    add the special tokens).
    :param is_encoder_decoder: True for Bart / T5
    :return: the model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)

    if is_encoder_decoder:
        family = "bart" if "bart" in model_name else "t5"
        model = ENC_DEC_CLASSES[family].from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add special tokens
    if not is_trained:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.add_special_tokens({'eos_token': '<eos>'})

        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    AddedToken("[narrative]", lstrip=False, rstrip=False),
                    AddedToken("[inference]", lstrip=False, rstrip=False)
                ]
            }
        )

        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()
    return tokenizer, model


def load_data(in_file):
    """
    Loads the dataset file
    
    in_file: json file with "narrative", "option1", "option2", "correctanswer" fields

    Returns a list of tuples (input, output)
    """
    examples = [json.loads(line.strip()) for line in open(in_file)]
    arr = []
    for ex in examples:
        arr.append(ex["narrative"].replace("<b>", "").replace("</b>", "")+' ====== '+ex[ex["correctanswer"]])
    return arr


def set_seed(seed):
    """
    Set the random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
