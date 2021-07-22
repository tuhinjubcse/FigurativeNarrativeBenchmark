import os
import json
import argparse
import itertools
import numpy as np

from rouge import Rouge
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

rouge = Rouge()


def main():
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--set", default="test", type=str, help="Which set to evaluate.")
    parser.add_argument("--device", type=int, required=False, help="CUDA device number or -1 for CPU", default=-1)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device > 0 else "cpu"
    bert_scorer = BERTScorer(lang="en", model_type="microsoft/deberta-large-mnli", device=device)

    print("\t".join(["LM", "Decoding", "BLEU-4", "ROUGE-L", "BertScore"]))

    for lm in ["gpt2-xl", "gpt3", "bart-large", "t5-large"]:
        for mode in ["", "_zeroshot", "_fewshot", "_context", "_literal"]:
            for decoding in ["k5", "p0.9"]:
                file_path = f"output/generative/{lm}{mode}_{args.set}_predictions_{decoding}.jsonl"

                if not os.path.exists(file_path):
                    continue

                data = [json.loads(line.strip()) for line in open(file_path)]

                gold = [[g.replace("<eos>", "").replace("<pad>", "").strip()
                         for g in ex["gold"]] for ex in data]
                predictions = [[pred.replace("<eos>", "").replace("<pad>", "").strip()
                                for pred in ex["predictions"]] for ex in data]

                # Cut after first period
                predictions = [[pred[:pred.rfind(".")] if "." in pred else pred for pred in preds]
                               for preds in predictions]

                # Remove empty predictions
                predictions = [[pred for pred in preds if len(pred) != 0] for preds in predictions]

                blue_score = np.mean([sentence_level_bleu(g, preds) if len(preds) > 0 else 0.0
                                      for g, preds in zip(gold, predictions)]) * 100.0

                rouge_score = np.mean([sentence_level_rouge(rouge, g, preds) if len(preds) > 0 else 0.0
                                       for g, preds in zip(gold, predictions)]) * 100.0

                bert_score = np.mean([compute_bert_score(bert_scorer, g, preds) if len(preds) > 0 else 0.0
                                       for g, preds in zip(gold, predictions)]) * 100.0

                print("\t".join([lm + mode, decoding, f"{blue_score:.3f}", f"{rouge_score:.3f}", f"{bert_score:.3f}"]))


def sentence_level_bleu(references, hypotheses):
    """
    Compute the maximum sentence-level BLEU score across references and hypotheses.
    """
    references = [ref.split() for ref in references]
    hypotheses = [hyp.split() for hyp in hypotheses]

    bleu_score = max(
        [sentence_bleu(ref, hyp, weights=[0.25] * 4, smoothing_function=SmoothingFunction().method1)
         for ref in references
         for hyp in hypotheses])

    return bleu_score


def sentence_level_rouge(rouge, references, hypotheses):
    """
    Compute the maximum sentence-level ROUGE score across references and hypotheses.
    """
    return max([rouge.get_scores([hyp], [ref])[0]["rouge-l"]["f"] for ref in references for hyp in hypotheses])


def compute_bert_score(bert_scorer, references, hypotheses):
    """
    Compute the maximum sentence-level BERTScore across references and hypotheses.
    """
    references, hypotheses = zip(*itertools.product(references, hypotheses))
    bert_score = bert_scorer.score(references, hypotheses)[-1].detach().numpy().max()
    return bert_score


if __name__ == "__main__":
    main()
