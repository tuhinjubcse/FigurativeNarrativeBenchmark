import re
import gzip
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase", type=str, help="Which phrase to search for.")
    parser.add_argument("--corpus_dir", type=str, help="The directory of the book corpus gz files")
    args = parser.parse_args()
    print(find_in_corpus(args.phrase, args.corpus_dir))


def detokenize(string):
    """
    Copied from https://github.com/peterwestuw/ZeroshotScoring/blob/main/utils.py
    """
    # ari custom
    string = string.replace("`` ", '"')
    string = string.replace(" ''", '"')
    string = string.replace("` ", '"')
    string = string.replace(" ' ", '" ')
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" :", ":")
    string = string.replace(" ;", ";")
    string = string.replace(" .", ".")
    string = string.replace(" !", "!")
    string = string.replace(" ?", "?")
    string = string.replace(" ,", ",")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    # string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    # string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    # ari custom
    string = string.replace(" n't ", "n't ")
    string = string.replace(" 'd ", "'d ")
    string = string.replace(" 'm ", "'m ")
    string = string.replace(" 're ", "'re ")
    string = string.replace(" 've ", "'ve ")
    return string


def find_in_corpus(phrase, corpus_dir):
    """
    Returns paragraphs containing this phrase 
    """
    phrases = [phrase]
    
    # Replace the word be with its inflected forms
    if "be" in set(phrase.split()):
        phrases += [re.sub(r"\bbe\b", w, phrase) for w in {"is", "are", "were", "was"}]
        
    for part in [1, 2]:
        with gzip.open(f"{corpus_dir}/books_large_p{part}.txt.gz", "rb") as f_in:
            prev = [""] * 5
            for line in f_in:
                line = detokenize(line.decode().strip())
                for phrase in phrases:
                    if phrase in line:
                        line = line.replace(phrase, f"<target>{phrase}</target>")
                        print(" ".join(prev + [line]) + "\n")

                prev = prev[-4:] + [line]

        
if __name__ == '__main__':
    main()
