import argparse
from pathlib import Path

import numpy as np
from evaluate import load


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', required=True)
    parser.add_argument('-t', '--translation', required=True)
    parser.add_argument('--model', default="bleurt-base-128")
    args = parser.parse_args()

    references = Path(args.reference).read_text().splitlines()
    translations = Path(args.translation).read_text().splitlines()
    assert len(references) == len(translations)

    bleurt = load("bleurt", module_type="metric")
    results = bleurt.compute(predictions=translations, references=references)
    # {'scores': [1.0295495986938477, 1.0445427894592285]}
    score = np.mean(results["scores"])
    print(f"{args.translation}\t{args.model}\t{score}")
