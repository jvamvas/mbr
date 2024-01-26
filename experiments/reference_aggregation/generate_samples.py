import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm

from fairseq_utils import load_model
from experiment_utils import SEEDS, Testset


parser = argparse.ArgumentParser()

parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
parser.add_argument('--seed', type=int, choices=range(10), required=True, help='Index of the random seed in the list of random seeds')
parser.add_argument('--num-samples', type=int, default=1024)
parser.add_argument('--epsilon-cutoff', type=float, default=0.02)

args = parser.parse_args()

seed = SEEDS[args.seed]

testset = Testset.from_wmt(args.testset, args.language_pair)

model = load_model(args.language_pair)

out_dir = Path(__file__).parent / "samples"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"samples.{args.testset}.{args.language_pair}.n{args.num_samples}.epsilon{args.epsilon_cutoff}.seed{args.seed}.jsonl"

with jsonlines.open(out_path, "w") as f:
    for source_sentence in tqdm(testset.source_sentences):
        f.write({
            "samples": model.sample(args.num_samples * [source_sentence], seed=seed, sampling_epsilon_cutoff=args.epsilon_cutoff),
        })
