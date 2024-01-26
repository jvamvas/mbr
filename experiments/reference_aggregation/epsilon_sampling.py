import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
parser.add_argument('--seed', type=int, choices=range(10), required=True, help='Index of the random seed in the list of random seeds')
parser.add_argument('--num-samples', type=int, default=1024)
parser.add_argument('--epsilon-cutoff', type=float, default=0.02)

args = parser.parse_args()

samples_dir = Path(__file__).parent / "samples"
assert samples_dir.exists()
samples_path = samples_dir / f"samples.{args.testset}.{args.language_pair}.n{args.num_samples}.epsilon{args.epsilon_cutoff}.seed{args.seed}.jsonl"
assert samples_path.exists()

out_dir = Path(__file__).parent / "translations"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"{args.testset}.{args.language_pair}.epsilon{args.epsilon_cutoff}.seed{args.seed}.{args.language_pair.split('-')[1]}"

with jsonlines.open(samples_path) as f_in, open(out_path, "w") as f_out:
    for line in tqdm(f_in):
        samples = line["samples"]
        assert len(samples) == args.num_samples
        f_out.write(samples[0] + "\n")
