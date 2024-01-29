"""
Calculates the top-k accuracy of ChrF when used as a coarse metric for COMET
"""
import argparse
from pathlib import Path

import jsonlines

from experiments.reference_aggregation.experiment_utils import Testset

parser = argparse.ArgumentParser()
parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
parser.add_argument('--seed', type=int, choices=range(10), required=True,
                    help='Index of the random seed in the list of random seeds')
parser.add_argument('--fine_utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
parser.add_argument('--topk', type=int, default=20,
                    help='Number of top translations that have been saved in the jsonl file')
parser.add_argument('--num-samples', type=int, default=1024)
parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
parser.add_argument('--accuracy-topk', type=int, default=None,
                    help='Number of top translations that are used to compute the accuracy (default: same as data-topk)')
parser.add_argument('--limit-segments', type=int, default=None,
                    help='Limit number of segments that are processed (used for testing)')
args = parser.parse_args()

out_dir = Path(__file__).parent

testset = args.testset
language_pair = args.language_pair
seed_no = args.seed
utility_name = args.fine_utility
topk = args.topk
num_samples = args.num_samples
epsilon_cutoff = args.epsilon_cutoff
accuracy_topk = args.accuracy_topk
limit_segments = args.limit_segments

assert topk <= num_samples
assert accuracy_topk <= topk

dataset = Testset.from_wmt(testset, language_pair, limit_segments=limit_segments)

samples_dir = out_dir / "samples"
assert samples_dir.exists()
samples_path = samples_dir / f"samples.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.jsonl"
assert samples_path.exists()
with jsonlines.open(samples_path) as f:
    samples = [line["samples"] for line in f]
samples = [sample[:num_samples] for sample in samples]
if limit_segments is not None:
    samples = samples[:limit_segments]

output_dir = out_dir / "validation_output"
assert output_dir.exists()
fine_output_path = output_dir / f"validation.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.{utility_name}.top{topk}.jsonl"

with jsonlines.open(fine_output_path) as f:
    fine_data = list(f)

chrf_output_path = output_dir / f"validation.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.chrf.top{topk}.jsonl"

with jsonlines.open(chrf_output_path) as f:
    chrf_data = list(f)

# Get n-by-n top-1 samples – should not matter which method
fine_n_by_n_lines = [line for line in fine_data if line["s"] == num_samples]
assert len(fine_n_by_n_lines) == 2
for ranking in zip(fine_n_by_n_lines[0]["rankings"], fine_n_by_n_lines[1]["rankings"]):
    assert ranking[0] == ranking[1]
fine_n_by_n_rankings = fine_n_by_n_lines[0]["rankings"]
fine_n_by_n_top1_samples = [samples[i][fine_n_by_n_rankings[i][0]].strip() for i in range(len(samples))]

# Get top-k accuracies for chrf (n-by-n)
chrf_method_lines = [line for line in chrf_data if line["s"] == num_samples]
assert len(chrf_method_lines) == 2
for ranking in zip(chrf_method_lines[0]["rankings"], chrf_method_lines[1]["rankings"]):
    assert ranking[0] == ranking[1]
chrf_n_by_n_rankings = chrf_method_lines[0]["rankings"]
chrf_n_by_n_topk_samples = [{samples[i][ranking].strip() for ranking in chrf_n_by_n_rankings[i][:accuracy_topk]} for i in
                  range(len(samples))]

num_correct = sum([1 if fine_n_by_n_top1_samples[i] in chrf_n_by_n_topk_samples[i] else 0 for i in range(len(samples))])
accuracy = num_correct / len(samples)

print(
    f"Testset: {args.testset}, language pair: {args.language_pair}, seed: {args.seed}, fine utility: {args.fine_utility}, topk: {args.topk}, method: {args.method}")
print(f"Top-{args.accuracy_topk} accuracy of chrf:")
print(f"Accuracy: {accuracy:.5f}")
