"""
Compared to plot_accuracy.py, makes plotting easier by reversing the labels (1024->1 etc.)

Prints top-k accuracy of ChrF as a baseline when COMET is the utility metric.
"""
import argparse

from experiments.reference_aggregation.plot_accuracy import main


parser = argparse.ArgumentParser()
parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
parser.add_argument('--seed', type=int, choices=range(10), required=True,
                    help='Index of the random seed in the list of random seeds')
parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
parser.add_argument('--topk', type=int, default=20,
                    help='Number of top translations that have been saved in the jsonl file')
parser.add_argument('--method', choices=['n_by_s', 'aggregate'], required=True)
parser.add_argument('--num-samples', type=int, default=1024)
parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
parser.add_argument('--accuracy-topk', type=int, default=None,
                    help='Number of top translations that are used to compute the accuracy (default: same as data-topk)')
parser.add_argument('--limit-segments', type=int, default=None,
                    help='Limit number of segments that are processed (used for testing)')
args = parser.parse_args()

if args.accuracy_topk is None:
    args.accuracy_topk = args.topk

series = main(
    testset=args.testset,
    language_pair=args.language_pair,
    seed_no=args.seed,
    utility_name=args.utility,
    topk=args.topk,
    method=args.method,
    num_samples=args.num_samples,
    epsilon_cutoff=args.epsilon_cutoff,
    accuracy_topk=args.accuracy_topk,
    limit_segments=args.limit_segments,
)

s_values = [s for s, _ in series]
reversed_s_values = list(reversed(s_values))
series_str = "".join([f"({s},{accuracy:.5f})" for s, accuracy in zip(reversed_s_values, [accuracy for _, accuracy in series])])
print(
    f"Testset: {args.testset}, language pair: {args.language_pair}, seed: {args.seed}, utility: {args.utility}, topk: {args.topk}, method: {args.method}")
print(f"Top-{args.accuracy_topk} accuracy:")
print(series_str)
print()

# # Print top-k accuracy of ChrF as a baseline
# if args.utility in {'cometinho', 'comet22'}:
#     chrf_series = main(
#         testset=args.testset,
#         language_pair=args.language_pair,
#         seed_no=args.seed,
#         utility_name='chrf',
#         topk=args.topk,
#         method=args.method,
#         num_samples=args.num_samples,
#         epsilon_cutoff=args.epsilon_cutoff,
#         accuracy_topk=args.accuracy_topk,
#         limit_segments=args.limit_segments,
#     )
#     chrf_n_by_n_accuracy = chrf_series[0][1]
#     print(f"Top-{args.accuracy_topk} accuracy of ChrF: {chrf_n_by_n_accuracy:.5f}")
