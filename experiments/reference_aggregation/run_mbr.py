import argparse
from pathlib import Path

from experiments.reference_aggregation.experiment_utils import Testset


parser = argparse.ArgumentParser()

parser.add_argument('--method', choices=['pairwise', 'aggregate', 'aggregate_to_fine'], required=True)
parser.add_argument('--topk', type=int, default=20, help='Number of samples to prune to in aggregate_to_fine method')
parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
parser.add_argument('--seed', type=int, choices=range(10), required=True, help='Index of the random seed in the list of random seeds')
parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
parser.add_argument('--limit-segments', type=int, default=None, help='Limit number of segments that are processed (used for testing)')

args = parser.parse_args()

translations_dir = Path(__file__).parent / "translations"
translations_dir.mkdir(exist_ok=True)

testset = Testset.from_wmt(args.testset, args.language_pair, limit_segments=args.segments)

# TODO Run mbr and save translations to file