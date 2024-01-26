import argparse
from pathlib import Path

from experiment_utils import Testset


parser = argparse.ArgumentParser()

parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
parser.add_argument('--seed', type=int, choices=range(10), required=True, help='Index of the random seed in the list of random seeds')
parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
parser.add_argument('--segments', type=int, default=None, help='Limit number of segments that are processed (used for testing)')

args = parser.parse_args()

out_dir = Path(__file__).parent / "validation_output"
out_dir.mkdir(exist_ok=True)

testset = Testset.from_wmt(args.testset, args.language_pair, limit_segments=args.segments)

