import argparse
from pathlib import Path

from experiment_utils import Testset


def main(testset: str, language_pair: str, seed: int, utility: str, topk: int = 20, limit_segments: int = None, out_dir: Path = None):
    if out_dir is None:
        out_dir = Path(__file__).parent

    output_dir = out_dir / "validation_output"
    output_dir.mkdir(exist_ok=True)

    testset = Testset.from_wmt(args.testset, args.language_pair, limit_segments=args.segments)

    # TODO Run mbr and save jsonl output

    translations_dir = out_dir / "translations"
    translations_dir.mkdir(exist_ok=True)

    # TODO Extract translations from jsonl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
    parser.add_argument('--topk', type=int, default=20, help='Number of top translations to save in the jsonl files')
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    args = parser.parse_args()

    main(args.testset, args.language_pair, args.seed, args.utility, args.topk, args.limit_segments)
