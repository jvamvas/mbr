import argparse
from pathlib import Path

from experiments.reference_aggregation.experiment_utils import Testset


def main(method: str, topk: int, testset: str, language_pair: str, seed: int, utility: str, limit_segments: int = None, out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent

    translations_dir = out_dir / "translations"
    translations_dir.mkdir(exist_ok=True)

    dataset = Testset.from_wmt(testset, language_pair, limit_segments=limit_segments)

    out_path = ...  # TODO
    # TODO Run mbr and save translations to file
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['pairwise', 'aggregate', 'aggregate_to_fine'], required=True)
    parser.add_argument('--topk', type=int, default=20,
                        help='Number of samples to prune to in aggregate_to_fine method')
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    args = parser.parse_args()

    out_path = main(args.method, args.topk, args.testset, args.language_pair, args.seed, args.utility, args.limit_segments)
    assert out_path.exists()
    print(f"Saved translations to {out_path}")
