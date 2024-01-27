import argparse
from pathlib import Path


def main(testset: str, language_pair: str, seed: int, utility: str, topk: int, method: str, out_dir: Path = None):
    if out_dir is None:
        out_dir = Path(__file__).parent

    output_dir = out_dir / "validation_output"
    assert output_dir.exists()

    # TODO Read output, calculate accuracy and print formatted series


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
    parser.add_argument('--topk', type=int, default=20, help='Number of top translations to save in the jsonl files')
    parser.add_argument('--method', choices=['nbys', 'aggregate'], required=True)
    args = parser.parse_args()

    main(args.testset, args.language_pair, args.seed, args.utility, args.topk, args.method)
