import argparse
import itertools
import logging
import math
from pathlib import Path

import jsonlines

from experiments.reference_aggregation.experiment_utils import Testset
from experiments.reference_aggregation.mbr_utils import load_utility


def main(testset: str, language_pair: str, seed_no: int, utility: str, topk: int = 20, num_samples: int = 1024, epsilon_cutoff: float = 0.02, limit_segments: int = None, out_dir: Path = None):
    if out_dir is None:
        out_dir = Path(__file__).parent

    output_dir = out_dir / "validation_output"
    output_dir.mkdir(exist_ok=True)

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

    assert len(samples) == len(dataset.source_sentences)
    assert all(len(sample) == num_samples for sample in samples)

    utility = load_utility(utility)

    if hasattr(utility, "compute_features"):
        input_sequences = set(itertools.chain.from_iterable(samples)) | set(dataset.source_sentences) | set(dataset.references)
        logging.info(f"Computing features for {len(input_sequences)} unique input sequences ...")
        utility.compute_features(input_sequences)

    # s = n/1, n/2, n/4, n/8, ..., n/n
    s_values = [int(num_samples / 2**i) for i in range(int(math.log2(num_samples)) + 1)]
    assert s_values[0] == num_samples
    assert s_values[-1] == 1

    # TODO Run mbr and save jsonl output with topk samples

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
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    args = parser.parse_args()

    main(args.testset, args.language_pair, args.seed, args.utility, args.topk, args.num_samples, args.epsilon_cutoff, args.limit_segments)
