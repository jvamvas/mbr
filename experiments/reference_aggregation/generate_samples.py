import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm

from fairseq_utils import load_model
from experiment_utils import SEEDS, Testset


def main(testset: str, language_pair: str, seed: int, num_samples: int, epsilon_cutoff: float, limit_segments: int = None, out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent

    testset = Testset.from_wmt(testset, language_pair, limit_segments=limit_segments)

    model = load_model(language_pair)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    out_path = samples_dir / f"samples.{testset}.{language_pair}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed}.jsonl"

    with jsonlines.open(out_path, "w") as f:
        for source_sentence in tqdm(testset.source_sentences):
            f.write({
                "samples": model.sample(num_samples * [source_sentence], seed=seed, sampling_epsilon_cutoff=epsilon_cutoff),
            })

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    args = parser.parse_args()
    seed = SEEDS[args.seed]

    out_path = main(args.testset, args.language_pair, seed, args.num_samples, args.epsilon_cutoff, args.limit_segments)
    assert out_path.exists()
    print(f"Saved samples to {out_path}")
