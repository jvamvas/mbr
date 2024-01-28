import argparse
from pathlib import Path

import jsonlines

from experiments.reference_aggregation.experiment_utils import Testset


def main(testset: str, language_pair: str, seed_no: int, utility_name: str, topk: int, method: str, num_samples: int = 1024, epsilon_cutoff: float = 0.02, limit_segments: int = None, out_dir: Path = None):
    if out_dir is None:
        out_dir = Path(__file__).parent

    assert topk <= num_samples

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
    output_path = output_dir / f"validation.{dataset}.n{num_samples}.epsilon{epsilon_cutoff}.seed{seed_no}.{utility_name}.top{topk}.jsonl"

    # TODO Read output, calculate accuracy and print formatted series
    with jsonlines.open(output_path) as f:
        data = list(f)

    # Get n-by-n top-1 samples – should not matter which method
    n_by_n_lines = [line for line in data if line["s"] == num_samples]
    assert len(n_by_n_lines) == 2
    for ranking in zip(n_by_n_lines[0]["rankings"], n_by_n_lines[1]["rankings"]):
        assert ranking[0] == ranking[1]
    n_by_n_rankings = n_by_n_lines[0]["rankings"]
    n_by_n_top1_samples = [samples[i][n_by_n_rankings[i][0]].strip() for i in range(len(samples))]

    # Get top-k accuracies for efficiency method
    method_lines = [line for line in data if line["method"] == method]
    assert len(method_lines) == len(data) / 2
    s_values = list(sorted([line["s"] for line in method_lines], reverse=True))
    accuracies = []  # for each s
    for s in s_values:
        s_lines = [line for line in method_lines if line["s"] == s]
        assert len(s_lines) == 1
        s_rankings = s_lines[0]["rankings"]
        s_topk_samples = [{samples[i][ranking].strip() for ranking in s_rankings[i][:topk]} for i in range(len(samples))]
        s_num_correct = sum([1 if n_by_n_top1_samples[i] in s_topk_samples[i] else 0 for i in range(len(samples))])
        s_accuracy = s_num_correct / len(samples)
        accuracies.append(s_accuracy)

    # Format: (1,-0.4)(2,-0.6)(4,-0.5)(8,0.1)(16,0.1)(32,0.2)(64,0.1)(128,-0.0)(256,-0.0)
    series = [(s, accuracy) for s, accuracy in zip(s_values, accuracies)]
    series_str = "".join([f"({s},{accuracy})" for s, accuracy in series])
    print(f"Testset: {testset}, language pair: {language_pair}, seed: {seed_no}, utility: {utility_name}, topk: {topk}, method: {method}:")
    print(series_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument('--seed', type=int, choices=range(10), required=True,
                        help='Index of the random seed in the list of random seeds')
    parser.add_argument('--utility', choices=['chrf', 'cometinho', 'comet22'], required=True)
    parser.add_argument('--topk', type=int, default=20, help='Number of top translations to save in the jsonl files')
    parser.add_argument('--method', choices=['n_by_s', 'aggregate'], required=True)
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--epsilon-cutoff', type=float, default=0.02)
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    args = parser.parse_args()

    main(args.testset, args.language_pair, args.seed, args.utility, args.topk, args.method, args.num_samples, args.epsilon_cutoff, args.limit_segments)
