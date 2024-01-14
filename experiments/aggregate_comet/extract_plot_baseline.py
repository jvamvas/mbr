import math
import sys
from pathlib import Path
from typing import List, Tuple

import jsonlines

utility_metric = sys.argv[1]
split = sys.argv[2]

if split == "valid":
    wmt = "wmt21"
elif split == "test":
    wmt = "wmt22"
else:
    raise ValueError(split)

language_pairs = ["en-de", "de-en", "en-ru", "ru-en"]
seed_no = 0
evaluation_metric = "comet22"
num_samples = 1024
num_references = num_samples
for language_pair in language_pairs:
    print(f"Language pair: {language_pair}")

    aggregate_path = Path(__file__).parent / f"results_{utility_metric}_{wmt}_{language_pair}_{num_samples}samples_seed{seed_no}.jsonl"
    baseline_path = Path(
        __file__).parent / f"results_{utility_metric}_n_by_s_{wmt}_{language_pair}_{num_samples}samples_seed{seed_no}.jsonl"
    beam_path = Path(__file__).parent / f"results_beam4.jsonl"

    with jsonlines.open(aggregate_path) as f:
        aggregate_data = list(f)

    with jsonlines.open(baseline_path) as f:
        baseline_data = list(f)

    with jsonlines.open(beam_path) as f:
        beam_data = list(f)

    # Line example: {"testset": "wmt21", "language_pair": "de-en", "seed_no": 0, "method": "MBR with aggregate COMET (1
    # aggregates from 1024 refs)", "num_aggregates": 1, "num_samples": 1024, "chrf": 57.662579358040325, "cometinho":
    # 60.126039303373545, "comet22": 85.5144088923931, "duration": 489.24234914779663, "transl...

    # Filter data for testset and seed_no
    aggregate_data = [line for line in aggregate_data if line["testset"] == wmt and line["seed_no"] == seed_no]
    assert aggregate_data
    baseline_data = [line for line in baseline_data if line["testset"] == wmt and line["seed_no"] == seed_no]
    assert baseline_data
    beam_data = [line for line in beam_data if line["testset"] == wmt]  # no seed (deterministic)
    assert beam_data

    # Filter data for language pair
    aggregate_data = [line for line in aggregate_data if line["language_pair"] == language_pair]
    assert len(aggregate_data) == math.log2(num_samples) + 1
    baseline_data = [line for line in baseline_data if line["language_pair"] == language_pair]
    assert len(baseline_data) == math.log2(num_samples)
    beam_data = [line for line in beam_data if line["language_pair"] == language_pair]
    assert len(beam_data) == 1

    beam_result = beam_data[0][evaluation_metric]

    # X-axis is ordered from most to least effective references
    # Reference aggregation: num_aggregates = num_references, num_references/2, num_references/4, ..., 1
    # N-by-S: num_subsamples = num_references, num_references/2, num_references/4, ..., 1

    X = [int(num_references / 2 ** i) for i in range(int(math.log2(num_references) + 1))]

    # Reference aggregation
    # Format: (1,-0.4)(2,-0.6)(4,-0.5)(8,0.1)(16,0.1)(32,0.2)(64,0.1)(128,-0.0)(256,-0.0)
    aggregate_series: List[Tuple[int, float]] = []
    for x in X:
        aggregate_row = [line for line in aggregate_data if line["num_aggregates"] == x]
        assert len(aggregate_row) == 1
        assert aggregate_row[0]["num_samples"] == num_samples
        aggregate_result = aggregate_row[0][evaluation_metric]
        delta = aggregate_result - beam_result
        aggregate_series.append((x, delta))
    print("".join(f"({x},{delta:.5f})" for x, delta in aggregate_series))
    print()

    # N-by-S
    baseline_series = []
    for x in X:
        if x != X[-1]:
            baseline_row = [line for line in baseline_data if line["num_subsamples"] == x]
            assert len(baseline_row) == 1
            assert baseline_row[0]["num_samples"] == num_samples
            baseline_result = baseline_row[0][evaluation_metric]
            delta = baseline_result - beam_result
            baseline_series.append((x, delta))
        else:
            # Copy result from aggregate, which is identical
            baseline_series.append(aggregate_series[-1])
    print("".join(f"({x},{delta:.5f})" for x, delta in baseline_series))
    print()
