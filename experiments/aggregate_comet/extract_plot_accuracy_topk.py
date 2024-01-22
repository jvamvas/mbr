import math
import sys
from pathlib import Path
from typing import List, Tuple

import jsonlines
import numpy as np

utility_metric = sys.argv[1]
split = sys.argv[2]

TOPK = 1

if split == "valid":
    wmt = "wmt21"
elif split == "test":
    wmt = "wmt22"
else:
    raise ValueError(split)

language_pairs = ["en-de", "de-en", "en-ru", "ru-en"]
seed_no = 0
num_samples = 1024
num_references = num_samples

for language_pair in language_pairs:
    print(f"Language pair: {language_pair}")

    aggregate_path = Path(__file__).parent / f"results_{utility_metric}_coarse50_{wmt}_{language_pair}_{num_samples}samples_seed{seed_no}.jsonl"
    n_by_s_path = Path(
        __file__).parent / f"results_{utility_metric}_n_by_s_coarse50_{wmt}_{language_pair}_{num_samples}samples_seed{seed_no}.jsonl"

    with jsonlines.open(aggregate_path) as f:
        aggregate_data = list(f)

    with jsonlines.open(n_by_s_path) as f:
        n_by_s_data = list(f)

    # Line example: {"testset": "wmt21", "language_pair": "de-en", "seed_no": 0, "method": "MBR with aggregate COMET (1
    # aggregates from 1024 refs)", "num_aggregates": 1, "num_samples": 1024, "chrf": 57.662579358040325, "cometinho":
    # 60.126039303373545, "comet22": 85.5144088923931, "duration": 489.24234914779663, "transl...

    # Filter data for testset and seed_no
    aggregate_data = [line for line in aggregate_data if line["testset"] == wmt and line["seed_no"] == seed_no]
    assert aggregate_data
    n_by_s_data = [line for line in n_by_s_data if line["testset"] == wmt and line["seed_no"] == seed_no]
    assert n_by_s_data
    
    # Filter data for language pair
    aggregate_data = [line for line in aggregate_data if line["language_pair"] == language_pair]
    assert len(aggregate_data) == math.log2(num_samples) + 1
    n_by_s_data = [line for line in n_by_s_data if line["language_pair"] == language_pair]
    assert len(n_by_s_data) == math.log2(num_samples)

    n_by_n_data = [line for line in aggregate_data if line["num_aggregates"] == num_references]
    assert len(n_by_n_data) == 1
    n_by_n_translations = [translations[0].strip() for translations in n_by_n_data[0]["topn"]]

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
        aggregate_translations = aggregate_row[0]["translations"]
        accuracy = np.mean([1 if b in {a_.strip() for a_ in a} else 0 for a, b in zip(aggregate_translations, n_by_n_translations)])
        aggregate_series.append((x, accuracy))
    # To make plotting easier, we will invert the ticks, so 1024 corresponds to 1, 512 to 2, 256 to 4, etc.
    inverted_aggregate_series = [(num_references / x, accuracy) for x, accuracy in aggregate_series]
    print("".join(f"({x},{accuracy:.5f})" for x, accuracy in inverted_aggregate_series))
    print()

    # N-by-S
    n_by_s_series = []
    for x in X:
        if x == X[0]:
            # Copy result from aggregate, which is identical
            n_by_s_series.append(aggregate_series[0])
        else:
            n_by_s_row = [line for line in n_by_s_data if line["num_subsamples"] == x]
            assert len(n_by_s_row) == 1
            assert n_by_s_row[0]["num_samples"] == num_samples
            n_by_s_translations = n_by_s_row[0]["translations"]
            accuracy = np.mean([1 if b in {a_.strip() for a_ in a} else 0 for a, b in zip(n_by_s_translations, n_by_n_translations)])
            n_by_s_series.append((x, accuracy))
    inverted_n_by_s_series = [(num_references / x, accuracy) for x, accuracy in n_by_s_series]
    print("".join(f"({x},{accuracy:.5f})" for x, accuracy in inverted_n_by_s_series))
    print()
