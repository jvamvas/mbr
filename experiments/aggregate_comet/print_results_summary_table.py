from collections import defaultdict
from pathlib import Path

import jsonlines

split = 'test'
# split = 'valid'

language_pairs = ["en-de", "de-en", "en-ru", "ru-en"]
metric = "comet22"
factor = 128

seed_nos = [0]  # Average results over multiple seeds
num_seeds = len(seed_nos)

# \begin{tabularx}{\textwidth}{Xrrrrr}
# \toprule
# & \textsc{en--de} & \textsc{de--en} & \textsc{en--ru} & \textsc{ru--en} & Time (total / utility) \\
# \midrule
# Beam search (size 4) & 85.9 & 83.9 & 82.9 & 84.3 & tba / - \\
# Epsilon sampling (\epsilon=0.02) & 83.4 & 81.6 & 80.7 & 81.7 & tba / - \\
# \midrule
# MBR with \chrf{} metric & & & & &  \\
# – pairwise & 85.8 & 84.1 & 83.5 & 84.4 & tba / tba \\
# – reference aggregation (factor 1024) & 85.9 & 84.1 & 83.5 & 84.4 & tba / tba \\
# \midrule
# MBR with \cometinho{} metric & & & & &  \\
# – pairwise & tba & tba & tba & tba & tba / tba \\
# – reference aggregation (factor 128) & tba & tba & tba & tba & tba / tba \\
# \midrule
# MBR with \comettt{} metric & & & & &  \\
# – pairwise & tba & tba & tba & tba & tba / tba \\
# – reference aggregation (factor 128) & tba & tba & tba & tba & tba / tba \\
# \bottomrule
# \end{tabularx}

print(r"""\begin{tabularx}{\textwidth}{Xrrrrr}
\toprule
& \textsc{en--de} & \textsc{de--en} & \textsc{en--ru} & \textsc{ru--en} & Time (total / utility) \\
\midrule""")

if split == "test":
    testset = "wmt22"
else:
    testset = "wmt21"

beam_results_path = Path(f"results_beam4.jsonl")

if beam_results_path.exists():
    with jsonlines.open(beam_results_path) as f:
        data = list(f)
    results = {result["language_pair"]: result[metric] for result in data if result["testset"] == testset}
    print(f"Beam search (size 4) & {results['en-de']:.1f} & {results['de-en']:.1f} & {results['en-ru']:.1f} & {results['ru-en']:.1f} & tba / - \\\\")

sampling_chrf_results_paths = [
    Path(f"results_chrf_{testset}_1024samples_seed{seed_no}.jsonl")
    for seed_no in seed_nos
]
chrf_data = []
for path in sampling_chrf_results_paths:
    with jsonlines.open(path) as f:
        chrf_data.append(list(f))

    sampling_results = defaultdict(list)
    for i, seed_no in enumerate(seed_nos):
        for result in chrf_data[i]:
            if result["method"] == "Sampling":
                sampling_results[result["language_pair"]].append(result[metric])
    print(f"Epsilon sampling (\epsilon=0.02) & {sum(sampling_results['en-de']) / num_seeds:.1f} & {sum(sampling_results['de-en']) / num_seeds:.1f} & {sum(sampling_results['en-ru']) / num_seeds:.1f} & {sum(sampling_results['ru-en']) / num_seeds:.1f} & tba / - \\\\")
    print(r"\midrule")

    print(r"MBR with \chrf{} metric & & & & &  \\")
    chrf_pairwise_results = defaultdict(list)
    for i, seed_no in enumerate(seed_nos):
        for result in chrf_data[i]:
            if result["method"] == "MBR with standard ChrF":
                chrf_pairwise_results[result["language_pair"]].append(result[metric])

    print(f"– pairwise & {sum(chrf_pairwise_results['en-de']) / num_seeds:.1f} & {sum(chrf_pairwise_results['de-en']) / num_seeds:.1f} & {sum(chrf_pairwise_results['en-ru']) / num_seeds:.1f} & {sum(chrf_pairwise_results['ru-en']) / num_seeds:.1f} & tba / tba \\\\")

    chrf_aggregate_results = defaultdict(list)
    for i, seed_no in enumerate(seed_nos):
        for result in chrf_data[i]:
            if result["method"] == "MBR with aggregate ChrF":
                chrf_aggregate_results[result["language_pair"]].append(result[metric])
    print(f"– reference aggregation (factor {factor}) & {sum(chrf_aggregate_results['en-de']) / num_seeds:.1f} & {sum(chrf_aggregate_results['de-en']) / num_seeds:.1f} & {sum(chrf_aggregate_results['en-ru']) / num_seeds:.1f} & {sum(chrf_aggregate_results['ru-en']) / num_seeds:.1f} & tba / tba \\\\")

print(r"\midrule")

print(r"MBR with \cometinho{} metric & & & & &  \\")
print(r"– pairwise & ", end="")
for lang_pair in language_pairs:
    paths = [
        Path(f"results_cometinho_{testset}_{lang_pair}_1024samples_seed{seed_no}.jsonl")
        for seed_no in seed_nos
    ]
    data = []
    for path in paths:
        with jsonlines.open(path) as f:
            data.append(list(f))

    results = defaultdict(list)
    for i, seed_no in enumerate(seed_nos):
        for result in data[i]:
            if result["num_aggregates"] == 1024 and result["testset"] == testset:
                results[result["language_pair"]].append(result[metric])
    print(f"{sum(results[lang_pair]) / num_seeds:.1f} & ", end="")
print(r"tba / tba \\")

print(rf"– reference aggregation (factor {factor}) & ", end="")
for lang_pair in language_pairs:
    paths = [
        Path(f"results_cometinho_{testset}_{lang_pair}_1024samples_seed{seed_no}.jsonl")
        for seed_no in seed_nos
    ]
    data = []
    for path in paths:
        with jsonlines.open(path) as f:
            data.append(list(f))

    results = defaultdict(list)
    for i, seed_no in enumerate(seed_nos):
        for result in data[i]:
            if result["num_aggregates"] == 1024 // factor and result["testset"] == testset:
                results[result["language_pair"]].append(result[metric])
    print(f"{sum(results[lang_pair]) / num_seeds:.1f} & ", end="")
print(r"tba / tba \\")

print(r"\midrule")

print(r"MBR with \comettt{} metric & & & & &  \\")
for lang_pair in language_pairs:
    paths = [
        Path(f"results_comet22_{testset}_{lang_pair}_1024samples_seed{seed_no}.jsonl")
        for seed_no in seed_nos
    ]
    data = []
    for path in paths:
        with jsonlines.open(path) as f:
            data.append(list(f))

    results = defaultdict(list)
    for i, seed_no in enumerate(seed_nos):
        for result in data[i]:
            if result["num_aggregates"] == 1024 and result["testset"] == testset:
                results[result["language_pair"]].append(result[metric])
    print(f"{sum(results[lang_pair]) / num_seeds:.1f} & ", end="")
print(r"tba / tba \\")

print(rf"– reference aggregation (factor {factor}) & ", end="")
for lang_pair in language_pairs:
    paths = [
        Path(f"results_comet22_{testset}_{lang_pair}_1024samples_seed{seed_no}.jsonl")
        for seed_no in seed_nos
    ]
    data = []
    for path in paths:
        with jsonlines.open(path) as f:
            data.append(list(f))

    results = defaultdict(list)
    for i, seed_no in enumerate(seed_nos):
        for result in data[i]:
            if result["num_aggregates"] == 1024 // factor and result["testset"] == testset:
                results[result["language_pair"]].append(result[metric])
    print(f"{sum(results[lang_pair]) / num_seeds:.1f} & ", end="")
print(r"tba / tba \\")

print(r"\bottomrule")
print(r"\end{tabularx}")
