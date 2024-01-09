from pathlib import Path

import jsonlines

split = 'test'
# split = 'valid'

language_pairs = ["en-de", "de-en", "en-ru", "ru-en"]
metric = "comet22"

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
# – reference aggregation (factor 32) & tba & tba & tba & tba & tba / tba \\
# \midrule
# MBR with \comettt{} metric & & & & &  \\
# – pairwise & tba & tba & tba & tba & tba / tba \\
# – reference aggregation (factor 32) & tba & tba & tba & tba & tba / tba \\
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

sampling_chrf_results_path = Path(f"results_chrf_{testset}_1024samples_seed0.jsonl")

if sampling_chrf_results_path.exists():
    with jsonlines.open(sampling_chrf_results_path) as f:
        data = list(f)
    sampling_results = {result["language_pair"]: result[metric] for result in data if result["method"] == "Sampling"}
    print(f"Epsilon sampling (\epsilon=0.02) & {sampling_results['en-de']:.1f} & {sampling_results['de-en']:.1f} & {sampling_results['en-ru']:.1f} & {sampling_results['ru-en']:.1f} & tba / - \\\\")
    print(r"\midrule")
    print(r"MBR with \chrf{} metric & & & & &  \\")
    chrf_pairwise_results = {result["language_pair"]: result[metric] for result in data if result["method"] == "MBR with standard ChrF"}
    print(f"– pairwise & {chrf_pairwise_results['en-de']:.1f} & {chrf_pairwise_results['de-en']:.1f} & {chrf_pairwise_results['en-ru']:.1f} & {chrf_pairwise_results['ru-en']:.1f} & tba / tba \\\\")
    chrf_aggregate_results = {result["language_pair"]: result[metric] for result in data if result["method"] == "MBR with aggregate ChrF"}
    print(f"– reference aggregation (factor 1024) & {chrf_aggregate_results['en-de']:.1f} & {chrf_aggregate_results['de-en']:.1f} & {chrf_aggregate_results['en-ru']:.1f} & {chrf_aggregate_results['ru-en']:.1f} & tba / tba \\\\")
else:
    print(r"Epsilon sampling (\epsilon=0.02) & & & & & tba / - \\")
    print(r"\midrule")
    print(r"MBR with \chrf{} metric & & & & &  \\")
    print(r"– pairwise & & & & & tba / tba \\")
    print(r"– reference aggregation (factor 1024) & & & & & tba / tba \\\\")

print(r"\midrule")

print(r"MBR with \cometinho{} metric & & & & &  \\")
print(r"– pairwise & ", end="")
for lang_pair in language_pairs:
    path = Path(f"results_cometinho_{testset}_{lang_pair}_1024samples_seed0.jsonl")
    if path.exists() and path.stat().st_size > 0:
        with jsonlines.open(path) as f:
            data = list(f)
        results = {result["language_pair"]: result[metric] for result in data if result["testset"] == testset and result["num_aggregates"] == 1024}
        print(f"{results[lang_pair]:.1f} & ", end="")
    else:
        print("tba & ", end="")
print(r"tba / tba \\\\")
print(r"– reference aggregation (factor 32) & ", end="")
for lang_pair in language_pairs:
    path = Path(f"results_cometinho_{testset}_{lang_pair}_32samples_seed0.jsonl")
    if path.exists() and path.stat().st_size > 0:
        with jsonlines.open(path) as f:
            data = list(f)
        results = {result["language_pair"]: result[metric] for result in data if result["testset"] == testset and result["num_aggregates"] == 32}
        print(f"{results[lang_pair]:.1f} & ", end="")
    else:
        print("tba & ", end="")
print(r"tba / tba \\\\")

print(r"\midrule")

print(r"MBR with \comettt{} metric & & & & &  \\")
print(r"– pairwise & ", end="")
for lang_pair in language_pairs:
    path = Path(f"results_comet22_{testset}_{lang_pair}_1024samples_seed0.jsonl")
    if path.exists() and path.stat().st_size > 0:
        with jsonlines.open(path) as f:
            data = list(f)
        results = {result["language_pair"]: result[metric] for result in data if result["testset"] == testset and result["num_aggregates"] == 1024}
        print(f"{results[lang_pair]:.1f} & ", end="")
    else:
        print("tba & ", end="")
print(r"tba / tba \\\\")
print(r"– reference aggregation (factor 32) & ", end="")
for lang_pair in language_pairs:
    path = Path(f"results_comet22_{testset}_{lang_pair}_32samples_seed0.jsonl")
    if path.exists() and path.stat().st_size > 0:
        with jsonlines.open(path) as f:
            data = list(f)
        results = {result["language_pair"]: result[metric] for result in data if result["testset"] == testset and result["num_aggregates"] == 32}
        print(f"{results[lang_pair]:.1f} & ", end="")
    else:
        print("tba & ", end="")
print(r"tba / tba \\\\")

print(r"\bottomrule")
print(r"\end{tabularx}")
