from pathlib import Path

import jsonlines

print("Top-1 of coarse should be identical to output of direct")
for metric in ["chrf", "cometinho", "comet22"]:
    print(f"Metric: {metric}")
    direct_path = Path(f"results_{metric}_wmt21_de-en_1024samples_seed0.jsonl")
    assert direct_path.exists()

    coarse_path = Path(f"results_{metric}_coarse50_wmt21_de-en_1024samples_seed0.jsonl")
    assert coarse_path.exists()

    with jsonlines.open(direct_path) as f:
        direct_data = list(f)

    with jsonlines.open(coarse_path) as f:
        coarse_data = list(f)

    for coarse_line, direct_line in zip(coarse_data, direct_data):
        assert coarse_line["num_aggregates"] == direct_line["num_aggregates"]
        num_identical = 0
        for coarse_translations, direct_translation in zip(coarse_line["topn"], direct_line["translations"]):
            if coarse_translations[0] == direct_translation:
                num_identical += 1
        print(f"Match rate for {coarse_line['num_aggregates']} aggregates: {num_identical / len(direct_line['translations'])}")

print("Output of pairwise coarse-to-fine should be identical to output of pairwise direct")

direct_path = Path("results_comet22_wmt21_de-en_1024samples_seed0.jsonl")
assert direct_path.exists()

coarse_path = Path("results_comet22_coarse50_wmt21_de-en_1024samples_seed0.jsonl")
assert coarse_path.exists()

fine_path = Path("results_comet22_fine_from_top20_comet22_wmt21_de-en_1024samples_seed0.jsonl")
assert fine_path.exists()

with jsonlines.open(direct_path) as f:
    direct_data = list(f)

with jsonlines.open(coarse_path) as f:
    coarse_data = list(f)

with jsonlines.open(fine_path) as f:
    fine_data = list(f)

fine_line = fine_data[-1]
assert fine_line["coarse_num_aggregates"] == 1024
direct_line = direct_data[-1]
assert direct_line["num_aggregates"] == 1024

num_identical = 0
for fine_translation, direct_translation in zip(fine_line["translations"], direct_line["translations"]):
    if fine_translation == direct_translation:
        num_identical += 1
print(f"Match rate for pairwise: {num_identical / len(fine_line['translations'])}")
