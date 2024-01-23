from pathlib import Path

direct_path = Path("results_comet22_wmt21_de-en_1024samples_seed0.jsonl")
assert direct_path.exists()

coarse_path = Path("results_comet22_coarse50_wmt21_de-en_1024samples_seed0.jsonl")
assert coarse_path.exists()

fine_path = Path("results_comet22_fine_from_top20_comet22_wmt21_de-en_1024samples_seed0.jsonl")
assert fine_path.exists()

with open(direct_path) as f:
    direct_data = list(f)

with open(coarse_path) as f:
    coarse_data = list(f)

with open(fine_path) as f:
    fine_data = list(f)

print("Top-1 of coarse should be identical to output of direct")
# This should be true for all aggregation levels
for coarse_line, direct_line in zip(coarse_data, direct_data):
    assert coarse_line["num_aggregates"] == direct_line["num_aggregates"]
    num_identical = 0
    for coarse_translation, direct_translation in zip(coarse_line["topn"][0], direct_line["translations"]):
        if coarse_translation == direct_translation:
            num_identical += 1
    print(f"Match rate for {coarse_line['num_aggregates']} aggregates: {num_identical / len(coarse_line['topn'][0])}")

print("Output of pairwise coarse-to-fine should be identical to output of pairwise direct")
fine_line = fine_data[-1]
assert fine_line["num_aggregates"] == 1024
direct_line = direct_data[-1]
assert direct_line["num_aggregates"] == 1024

num_identical = 0
for fine_translation, direct_translation in zip(fine_line["translations"], direct_line["translations"]):
    if fine_translation == direct_translation:
        num_identical += 1
print(f"Match rate for {fine_line['num_aggregates']} aggregates: {num_identical / len(fine_line['translations'])}")
