import sys
from pathlib import Path

import jsonlines

jsonl_path = Path(sys.argv[1])

with jsonlines.open(jsonl_path) as f:
    data = [line for line in f]

baseline_path = Path("results_beam4.jsonl")

with jsonlines.open(baseline_path) as f:
    baseline_data = [line for line in f]
testset = data[0]["testset"]
language_pair = data[0]["language_pair"]
baseline = [line["comet22"] for line in baseline_data if testset == line["testset"] and language_pair == line["language_pair"]][0]

# Line example:
# {"testset": "wmt21", "language_pair": "de-en", "seed_no": 0, "method": "MBR with aggregate COMET (1 aggregates from 1024 refs)", "num_aggregates": 1, "num_samples": 1024, "chrf": 57.662579358040325, "cometinho": 60.126039303373545, "comet22": 85.5144088923931, "duration": 489.24234914779663, "transl...

print("Series 1 (COMET-22 delta):")
# Format: (1,-0.4)(2,-0.6)(4,-0.5)(8,0.1)(16,0.1)(32,0.2)(64,0.1)(128,-0.0)(256,-0.0)
num_aggregates = list(reversed(sorted(set(line["num_aggregates"] for line in data))))
num_segments = len(data[0]["translations"])
num_samples = max(num_aggregates)

for k in num_aggregates:
    row = [line for line in data if line["num_aggregates"] == k][0]
    delta = (row["comet22"] - baseline)
    print(f"({int(num_samples/k)},{delta:.5f})", end="")
print()

print("Series 2 (time per segment):")
for k in num_aggregates:
    row = [line for line in data if line["num_aggregates"] == k][0]
    duration = row["duration"]
    print(f"({int(num_samples/k)},{duration / num_segments:.5f})", end="")
print()
