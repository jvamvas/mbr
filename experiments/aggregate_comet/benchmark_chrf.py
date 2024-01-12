import random
import sys
import time
from pathlib import Path

import jsonlines
import sacrebleu
from datasets import load_dataset

from utils import mbr_standard_chrf, mbr_aggregate_chrf

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

split = "test"

num_samples = 1024
epsilon_cutoff = 0.02

num_segments = 512

if split == "valid":
    wmt = "wmt21"
elif split == "test":
    wmt = "wmt22"
else:
    raise ValueError(split)


random.seed(42)
random_indices = random.sample(range(500), num_segments)

samples_dir = Path(__file__).parent / f"samples_{wmt}"
samples_name = f"transformer.wmt19.{language_pair}.single_model.1024samples.epsilon{epsilon_cutoff}.seed0.jsonl"
samples_path = samples_dir / samples_name
assert samples_path.exists(), samples_path

samples = []  # segments x num_samples
with jsonlines.open(samples_path) as f:
    for line in f:
        samples.append(line["samples"])

# Cut to desired number of samples
samples = [row[:num_samples] for row in samples]
for row in samples:
    assert len(row) == num_samples

src_path = sacrebleu.get_source_file(wmt, language_pair)
ref_path = sacrebleu.get_reference_files(wmt, language_pair)[0]
dataset = load_dataset("text", data_files={"test": src_path})
references = Path(ref_path).read_text().splitlines()
source_sequences = dataset["test"]["text"]
assert len(dataset["test"]) == len(references) == len(source_sequences)

samples = [samples[i] for i in random_indices]
source_sequences = [source_sequences[i] for i in random_indices]
references = [references[i] for i in random_indices]

# pairwise chrf
time_start = time.time()
translations = mbr_standard_chrf(
    samples=samples,
    references=references,
)
time_end = time.time()
time_delta = time_end - time_start

print(f"Pairwise CHRF:")
print("Total time:", time_delta)
print("Time per segment:", time_delta / num_segments)

# aggregate chrf
time_start = time.time()
translations = mbr_aggregate_chrf(
    samples=samples,
    references=references,
)
time_end = time.time()
time_delta = time_end - time_start

print(f"Aggregate CHRF:")
print("Total time:", time_delta)
print("Time per segment:", time_delta / num_segments)
