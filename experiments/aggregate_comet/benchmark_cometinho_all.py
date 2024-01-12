import math
import random
import sys
from pathlib import Path

import evaluate
import jsonlines
import sacrebleu
from datasets import load_dataset

from utils import run_all_comet_factors

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

split = "test"

num_samples = 1024
epsilon_cutoff = 0.02

num_segments = 16

if split == "valid":
    wmt = "wmt21"
elif split == "test":
    wmt = "wmt22"
else:
    raise ValueError(split)


random.seed(42)
random_indices = random.sample(range(500), num_segments)

cometinho = evaluate.load("comet", "eamt22-cometinho-da")
cometinho.scorer = cometinho.scorer.to("cuda:0")

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
translation_lists, durations = run_all_comet_factors(
    cometinho,
    samples=[[row[i] for row in samples] for i in range(len(samples[0]))],
    references=[[row[i] for row in samples] for i in range(len(samples[0]))],
    inputs=source_sequences,
    batch_size_embed=8192,
    batch_size_estimate=8192,
)

print("Total:")
for i, aggregation_factor in enumerate(reversed([2**i for i in range(len(durations))])):
    print(f"Aggregation factor {aggregation_factor}: {durations[i]:.2f}s")

print("Average per segment:")
for i, aggregation_factor in enumerate(reversed([2**i for i in range(len(durations))])):
    print(f"Aggregation factor {aggregation_factor}: {durations[i] / num_segments:.2f}s")
