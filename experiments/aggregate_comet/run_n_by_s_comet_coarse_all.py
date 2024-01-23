import sys
from pathlib import Path

import evaluate
import jsonlines
import sacrebleu
from datasets import load_dataset

from utils import run_all_comet_n_by_s

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

split = sys.argv[2]
assert split in ["valid", "test"]

seed_no = int(sys.argv[3])

num_samples = 1024
epsilon_cutoff = 0.02
topn = 50

if split == "valid":
    wmt = "wmt21"
elif split == "test":
    wmt = "wmt22"
else:
    raise ValueError(split)

print(f"Using {split} split (={wmt}) with {num_samples} samples")

samples_dir = Path(__file__).parent / f"samples_{wmt}"
samples_name = f"transformer.wmt19.{language_pair}.single_model.1024samples.epsilon{epsilon_cutoff}.seed{seed_no}.jsonl"
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
unique_sample_counts = [len(set(row)) for row in samples]
print(f"Average number of unique samples: {sum(unique_sample_counts) / len(unique_sample_counts):.2f}")

results_file = jsonlines.open(Path(__file__).parent / f"results_comet22_n_by_s_coarse{topn}_{wmt}_{language_pair}_{num_samples}samples_seed{seed_no}.jsonl", "w")

comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")

src_path = sacrebleu.get_source_file(wmt, language_pair)
ref_path = sacrebleu.get_reference_files(wmt, language_pair)[0]
dataset = load_dataset("text", data_files={"test": src_path})
references = Path(ref_path).read_text().splitlines()
source_sequences = dataset["test"]["text"]
assert len(dataset["test"]) == len(references) == len(source_sequences)

# print("Only using 16 samples for testing")
# samples = samples[:16]
# source_sequences = source_sequences[:16]
# references = references[:16]

comet.scorer = comet.scorer.to("cuda:0").eval()
translation_lists, durations = run_all_comet_n_by_s(
    comet,
    samples=[[row[i] for row in samples] for i in range(len(samples[0]))],
    references=[[row[i] for row in samples] for i in range(len(samples[0]))],
    inputs=source_sequences,
    batch_size_embed=128,
    batch_size_estimate=128,
    return_top_n=topn,
)

for i, (translations, duration) in enumerate(zip(translation_lists, durations)):
    if not translations:
        continue

    results_file.write({
        "testset": wmt,
        "language_pair": language_pair,
        "seed_no": seed_no,
        "method": f"N-by-S MBR with COMET (S={int(2**i)} subsamples from {num_samples} refs)",
        "num_subsamples": int(2**i),
        "num_samples": num_samples,
        "duration": duration,
        "topn": translations,
    })

results_file.close()
