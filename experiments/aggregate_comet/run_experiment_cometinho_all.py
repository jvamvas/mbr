import sys
from pathlib import Path

import evaluate
import jsonlines
import sacrebleu
from datasets import load_dataset

from utils import run_all_comet_variants

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]
seed_no = int(sys.argv[2])

num_samples = 256
epsilon_cutoff = 0.02

samples_name = f"transformer.wmt19.{language_pair}.single_model.1024samples.epsilon{epsilon_cutoff}.seed{seed_no}.jsonl"
samples_path = Path(__file__).parent / "samples" / samples_name
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


results_file = jsonlines.open(Path(__file__).parent / f"results_cometinho_{language_pair}_{num_samples}samples_seed{seed_no}.jsonl", "w")

chrf = evaluate.load("chrf")
cometinho = evaluate.load("comet", "eamt22-cometinho-da")
comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")

src_path = sacrebleu.get_source_file("wmt19", language_pair)
ref_path = sacrebleu.get_reference_files("wmt19", language_pair)[0]
dataset = load_dataset("text", data_files={"test": src_path})
references = Path(ref_path).read_text().splitlines()
source_sequences = dataset["test"]["text"]
assert len(dataset["test"]) == len(references) == len(source_sequences)

print("Only using 16 samples for testing")
samples = samples[:16]
source_sequences = source_sequences[:16]
references = references[:16]

cometinho.scorer = cometinho.scorer.to("cuda:0")
translation_lists, durations = run_all_comet_variants(
    cometinho,
    samples=[[row[i] for row in samples] for i in range(len(samples[0]))],
    references=[[row[i] for row in samples] for i in range(len(samples[0]))],
    inputs=source_sequences,
    batch_size_embed=256,
    batch_size_estimate=256,
)

for i, (translations, duration) in enumerate(zip(translation_lists, durations)):
    chrf_score = chrf.compute(
        predictions=translations,
        references=references,
    )
    cometinho_score = cometinho.compute(
        predictions=translations,
        references=references,
        sources=source_sequences,
        # gpus=0,
    )
    comet22_score = comet.compute(
        predictions=translations,
        references=references,
        sources=source_sequences,
        # gpus=0,
    )
    results_file.write({
        "language_pair": language_pair,
        "method": f"MBR with aggregate COMET ({int(2**i)}/{len(samples)} partition",
        "chrf": chrf_score["score"],
        "cometinho": 100 * cometinho_score["mean_score"],
        "comet22": 100 * comet22_score["mean_score"],
        "duration": duration,
    })

results_file.close()