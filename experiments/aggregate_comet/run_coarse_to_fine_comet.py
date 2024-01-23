import sys
from pathlib import Path

import evaluate
import jsonlines
import sacrebleu
from datasets import load_dataset

from utils import mbr_standard_comet

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

split = sys.argv[2]
assert split in ["valid", "test"]

seed_no = int(sys.argv[3])

coarse_metric = sys.argv[4]
assert coarse_metric in ["chrf", "cometinho", "comet22"]

num_samples = 1024
num_coarse = 20
coarse_nums_aggregates = [1, 1024]
epsilon_cutoff = 0.02

if split == "valid":
    wmt = "wmt21"
elif split == "test":
    wmt = "wmt22"
else:
    raise ValueError(split)

print(f"Using {split} split (={wmt})")

samples_dir = Path(__file__).parent / f"samples_{wmt}"
samples_name = f"transformer.wmt19.{language_pair}.single_model.1024samples.epsilon{epsilon_cutoff}.seed{seed_no}.jsonl"
samples_path = samples_dir / samples_name
assert samples_path.exists(), samples_path

samples_references = []  # segments x num_samples
with jsonlines.open(samples_path) as f:
    for line in f:
        samples_references.append(line["samples"])

# Cut to desired number of samples
samples_references = [row[:num_samples] for row in samples_references]
for row in samples_references:
    assert len(row) == num_samples
unique_sample_counts = [len(set(row)) for row in samples_references]
print(f"Average number of unique samples: {sum(unique_sample_counts) / len(unique_sample_counts):.2f}")

coarse_path = Path(__file__).parent / f"results_{coarse_metric}_coarse50_{wmt}_{language_pair}_{num_samples}samples_seed{seed_no}.jsonl"
assert coarse_path.exists(), coarse_path

chrf = evaluate.load("chrf")
cometinho = evaluate.load("comet", "eamt22-cometinho-da")
comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")

with jsonlines.open(coarse_path) as f:
    all_coarse_lines = list(f)

results_file = jsonlines.open(Path(__file__).parent / f"results_comet22_fine_from_top{num_coarse}_{coarse_metric}_{wmt}_{language_pair}_{num_samples}samples_seed{seed_no}.jsonl", "w")

for coarse_num_aggregates in coarse_nums_aggregates:
    coarse_lines = [line for line in all_coarse_lines if line["num_aggregates"] == coarse_num_aggregates]
    assert len(coarse_lines) == 1
    samples_hypotheses = [row[:num_coarse] for row in coarse_lines[0]["topn"]]  # segments x num_coarse

    unique_sample_counts = [len(set(row)) for row in samples_hypotheses]
    print(f"Average number of unique top-n samples: {sum(unique_sample_counts) / len(unique_sample_counts):.2f}")

    src_path = sacrebleu.get_source_file(wmt, language_pair)
    ref_path = sacrebleu.get_reference_files(wmt, language_pair)[0]
    dataset = load_dataset("text", data_files={"test": src_path})
    references = Path(ref_path).read_text().splitlines()
    source_sequences = dataset["test"]["text"]
    assert len(dataset["test"]) == len(references) == len(source_sequences)

    # print("Only using 16 samples for testing")
    # samples_hypotheses = samples_hypotheses[:16]
    # samples_references = samples_references[:16]
    # source_sequences = source_sequences[:16]
    # references = references[:16]

    comet.scorer = comet.scorer.to("cuda:0").eval()
    translations = mbr_standard_comet(
        comet,
        samples=[[row[i] for row in samples_hypotheses] for i in range(len(samples_hypotheses[0]))],
        references=[[row[i] for row in samples_references] for i in range(len(samples_references[0]))],
        inputs=source_sequences,
        batch_size_embed=128,
        batch_size_estimate=128,
    )

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
        "testset": wmt,
        "language_pair": language_pair,
        "seed_no": seed_no,
        "method": f"Coarse-to-fine; coarse={coarse_metric}, top {num_coarse}, fine=COMET22",
        "num_coarse": num_coarse,
        "coarse_metric": coarse_metric,
        "coarse_num_aggregates": coarse_num_aggregates,
        "chrf": chrf_score["score"],
        "cometinho": 100 * cometinho_score["mean_score"],
        "comet22": 100 * comet22_score["mean_score"],
        "translations": translations,
    })

results_file.close()
