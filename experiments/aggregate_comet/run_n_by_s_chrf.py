import sys
import time
from pathlib import Path
from typing import List, Callable

import evaluate
import jsonlines
import sacrebleu
from datasets import load_dataset

from utils import mbr_standard_chrf, mbr_aggregate_chrf

split = sys.argv[1]
assert split in ["valid", "test"]

seed_no = int(sys.argv[2])

num_samples = 1024  # N
num_references = 128  # S; num_samples / num_references corresponds to aggregation factor
epsilon_cutoff = 0.02

if split == "valid":
    wmt = "wmt21"
elif split == "test":
    wmt = "wmt22"
else:
    raise ValueError(split)

results_file = jsonlines.open(Path(__file__).parent / f"results_{num_samples}n_by_{num_references}s_chrf_{wmt}_seed{seed_no}.jsonl", "w")

chrf = evaluate.load("chrf")
cometinho = evaluate.load("comet", "eamt22-cometinho-da")
comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")

for language_pair in ["de-en", "en-de", "en-ru", "ru-en"]:
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

    def do_evaluate(method: str, select_func: Callable[[List[str]], List[str]]):
        time_start = time.time()
        translations = select_func(samples)
        assert len(translations) == len(source_sequences)
        time_end = time.time()

        chrf_score = chrf.compute(
            predictions=translations,
            references=references,
        )
        cometinho_score = cometinho.compute(
            predictions=translations,
            references=references,
            sources=source_sequences,
            gpus=0,
        )
        comet22_score = comet.compute(
            predictions=translations,
            references=references,
            sources=source_sequences,
            gpus=0,
        )
        results_file.write({
            "language_pair": language_pair,
            "method": method,
            "chrf": chrf_score["score"],
            "cometinho": 100 * cometinho_score["mean_score"],
            "comet22": 100 * comet22_score["mean_score"],
            "duration": time_end - time_start,
        })


    # Random selection (= sampling)
    print("Random selection")
    select_func = lambda samples: [row[0] for row in samples]
    do_evaluate("Sampling", select_func)

    select_func = lambda samples: mbr_standard_chrf(
        samples=samples,
        references=[row[:num_references] for row in samples],
    )
    do_evaluate("MBR with standard ChrF", select_func)

    select_func = lambda samples: mbr_aggregate_chrf(
        samples=samples,
        references=[row[:num_references] for row in samples],
    )
    do_evaluate("MBR with aggregate ChrF", select_func)

results_file.close()
