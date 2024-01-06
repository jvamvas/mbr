import sys
import time
from pathlib import Path
from typing import List, Callable

import evaluate
import jsonlines
import sacrebleu
from datasets import load_dataset

from utils import mbr_standard_comet, mbr_aggregate_comet

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

num_samples = 256
seed = 42
epsilon_cutoff = 0.02

samples_name = f"transformer.wmt19.{language_pair}.single_model.{num_samples}samples.epsilon{epsilon_cutoff}.seed{seed}.jsonl"
samples_path = Path(__file__).parent / "samples" / samples_name
assert samples_path.exists(), samples_path

samples = []  # segments x num_samples
with jsonlines.open(samples_path) as f:
    for line in f:
        samples.append(line["samples"][0])

results_file = jsonlines.open(Path(__file__).parent / f"results_cometinho_{language_pair}.jsonl", "w")

chrf = evaluate.load("chrf")
cometinho = evaluate.load("comet", "eamt22-cometinho-da")
comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")

src_path = sacrebleu.get_source_file("wmt19", language_pair)
ref_path = sacrebleu.get_reference_files("wmt19", language_pair)[0]
dataset = load_dataset("text", data_files={"test": src_path})
references = Path(ref_path).read_text().splitlines()
source_sequences = dataset["test"]["text"]
assert len(dataset["test"]) == len(references) == len(source_sequences)

# print("Only using 20 samples for testing")
# samples = samples[:20]
# source_sequences = source_sequences[:20]
# references = references[:20]

def do_evaluate(method: str, select_func: Callable[[List[str]], List[str]]):
    time_start = time.time()
    translations = select_func(samples, source_sequences)
    assert len(translations) == len(source_sequences)
    time_end = time.time()  # TODO use timeit instead

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
        "method": method,
        "chrf": chrf_score["score"],
        "cometinho": cometinho_score["mean_score"],
        "comet22": comet22_score["mean_score"],
        "duration": time_end - time_start,
    })


# Random selection (= sampling)
print("Random selection")
select_func = lambda samples, source_sequences: [row[0] for row in samples]
do_evaluate("Sampling", select_func)

# MBR with standard COMET
cometinho.scorer = cometinho.scorer.to("cuda:0")
select_func = lambda samples, source_sequences: mbr_standard_comet(
    cometinho,
    samples=[[row[i] for row in samples] for i in range(len(samples[0]))],
    references=[[row[i] for row in samples] for i in range(len(samples[0]))],
    inputs=source_sequences,
    batch_size_embed=128,
    batch_size_estimate=128,
)
do_evaluate("MBR with standard COMET", select_func)


# MBR with aggregate COMET
cometinho.scorer = cometinho.scorer.to("cuda:0")
select_func = lambda samples, source_sequences: mbr_aggregate_comet(
    cometinho,
    samples=[[row[i] for row in samples] for i in range(len(samples[0]))],
    references=[[row[i] for row in samples] for i in range(len(samples[0]))],
    inputs=source_sequences,
    batch_size_embed=128,
    batch_size_estimate=128,
)
do_evaluate("MBR with aggregate COMET", select_func)

results_file.close()
