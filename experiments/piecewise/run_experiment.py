import sys
import time
from copy import deepcopy
from pathlib import Path

import evaluate
import jsonlines
import sacrebleu
from datasets import load_dataset
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, AutoTokenizer, pipeline, set_seed, GenerationConfig

from mbr import MBRConfig
from mbr.modeling import PiecewiseMBR, MBR

language_pair = sys.argv[1]

batch_size = 16

results_file = jsonlines.open(Path(__file__).parent / f"results_{language_pair}.jsonl", "w", flush=True)

model_name = f"facebook/m2m100_1.2B"
mbr_model = MBR(M2M100ForConditionalGeneration).from_pretrained(model_name).to(0)
tokenizer = AutoTokenizer.from_pretrained(model_name)
mt_pipeline = pipeline(
    "translation_" + language_pair.split("-")[0] + "_to_" + language_pair.split("-")[1],
    model=mbr_model,
    tokenizer=tokenizer,
    device=0,
)
evaluation_metric_chrf = evaluate.load("chrf")
evaluation_metric_comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")

src_path = sacrebleu.get_source_file("wmt21", language_pair)
ref_path = sacrebleu.get_reference_files("wmt21", language_pair)[0]
dataset = load_dataset("text", data_files={"test": src_path})
references = Path(ref_path).read_text().splitlines()
assert len(dataset["test"]) == len(references)

# # Testing: Restrict to 64 examples
# dataset["test"] = dataset["test"].select(range(1))
# references = references[:1]

# MBR Baseline
print("MBR Baseline", flush=True)
generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.do_sample = True
generation_config.num_beams = 1
generation_config.early_stopping = False
generation_config.epsilon_cutoff = 0.02

base_mbr_config = MBRConfig(
    num_samples=256,
    num_references=256,
    metric="fastchrf",
)
mbr_configs = {"mbr_baseline": base_mbr_config}

for method, mbr_config in mbr_configs.items():
    set_seed(42)
    time_start = time.time()
    outputs = mt_pipeline(
        dataset["test"]["text"],
        mbr_config=mbr_config,
        generation_config=generation_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        progress_bar=True
    )
    translations = []
    for batch in tqdm(outputs):
        if isinstance(batch, dict):
            batch = [batch]
        translations += [translation["translation_text"] for translation in batch]
    time_end = time.time()

    chrf_score = evaluation_metric_chrf.compute(
        predictions=translations,
        references=references,
    )
    comet_score = evaluation_metric_comet.compute(
        predictions=translations,
        references=references,
        sources=dataset["test"]["text"],
        # gpus=0,
    )
    results_file.write({
        "language_pair": language_pair,
        "method": method,
        "chrf": chrf_score["score"],
        "comet22": comet_score["mean_score"],
        "duration": time_end - time_start,
        "translations": translations,
    })


# Piecewise MBR

print("Piecewise MBR", flush=True)
del mbr_model
piecewise_mbr_model = PiecewiseMBR(M2M100ForConditionalGeneration).from_pretrained(model_name)
mt_pipeline.model = piecewise_mbr_model.to(mt_pipeline.device)

piecewise_mbr_configs = {}
# for piece_length in [1, 2, 4, 8, 16]:
for piece_length in [8]:
    mbr_config = deepcopy(base_mbr_config)
    mbr_config.piecewise = True
    mbr_config.piece_length = piece_length
    piecewise_mbr_configs[f"mbr_piecewise_{piece_length}"] = mbr_config

for method, mbr_config in piecewise_mbr_configs.items():
    print(method, flush=True)
    set_seed(42)
    time_start = time.time()
    outputs = mt_pipeline(
        dataset["test"]["text"],
        mbr_config=mbr_config,
        piece_length=mbr_config.piece_length,
        generation_config=generation_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        progress_bar=True,
        verbose=True,
    )
    translations = []
    for batch in tqdm(outputs):
        if isinstance(batch, dict):
            batch = [batch]
        translations += [translation["translation_text"] for translation in batch]
    time_end = time.time()

    chrf_score = evaluation_metric_chrf.compute(
        predictions=translations,
        references=references,
    )
    comet_score = evaluation_metric_comet.compute(
        predictions=translations,
        references=references,
        sources=dataset["test"]["text"],
        # gpus=0,
    )
    results_file.write({
        "language_pair": language_pair,
        "method": method,
        "chrf": chrf_score["score"],
        "comet22": comet_score["mean_score"],
        "duration": time_end - time_start,
        "translations": translations,
    })

del piecewise_mbr_model

# Beam search
print("Beam search", flush=True)
model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(mt_pipeline.device)
mt_pipeline.model = model
generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.num_beams = 4

set_seed(42)
time_start = time.time()
outputs = mt_pipeline(
    dataset["test"]["text"],
    generation_config=generation_config,
    batch_size=batch_size,
)
translations = []
for batch in tqdm(outputs):
    if isinstance(batch, dict):
        batch = [batch]
    translations += [translation["translation_text"] for translation in batch]
time_end = time.time()

chrf_score = evaluation_metric_chrf.compute(
    predictions=translations,
    references=references,
)
comet_score = evaluation_metric_comet.compute(
    predictions=translations,
    references=references,
    sources=dataset["test"]["text"],
)
results_file.write({
    "language_pair": language_pair,
    "method": f"beam search (beam size {generation_config.num_beams})",
    "chrf": chrf_score["score"],
    "comet22": comet_score["mean_score"],
    "duration": time_end - time_start,
    "translations": translations,
})

results_file.close()
