import time
from copy import deepcopy
from pathlib import Path

import evaluate
import jsonlines
import sacrebleu
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import FSMTForConditionalGeneration, AutoTokenizer, pipeline, set_seed, GenerationConfig

from mbr import MBR, MBRConfig
from mbr.metrics.comet import CometMetricRunner

language_pair = "de-en"
batch_size = 16

results_file = jsonlines.open(Path(__file__).parent / f"results_{language_pair}.19.jsonl", "w")

model_name = "facebook/wmt19-de-en"
model = MBR(FSMTForConditionalGeneration).from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
mt_pipeline = pipeline(
    "translation_" + language_pair.split("-")[0] + "_to_" + language_pair.split("-")[1],
    model=model,
    tokenizer=tokenizer,
    device=(0 if torch.cuda.is_available() else -1),
)
evaluation_metric_chrf = evaluate.load("chrf")
evaluation_metric_comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")

src_path = sacrebleu.get_source_file("wmt18", language_pair)
ref_path = sacrebleu.get_reference_files("wmt18", language_pair)[0]
dataset = load_dataset("text", data_files={"test": src_path})
references = Path(ref_path).read_text().splitlines()
assert len(dataset["test"]) == len(references)

# MBR
generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.do_sample = True
generation_config.num_beams = 1
generation_config.early_stopping = False
generation_config.epsilon_cutoff = 0.02

base_mbr_config = MBRConfig(
    num_samples=256,
    num_references=256,
)
base_mbr_config.metric_cache_size = batch_size * base_mbr_config.num_samples * base_mbr_config.num_references
mbr_configs = {}

# MBR without pruning (metric: ChrF++)
mbr_config = deepcopy(base_mbr_config)
mbr_config.metric = "chrf"
mbr_config.metric_output_field = "score"
mbr_config.metric_kwargs = {"word_order": 2, "eps_smoothing": True}
mbr_configs["MBR without pruning (metric: ChrF++)"] = mbr_config

# # MBR without pruning (metric: COMET)
# mbr_config = deepcopy(base_mbr_config)
# mbr_config.metric = "comet"
# mbr_config.metric_config_name = "Unbabel/wmt22-comet-da"
# mbr_config.metric_output_field = "mean_score"
# mbr_configs["MBR without pruning (metric: COMET)"] = mbr_config

# Pruning 𝛼=0.99 (metric: ChrF++)
mbr_config = deepcopy(base_mbr_config)
mbr_config.pruning = "confidence"
mbr_config.pruning_alpha = 0.99
mbr_config.initial_num_references = 16
mbr_config.metric = "chrf"
mbr_config.metric_output_field = "score"
mbr_config.metric_kwargs = {"word_order": 2, "eps_smoothing": True}
mbr_configs["Pruning 𝛼=0.99 (metric: ChrF++)"] = mbr_config

# # Pruning 𝛼=0.99 (metric: COMET)
# mbr_config = deepcopy(base_mbr_config)
# mbr_config.pruning = "confidence"
# mbr_config.pruning_alpha = 0.99
# mbr_config.initial_num_references = 8
# mbr_config.metric = "comet"
# mbr_config.metric_config_name = "Unbabel/wmt22-comet-da"
# mbr_config.metric_output_field = "mean_score"
# mbr_configs["Pruning 𝛼=0.99 (metric: COMET)"] = mbr_config

# Pruning 𝛼=0.9 (metric: ChrF++)
mbr_config = deepcopy(mbr_configs["Pruning 𝛼=0.99 (metric: ChrF++)"])
mbr_config.pruning_alpha = 0.9
mbr_configs["Pruning 𝛼=0.9 (metric: ChrF++)"] = mbr_config

# # Pruning 𝛼=0.9 (metric: COMET)
# mbr_config = deepcopy(mbr_configs["Pruning 𝛼=0.99 (metric: COMET)"])
# mbr_config.pruning_alpha = 0.9
# mbr_configs["Pruning 𝛼=0.9 (metric: COMET)"] = mbr_config

for method, mbr_config in mbr_configs.items():

    if mbr_config.metric == "comet":
        # Use an optimized metric runner for COMET
        metric_runner = CometMetricRunner(
            mbr_config,
            tokenizer,
            device=mt_pipeline.device,
            batch_size_embed=64,
            batch_size_estimate=64,
            progress_bar=True,
        )
    else:
        metric_runner = None

    set_seed(42)
    time_start = time.time()
    outputs = mt_pipeline(
        dataset["test"]["text"],
        mbr_config=mbr_config,
        generation_config=generation_config,
        tokenizer=tokenizer,
        metric_runner=metric_runner,
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
        word_order=2,  # ChrF++
        eps_smoothing=True,
    )
    comet_score = evaluation_metric_comet.compute(
        predictions=translations,
        references=references,
        sources=dataset["test"]["text"],
        gpus=0,
    )
    results_file.write({
        "language_pair": language_pair,
        "method": method,
        "chrf++": chrf_score["score"],
        "comet22": comet_score["mean_score"],
        "duration": time_end - time_start,
        "translations": translations,
    })

# Beam search
model = FSMTForConditionalGeneration.from_pretrained(model_name).half().to(mt_pipeline.device)
mt_pipeline.model = model
generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.num_beams = 10

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
    word_order=2,  # ChrF++
    eps_smoothing=True,
)
comet_score = evaluation_metric_comet.compute(
    predictions=translations,
    references=references,
    sources=dataset["test"]["text"],
    gpus=0,
)
results_file.write({
    "language_pair": language_pair,
    "method": f"beam search (beam size {generation_config.num_beams})",
    "chrf++": chrf_score["score"],
    "comet22": comet_score["mean_score"],
    "duration": time_end - time_start,
    "translations": translations,
})

results_file.close()
