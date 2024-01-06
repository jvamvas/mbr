import sys
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

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

batch_size = 32

results_file = jsonlines.open(Path(__file__).parent / f"results_{language_pair}.b.jsonl", "w")

model_name = f"facebook/wmt19-{language_pair}"
model = MBR(FSMTForConditionalGeneration).from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
mt_pipeline = pipeline(
    "translation_" + language_pair.split("-")[0] + "_to_" + language_pair.split("-")[1],
    model=model,
    tokenizer=tokenizer,
    device=(0 if torch.cuda.is_available() else -1),
)
evaluation_metric_chrf = evaluate.load("chrf")
evaluation_metric_cometinho = evaluate.load("comet", "eamt22-cometinho-da")
evaluation_metric_comet22 = evaluate.load("comet", "Unbabel/wmt22-comet-da")

src_path = sacrebleu.get_source_file("wmt19", language_pair)
ref_path = sacrebleu.get_reference_files("wmt19", language_pair)[0]
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

# # MBR with standard COMET
# mbr_config = deepcopy(base_mbr_config)
# mbr_config.metric = "comet"
# mbr_config.metric_config_name = "eamt22-cometinho-da"
# mbr_config.metric_output_field = "mean_score"
# mbr_configs["MBR with standard COMET"] = mbr_config

# MBR with aggregate COMET
mbr_config = deepcopy(base_mbr_config)
mbr_config.metric = "aggregate_comet"
mbr_config.metric_config_name = "eamt22-cometinho-da"
mbr_config.metric_output_field = "mean_score"
mbr_configs["MBR with aggregate COMET"] = mbr_config

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
    cometinho_score = evaluation_metric_cometinho.compute(
        predictions=translations,
        references=references,
        sources=dataset["test"]["text"],
        gpus=0,
    )
    comet22_score = evaluation_metric_comet22.compute(
        predictions=translations,
        references=references,
        sources=dataset["test"]["text"],
        gpus=0,
    )
    results_file.write({
        "language_pair": language_pair,
        "method": method,
        "chrf": chrf_score["score"],
        "cometinho": cometinho_score["mean_score"],
        "comet22": comet22_score["mean_score"],
        "duration": time_end - time_start,
        "translations": translations,
    })

# Beam search
model = FSMTForConditionalGeneration.from_pretrained(model_name).half().to(mt_pipeline.device)
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
cometinho_score = evaluation_metric_cometinho.compute(
    predictions=translations,
    references=references,
    sources=dataset["test"]["text"],
    gpus=0,
)
comet22_score = evaluation_metric_comet22.compute(
    predictions=translations,
    references=references,
    sources=dataset["test"]["text"],
    gpus=0,
)
results_file.write({
    "language_pair": language_pair,
    "method": f"beam search (beam size {generation_config.num_beams})",
    "chrf": chrf_score["score"],
    "cometinho": cometinho_score["mean_score"],
    "comet22": comet22_score["mean_score"],
    "duration": time_end - time_start,
    "translations": translations,
})

results_file.close()