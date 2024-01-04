import sys
from pathlib import Path

import jsonlines
import sacrebleu
import torch
from datasets import load_dataset
from tqdm import tqdm

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

num_samples = 256
seed = 42

src_path = sacrebleu.get_source_file("wmt19", language_pair)
dataset = load_dataset("text", data_files={"test": src_path})
source_sentences = dataset["test"]["text"]

model = torch.hub.load('pytorch/fairseq', f'transformer.wmt19.{language_pair}.single_model')
model.eval()
if torch.cuda.is_available():
    model.cuda()

out_dir = Path(__file__).parent / f"samples"
out_dir.mkdir(exist_ok=True, parents=True)
out_path = out_dir / f"wmt19.{language_pair}.{num_samples}.unbiased.seed{seed}.jsonl"

with jsonlines.open(out_path, "w") as f:
    for source_sentence in tqdm(source_sentences):
        f.write({
            "samples": [
                model.sample(num_samples * [source_sentence], sampling=True)
            ]
        })
