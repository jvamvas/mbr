import sys
from pathlib import Path

import jsonlines
import sacrebleu
from datasets import load_dataset
from tqdm import tqdm

from fairseq_utils import load_model

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

SEEDS = [
    553589,
    456178,
    817304,
    6277,
    792418,
    707983,
    249859,
    272618,
    760402,
    472974,
]

seed_no = int(sys.argv[2])
seed = SEEDS[seed_no]

num_samples = 1024
epsilon_cutoff = 0.02

src_path = sacrebleu.get_source_file("wmt21", language_pair)
dataset = load_dataset("text", data_files={"test": src_path})
source_sentences = dataset["test"]["text"]

model = load_model(language_pair)

out_dir = Path(__file__).parent / f"samples_wmt21"
out_dir.mkdir(exist_ok=True, parents=True)
out_path = out_dir / f"{model}.{num_samples}samples.epsilon{epsilon_cutoff}.seed{seed_no}.jsonl"

with jsonlines.open(out_path, "w") as f:
    for source_sentence in tqdm(source_sentences):
        f.write({
            "samples": model.sample(num_samples * [source_sentence], seed=seed, sampling_epsilon_cutoff=epsilon_cutoff),
        })
