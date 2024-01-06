import sys
from pathlib import Path

import sacrebleu
from datasets import load_dataset

from fairseq_utils import load_model

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

beam_size = 4

src_path = sacrebleu.get_source_file("wmt19", language_pair)
dataset = load_dataset("text", data_files={"test": src_path})
source_sentences = dataset["test"]["text"]

model = load_model(language_pair)

out_dir = Path(__file__).parent / f"translations"
out_dir.mkdir(exist_ok=True, parents=True)
out_path = out_dir / f"{model}.beam{beam_size}.{language_pair.split('-')[1]}"

translations = model.translate(source_sentences, beam_size=beam_size)
assert len(translations) == len(source_sentences)

with open(out_path, "w") as f:
    for translation in translations:
        f.write(translation + "\n")
