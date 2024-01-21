import sys
from pathlib import Path

import numpy as np
import sacrebleu
from tqdm import tqdm
from lmql.runtime.bopenai import get_stats

from gemba import GembaMetric

LIMIT_N = 10

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]
translations_path = Path(sys.argv[2])
assert translations_path.exists()

language_codes = {
    "en": "English",
    "de": "German",
    "ru": "Russian",
}

OPENAI_MODEL = "openai/babbage-002"
# OPENAI_MODEL = "openai/gpt-4-1106-preview"
gemba = GembaMetric(OPENAI_MODEL)

wmt = "wmt22"
src_path = sacrebleu.get_source_file(wmt, language_pair)
ref_path = sacrebleu.get_reference_files(wmt, language_pair)[0]
source_sequences = Path(src_path).read_text().splitlines()
references = Path(ref_path).read_text().splitlines()
assert len(source_sequences) == len(references)
translations = Path(translations_path).read_text().splitlines()
assert len(source_sequences) == len(translations)

if LIMIT_N is not None:
    print(f"Limiting to {LIMIT_N} segments")
    source_sequences = source_sequences[:LIMIT_N]
    references = references[:LIMIT_N]
    translations = translations[:LIMIT_N]

scores = []
for source, reference, translation in zip(tqdm(source_sequences), references, translations):
    data = {
        "src": source,
        "ref": reference,
        "hyp": translation,
        "src_lang": language_codes[language_pair.split("-")[0]],
        "tgt_lang": language_codes[language_pair.split("-")[1]],
    }
    score = gemba.score(**data)
    scores.append(score)

try:
    print(get_stats())
except RuntimeError:
    pass

print("Number of segments", len(scores))
print("NaN rate", sum([1 for score in scores if score is None]) / len(scores))
print("Average score:", np.nanmean(np.array(scores, dtype=float)))
