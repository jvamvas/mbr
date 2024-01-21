import sys
from pathlib import Path

import numpy as np
import sacrebleu
from tqdm import tqdm

from gemba import CREDENTIALS
from gemba.cache import Cache
from gemba.gpt_api import GptApi
from gemba.prompt import prompts, language_codes

# OPENAI_MODEL = "babbage-002"
OPENAI_MODEL = "gpt-3.5-turbo-1106"
# OPENAI_MODEL = "gpt-4-1106-preview"

# LIMIT_N = 10
LIMIT_N = None

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]
translations_path = Path(sys.argv[2])
assert translations_path.exists()

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

cache_filename = f"{OPENAI_MODEL}_{wmt}_{language_pair}.jsonl"
cache = Cache(cache_filename)

gptapi = GptApi(CREDENTIALS.credentials)
gemba_type = "GEMBA-DA_ref"

scores = []
for source, reference, translation in zip(tqdm(source_sequences), references, translations):
    data = {
        "source_seg": source,
        "target_seg": translation,
        "reference_seg": reference,
        "source_lang": language_codes[language_pair.split("-")[0]],
        "target_lang": language_codes[language_pair.split("-")[1]],
    }
    prompt = prompts[gemba_type]["prompt"].format(**data)
    parsed_answers = gptapi.request(prompt, OPENAI_MODEL, prompts[gemba_type]["validate_answer"], cache=cache)
    score = parsed_answers[0]["answer"]
    scores.append(score)

print("Model:", OPENAI_MODEL)
print("Number of segments", len(scores))
print("NaN rate", sum([1 for score in scores if score is None]) / len(scores))
print("Average score:", np.nanmean(np.array(scores, dtype=float)))
