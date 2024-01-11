import time
import random

import sacrebleu
from datasets import load_dataset

from fairseq_utils import load_model

language_pairs = ["de-en", "en-de", "en-ru", "ru-en"]

num_samples = 1024
beam_size = 4

num_segments = 32  # total over all language pairs
assert num_segments % len(language_pairs) == 0

total_time = 0

for language_pair in language_pairs:
    src_path = sacrebleu.get_source_file("wmt22", language_pair)
    dataset = load_dataset("text", data_files={"test": src_path})
    source_sentences = dataset["test"]["text"]
    source_sentences = random.sample(source_sentences, num_segments // len(language_pairs))
    model = load_model(language_pair)

    time_start = time.time()
    for source_sentence in source_sentences:
        samples = model.translate(num_samples * [source_sentence], beam_size=beam_size)
    time_end = time.time()
    total_time += time_end - time_start

print(f"Total time elapsed: {total_time:.2f}s")
print(f"Average time per segment: {total_time / num_segments:.2f}s")
