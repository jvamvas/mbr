from pathlib import Path

import evaluate
import jsonlines
import sacrebleu
from datasets import load_dataset

beam_size = 4

results_file = jsonlines.open(Path(__file__).parent / f"results_beam{beam_size}.jsonl", "w")

for language_pair in ["de-en", "en-de", "en-ru", "ru-en"]:
    translations_name = f"transformer.wmt19.{language_pair}.single_model.beam{beam_size}.{language_pair.split('-')[1]}"
    translations_path = Path(__file__).parent / "translations" / translations_name
    assert translations_path.exists(), translations_path
    translations = translations_path.read_text().splitlines()

    chrf = evaluate.load("chrf")
    cometinho = evaluate.load("comet", "eamt22-cometinho-da")
    comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")

    src_path = sacrebleu.get_source_file("wmt19", language_pair)
    ref_path = sacrebleu.get_reference_files("wmt19", language_pair)[0]
    dataset = load_dataset("text", data_files={"test": src_path})
    references = Path(ref_path).read_text().splitlines()
    source_sequences = dataset["test"]["text"]
    assert len(dataset["test"]) == len(references) == len(source_sequences) == len(translations)


    chrf_score = chrf.compute(
        predictions=translations,
        references=references,
    )
    cometinho_score = cometinho.compute(
        predictions=translations,
        references=references,
        sources=source_sequences,
        gpus=0,
    )
    comet22_score = comet.compute(
        predictions=translations,
        references=references,
        sources=source_sequences,
        gpus=0,
    )
    results_file.write({
        "language_pair": language_pair,
        "method": f"beam search with beam size {beam_size}",
        "chrf": chrf_score["score"],
        "cometinho": cometinho_score["mean_score"],
        "comet22": comet22_score["mean_score"],
    })

results_file.close()
