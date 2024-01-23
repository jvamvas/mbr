from pathlib import Path

import sacrebleu

out_dir = Path(__file__).parent / "references"
out_dir.mkdir(exist_ok=True)

for split in ["valid", "test"]:
    language_pairs = ["en-de", "de-en", "en-ru", "ru-en"]

    seed_nos = [0, 1]
    num_seeds = len(seed_nos)

    if split == "test":
        testset = "wmt22"
    else:
        testset = "wmt21"

    for language_pair in language_pairs:
        ref_path = sacrebleu.get_reference_files(testset, language_pair)[0]
        references = Path(ref_path).read_text().splitlines()
        with open(out_dir / f"{testset}_{language_pair}.{language_pair.split('-')[1]}", "w") as f:
            f.write("\n".join(references))
