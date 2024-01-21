from pathlib import Path

import jsonlines

out_dir = Path(__file__).parent / "translations"
out_dir.mkdir(exist_ok=True)

split = 'test'

language_pairs = ["en-de", "de-en", "en-ru", "ru-en"]

seed_nos = [0, 1]
num_seeds = len(seed_nos)

if split == "test":
    testset = "wmt22"
else:
    testset = "wmt21"

for language_pair in language_pairs:
    for seed_no in seed_nos:
        # Sampling
        samples_dir = Path(__file__).parent / f"samples_{testset}"
        samples_path = samples_dir / f"transformer.wmt19.{language_pair}.single_model.1024samples.epsilon0.02.seed{seed_no}.jsonl"
        out_path = out_dir / f"epsilon_sampling_{testset}_{language_pair}_seed{seed_no}.txt"
        with jsonlines.open(samples_path) as f_in, open(out_path, "w") as f_out:
            f_out.write("\n".join([line["samples"][0] for line in f_in]))

        # ChrF
        chrf_results_path = Path(f"results_chrf_{testset}_{language_pair}_1024samples_seed{seed_no}.jsonl")
        with jsonlines.open(chrf_results_path) as f:
            data = list(f)
        for row in data:
            out_path = out_dir / f"chrf_{testset}_{language_pair}_1024samples_seed{seed_no}_{row['num_aggregates']}aggregates.txt"
            with open(out_path, "w") as f:
                f.write("\n".join(row["translations"]))

        # Cometinho
        cometinho_results_path = Path(f"results_cometinho_{testset}_{language_pair}_1024samples_seed{seed_no}.jsonl")
        with jsonlines.open(cometinho_results_path) as f:
            data = list(f)
        for row in data:
            out_path = out_dir / f"cometinho_{testset}_{language_pair}_1024samples_seed{seed_no}_{row['num_aggregates']}aggregates.txt"
            with open(out_path, "w") as f:
                f.write("\n".join(row["translations"]))

        # Comet
        comet_results_path = Path(f"results_comet22_{testset}_{language_pair}_1024samples_seed{seed_no}.jsonl")
        with jsonlines.open(comet_results_path) as f:
            data = list(f)
        for row in data:
            out_path = out_dir / f"comet22_{testset}_{language_pair}_1024samples_seed{seed_no}_{row['num_aggregates']}aggregates.txt"
            with open(out_path, "w") as f:
                f.write("\n".join(row["translations"]))
