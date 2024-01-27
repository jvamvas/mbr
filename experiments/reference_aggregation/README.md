## Code for the paper "Efficient Minimum Bayes Risk Decoding with Reference Aggregation"

- The research code in this directory implements _reference aggregation_, an efficiency method for MBR that uses aggregate reference representations for faster utility estimation.
- We apply reference aggregation to two metrics: ChrF and COMET.
- Unlike the **mbr** package, the code in this directory is purely research-oriented (= reproducing the tables and figures in our paper) and not optimized for usability.

## Installation
- Requires Python >= 3.9 and PyTorch.
- `pip install -r requirements.txt`

## Reproducing the experiments

### Creating the samples
- Warning: The following code downloads a large translation model from PyTorch Hub (if not already present) and create 1024 samples per segment, which will take some time.
- Samples will be stored in a JSON lines file in the directory `samples/`.
```bash
python generate_samples.py --testset wmt21 --language-pair en-de --seed 0
``` 

### Figure 1: Top-20 accuracy
#### Generating the translations
- Performing this analysis is computationally heavy because we run it for many different values of _s_ (x-axis of Figure 1).
- We run N-by-N MBR, N-by-S MBR and Reference Aggregation in a single script, and all values of _s_, so that the embedding part of COMET only needs to run once.
- The results are stored in a JSON lines file in the directory `validation_output/`. Each line describes the output for one method and one value of _s_.
- In addition, the translations will be stored in text files (one translation per line) in the `translations/` directory, to allow for easy evaluation.
- The utility metric is either `"chrf"`, `"cometinho"` or `"comet22"`.
```bash
python validation.py --testset wmt21 --language-pair en-de --seed 0 --utility comet22 --topk 20
```
#### Calculating accuracy
- After the script has run, the series for Figure 1 (top-20 accuracy) can be printed as follows.
- The method can be either `"nbys"` or `"aggregate"`.
```bash
python plot_accuracy.py --testset wmt21 --language-pair en-de --seed 0 --utility comet22 --topk 20 --method aggregate
```

### Table 1: Test results

#### Generating the translations
- In the test results table, we compare the translation quality of beam search, epsilon sampling, standard (pairwise) MBR, and reference aggregation. For COMET, we also experiment with aggregate-to-fine MBR.
- The following scripts create the translations and store them in the `translations/` directory.
```bash
# Beam search
python baseline_beam_search.py --language-pair en-de --testset wmt22
# MBR with CHRF metric – standard MBR
python run_mbr.py --method pairwise --testset wmt22 --language-pair en-de --seed 0 --utility chrf
# MBR with CHRF metric – reference aggregation
python run_mbr.py --method aggregate --testset wmt22 --language-pair en-de --seed 0 --utility chrf
# MBR with COMET-22 metric – standard MBR
python run_mbr.py --method pairwise --testset wmt22 --language-pair en-de --seed 0 --utility comet22
# MBR with COMET-22 metric – reference aggregation
python run_mbr.py --method aggregate --testset wmt22 --language-pair en-de --seed 0 --utility comet22
# MBR with COMET-22 metric – aggregate-to-fine MBR
python run_mbr.py --method aggregate_to_fine --topk 20 --testset wmt22 --language-pair en-de --seed 0 --utility comet22
```
- For epsilon sampling, we simply read the JSON lines file created by `generate_samples.py` and extract the first sample for each segment.
```bash
python baseline_epsilon_sampling.py --testset wmt22 --language-pair en-de --seed 0
```

#### Saving the source sequences and references in a text file
- The sequences will be stored in text files in the `translations/` directory
```bash
python save_src_and_ref.py --testset wmt22 --language-pair en-de
```

#### Evaluating the translations
- Use code of your choice (e.g., https://github.com/mjpost/sacrebleu) to perform the evaluation.