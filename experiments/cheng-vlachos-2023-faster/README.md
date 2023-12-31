This directory uses the [**mbr**](https://github.com/ZurichNLP/mbr) package to reproduce an experiment from the paper [Faster Minimum Bayes Risk Decoding with Confidence-based Pruning](https://aclanthology.org/2023.emnlp-main.767/) (Cheng & Vlachos, 2023).

## Setup
* Task: Machine translation
* Translation direction: de-en
* MBR metrics: chrF++ ([Popović, 2017](https://aclanthology.org/W17-4770/)) and COMET-22 ([Rei et al., 2022](https://aclanthology.org/2022.wmt-1.52/))
* Number of samples: 256
* Sampling approach: epsilon sampling with ε=0.02
* Samples and references are the same
* Confidence-based pruning ([Cheng & Vlachos, 2023](https://aclanthology.org/2023.emnlp-main.767/)) with 𝛼=0.99 or 𝛼=0.9, and r₁=16 (chrF++) or r₁=8 (COMET-22)
* Test set: newstest2018
* Evaluation metrics: chrF++ and COMET-22
* Baselines: MBR without pruning; beam search with beam size 10

## Differences to the paper
* The paper used custom models trained without label smoothing, this reproduction uses an open-source model ([Ng et al., WMT 2019](https://aclanthology.org/W19-5333/)).
* The paper used different sets as samples and references. Samples were generated using beam search. This reproduction uses the same set as samples and references, generated with epsilon sampling.
* In the paper, the segments of the test set were translated one by one – batching was used for sampling, but not for the overall MBR decoding process. Our implementation supports batched translation, and we use a batch size of 16 in this experiment.
* The paper proposes to generate the references during pruning and to terminate early if all samples but one have been pruned. This implementation generates the complete set of references before pruning.

## Results