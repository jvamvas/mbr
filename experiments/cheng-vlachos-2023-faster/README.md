This directory uses the [**mbr**](https://github.com/ZurichNLP/mbr) package to reproduce an experiment from the paper [Faster Minimum Bayes Risk Decoding with Confidence-based Pruning](https://aclanthology.org/2023.emnlp-main.767/) (Cheng & Vlachos, 2023).

## Setup
* Task: Machine translation
* Translation direction: de-en
* MBR metric: COMET-22 ([Rei et al., 2022](https://aclanthology.org/2022.wmt-1.52/))
* Number of samples: 256
* Sampling approach: epsilon sampling with ε=0.02
* Samples and references are the same
* Confidence-based pruning ([Cheng & Vlachos, 2023](https://aclanthology.org/2023.emnlp-main.767/)) with 𝛼=0.99 or 𝛼=0.9, and r₁=8
* Test set: newstest21
* Evaluation metrics: chrF++ and COMET-22
* Baselines: MBR without pruning; beam search with beam size 10

## Differences to the paper
* The paper used custom models trained without label smoothing, this reproduction uses an open-source model ([Ng et al., WMT 2019](https://aclanthology.org/W19-5333/)).
* The paper evaluated on newstest18, this reproduction evaluates on newstest21.
* The paper used different sets as samples and references. Samples were generated using beam search. This reproduction uses the same set as samples and references, generated with epsilon sampling.
* In the paper, the segments of the test set were translated one by one – batching was used for sampling, but not for the overall MBR decoding process. Our implementation supports batched translation, and we use a batch size of 16 in this experiment.
* The paper proposed terminating early if there is only one sample left. This implementation supports early termination only if the condition is met for all items in the batch.
* The paper reports average results over 10 runs. Here, we report results for a single run.

## Results
