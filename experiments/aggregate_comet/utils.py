import itertools
import math
import random
import time
from typing import List, Dict, Set, Tuple

import numpy as np
import torch
from fastchrf import pairwise_chrf, aggregate_chrf
from tqdm import tqdm


@torch.no_grad()
def mbr_standard_comet(
        comet,
        samples: List[List[str]],  # num_samples x batch_size
        references: List[List[str]],  # num_references x batch_size
        inputs: List[str],  # batch_size
        batch_size_embed: int = 1,
        batch_size_estimate: int = 1,
) -> List[str]:
    batch_size = len(samples[0])
    metric_scores = torch.zeros((batch_size, len(samples), len(references)))
    for i in tqdm(list(range(batch_size)), desc="comet"):
        # Embed all sequences
        all_samples = [sample[i] for sample in samples]
        all_references = [reference[i] for reference in references]
        all_sequences = set(all_samples + all_references + inputs)

        all_embeddings: Dict[str, torch.FloatTensor] = {}
        # Compute embeddings
        if all_sequences:
            all_sequences = list(all_sequences)
            encodings = comet.scorer.encoder.prepare_sample(all_sequences).to(comet.scorer.device)
            batches = itertools.zip_longest(range(0, len(all_sequences), batch_size_embed),
                                            range(batch_size_embed, len(all_sequences), batch_size_embed))
            for start_idx, end_idx in batches:
                embeddings = comet.scorer.get_sentence_embedding(
                    input_ids=encodings["input_ids"][start_idx:end_idx],
                    attention_mask=encodings["attention_mask"][start_idx:end_idx],
                )
                for j in range(start_idx, end_idx if end_idx is not None else len(all_sequences)):
                    embedding = embeddings[j - start_idx]
                    all_embeddings[all_sequences[j]] = embedding

        # Collect all input triples in a list
        input_triples: Set[Tuple[str, str, str]] = set()
        for j in range(len(samples)):
            for k in range(len(references)):
                input_triples.add((inputs[i], samples[j][i], references[k][i]))

        input_triple_scores: Dict[Tuple[str, str, str], torch.FloatTensor] = {}
        # Compute scores for input triples
        input_triples: List = list(input_triples)
        batches = itertools.zip_longest(range(0, len(input_triples), batch_size_estimate),
                                        range(batch_size_estimate, len(input_triples),
                                              batch_size_estimate))
        for start_idx, end_idx in batches:
            batch = input_triples[start_idx:end_idx]
            batch_scores = comet.scorer.estimate(
                src_sentemb=torch.stack([all_embeddings[triple[0]] for triple in batch]),
                mt_sentemb=torch.stack([all_embeddings[triple[1]] for triple in batch]),
                ref_sentemb=torch.stack([all_embeddings[triple[2]] for triple in batch]),
            )
            for j in range(start_idx, end_idx if end_idx is not None else len(input_triples)):
                triple = batch[j - start_idx]
                score = batch_scores.score[j - start_idx]
                input_triple_scores[triple] = score

        for j in range(len(samples)):
            for k in range(len(references)):
                metric_scores[i, j, k] = input_triple_scores[(inputs[i], samples[j][i], references[k][i])]

    metric_scores = metric_scores.mean(dim=-1)
    translations = []
    for i in range(batch_size):
        max_index = metric_scores[i].argmax()
        translation = samples[max_index][i]
        translations.append(translation)
    return translations


@torch.no_grad()
def mbr_aggregate_comet(
                    comet,
                    samples: List[List[str]],
                    references: List[List[str]],
                    inputs: List[str],
                    batch_size_embed: int = 1,
                    batch_size_estimate: int = 1,
                    ) -> torch.FloatTensor:
    if inputs is None:
        raise NotImplementedError("CometMetricRunner requires source sequences (`inputs`) to be provided")
    batch_size = len(samples[0])
    metric_scores = torch.zeros((batch_size, len(samples)))
    for i in tqdm(list(range(batch_size)), desc="comet"):
        # Embed all sequences
        all_samples = [sample[i] for sample in samples]
        all_references = [reference[i] for reference in references]
        all_sequences = set(all_samples + all_references + inputs)

        all_embeddings: Dict[str, torch.FloatTensor] = {}
        # Compute embeddings
        if all_sequences:
            all_sequences = list(all_sequences)
            encodings = comet.scorer.encoder.prepare_sample(all_sequences).to(comet.scorer.device)
            batches = itertools.zip_longest(range(0, len(all_sequences), batch_size_embed),
                                            range(batch_size_embed, len(all_sequences), batch_size_embed))
            for start_idx, end_idx in batches:
                embeddings = comet.scorer.get_sentence_embedding(
                    input_ids=encodings["input_ids"][start_idx:end_idx],
                    attention_mask=encodings["attention_mask"][start_idx:end_idx],
                )
                for j in range(start_idx, end_idx if end_idx is not None else len(all_sequences)):
                    embedding = embeddings[j - start_idx]
                    all_embeddings[all_sequences[j]] = embedding

        # Compute average reference embedding
        avg_reference_embedding = torch.stack([all_embeddings[reference] for reference in all_references]).mean(dim=0)

        # Collect all input triples in a list
        input_triples: Set[Tuple[str, str, str]] = set()
        for j in range(len(samples)):
            input_triples.add((inputs[i], samples[j][i], "avg"))

        input_triple_scores: Dict[Tuple[str, str, str], torch.FloatTensor] = {}
        # Compute scores for input triples
        input_triples: List = list(input_triples)
        batches = itertools.zip_longest(range(0, len(input_triples), batch_size_estimate),
                                        range(batch_size_estimate, len(input_triples),
                                              batch_size_estimate))
        for start_idx, end_idx in batches:
            batch = input_triples[start_idx:end_idx]
            batch_scores = comet.scorer.estimate(
                src_sentemb=torch.stack([all_embeddings[triple[0]] for triple in batch]),
                mt_sentemb=torch.stack([all_embeddings[triple[1]] for triple in batch]),
                ref_sentemb=avg_reference_embedding.unsqueeze(0).repeat(len(batch), 1),
            )
            for j in range(start_idx, end_idx if end_idx is not None else len(input_triples)):
                triple = batch[j - start_idx]
                score = batch_scores.score[j - start_idx]
                input_triple_scores[triple] = score

        for j in range(len(samples)):
            metric_scores[i, j] = input_triple_scores[(inputs[i], samples[j][i], "avg")]

    translations = []
    for i in range(batch_size):
        max_index = metric_scores[i].argmax()
        translation = samples[max_index][i]
        translations.append(translation)
    return translations


@torch.no_grad()
def run_all_comet_variants(
        comet,
        samples: List[List[str]],  # num_samples x batch_size
        references: List[List[str]],  # num_references x batch_size
        inputs: List[str],  # batch_size
        batch_size_embed: int = 1,
        batch_size_estimate: int = 1,
) -> Tuple[Tuple[List[str], ...], Tuple[float, ...]]:
    """
    Experimental implementation of reference aggregation with COMET.
    Returns several set of translations
    - MBR with aggregate COMET (2**0 = 1 subset = average of all references)
    - MBR with aggregate COMET (2**1 = 2 subsets = average of n/2 references)
    - MBR with aggregate COMET (2**2 = 4 subsets = average of n/4 references)
    ...
    - MBR with aggregate COMET (2**log2(n) = n subsets = pairwise COMET)
    Also returns the duration of each method.
    """
    batch_size = len(samples[0])
    num_samples = len(samples)
    num_references = len(references)
    num_iterations = math.log2(num_references) + 1
    assert num_iterations.is_integer()
    num_iterations = int(num_iterations)

    total_embedding_time = 0
    scoring_times = np.zeros(num_iterations)

    all_translations: List[List[str]] = [list() for _ in range(num_iterations)]

    for i in tqdm(list(range(batch_size)), desc="comet"):

        # Compute embeddings
        start = time.time()
        all_samples = [sample[i] for sample in samples]
        all_references = [reference[i] for reference in references]
        all_sequences = set(all_samples + all_references + inputs)
        all_embeddings: Dict[str, torch.FloatTensor] = {}
        if all_sequences:
            all_sequences = list(all_sequences)
            encodings = comet.scorer.encoder.prepare_sample(all_sequences).to(comet.scorer.device)
            batches = itertools.zip_longest(range(0, len(all_sequences), batch_size_embed),
                                            range(batch_size_embed, len(all_sequences), batch_size_embed))
            for start_idx, end_idx in batches:
                embeddings = comet.scorer.get_sentence_embedding(
                    input_ids=encodings["input_ids"][start_idx:end_idx],
                    attention_mask=encodings["attention_mask"][start_idx:end_idx],
                )
                for j in range(start_idx, end_idx if end_idx is not None else len(all_sequences)):
                    embedding = embeddings[j - start_idx]
                    all_embeddings[all_sequences[j]] = embedding
        end = time.time()
        total_embedding_time += (end - start)

        iterations = list(range(num_iterations))
        # Shuffle to make time measurements more robust
        random.shuffle(iterations)
        for j in iterations:
            start = time.time()
            num_agg_references = int(2 ** j)
            subset_size = num_references // num_agg_references
            metric_scores = torch.zeros((num_samples, num_agg_references))

            # Compute average reference embeddings
            reference_embeddings = torch.stack([all_embeddings[reference] for reference in all_references])
            avg_reference_embeddings = reference_embeddings.view(num_agg_references, subset_size, -1).mean(dim=1)

            # Collect all input triples in a list
            input_triples: Set[Tuple[str, str, str]] = set()
            for k in range(len(samples)):
                for m in range(num_agg_references):
                    input_triples.add((inputs[i], samples[k][i], f"agg{m}"))

            # Compute scores for input triples
            input_triple_scores: Dict[Tuple[str, str, str], torch.FloatTensor] = {}
            input_triples: List = list(input_triples)
            batches = itertools.zip_longest(range(0, len(input_triples), batch_size_estimate),
                                            range(batch_size_estimate, len(input_triples),
                                                  batch_size_estimate))
            for start_idx, end_idx in batches:
                batch = input_triples[start_idx:end_idx]
                batch_scores = comet.scorer.estimate(
                    src_sentemb=torch.stack([all_embeddings[triple[0]] for triple in batch]),
                    mt_sentemb=torch.stack([all_embeddings[triple[1]] for triple in batch]),
                    ref_sentemb=torch.stack([avg_reference_embeddings[int(triple[2][3:])] for triple in batch]),
                )
                for k in range(start_idx, end_idx if end_idx is not None else len(input_triples)):
                    triple = batch[k - start_idx]
                    score = batch_scores.score[k - start_idx]
                    input_triple_scores[triple] = score

            for k in range(len(samples)):
                for m in range(num_agg_references):
                    metric_scores[k, m] = input_triple_scores[(inputs[i], samples[k][i], f"agg{m}")]

            metric_scores = metric_scores.mean(dim=-1)
            max_index = metric_scores.argmax()
            translation = samples[max_index][i]
            all_translations[j].append(translation)
            end = time.time()
            scoring_times[j] += (end - start)

    durations = total_embedding_time + scoring_times
    return tuple(all_translations), tuple(durations)


def mbr_standard_chrf(
        samples: List[List[str]],  # batch_size x num_samples
        references: List[List[str]],  # batch_size x num_references
):
    batch_size = len(samples)
    scores_per_reference = pairwise_chrf(
        samples,
        references,
    )
    scores_per_reference = np.array(scores_per_reference)
    scores = scores_per_reference.mean(axis=-1)
    translations = []
    for i in range(batch_size):
        max_index = scores[i].argmax()
        translation = samples[max_index][i]
        translations.append(translation)
    return translations


def mbr_aggregate_chrf(
        samples: List[List[str]],  # batch_size x num_samples
        references: List[List[str]],  # batch_size x num_references
):
    batch_size = len(samples)
    scores = aggregate_chrf(
        samples,
        references,
    )
    translations = []
    for i in range(batch_size):
        max_index = scores[i].argmax()
        translation = samples[max_index][i]
        translations.append(translation)
    return translations
