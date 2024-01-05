import itertools
from typing import List, Dict, Set, Tuple

import torch
from tqdm import tqdm


@torch.no_grad()
def standard_comet(
        comet,
        samples: List[List[str]],
        references: List[List[str]],
        inputs: List[str],
        batch_size_embed: int = 1,
        batch_size_estimate: int = 1,
) -> torch.FloatTensor:
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

    return metric_scores.mean(dim=-1)


@torch.no_grad()
def aggregate_comet(
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

    return metric_scores
