from pathlib import Path
from unittest import TestCase

import numpy as np
from scipy.stats import kendalltau
from transformers import AutoTokenizer

from mbr import MBRConfig
from mbr.metrics.aggregate_comet import AggregateCometMetricRunner
from mbr.metrics.comet import CometMetricRunner


class AggregateCometTestCase(TestCase):

    def setUp(self):
        self.mbr_config = MBRConfig(
            metric="comet",
            metric_config_name="eamt22-cometinho-da",
            metric_output_field="mean_score",
        )
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.comet_runner = CometMetricRunner(
            self.mbr_config,
            self.tokenizer,
            batch_size_embed=64,
            batch_size_estimate=64,
            progress_bar=True,
            device=0,
        )
        self.aggregate_comet_runner = AggregateCometMetricRunner(
            self.mbr_config,
            self.tokenizer,
            batch_size_embed=64,
            batch_size_estimate=64,
            progress_bar=True,
            device=0,
        )

    def test_correlation(self):
        from sacrebleu import DATASETS
        wmt22 = DATASETS['wmt22']
        wmt22.process_to_text()
        for langpair in wmt22.langpairs:
            src_path = wmt22.get_source_file(langpair)
            reference_paths = wmt22.get_reference_files(langpair)
            all_paths = wmt22.get_files(langpair)
            hypotheses = []  # systems x segments (= batch_size)
            for path in all_paths:
                if path not in reference_paths:
                    hypotheses.append(Path(path).read_text().splitlines())
            source_sequences = Path(src_path).read_text().splitlines()
            # Use hypotheses as references
            references = hypotheses

            comet_scores_per_reference = self.comet_runner._compute_str_metric(
                samples=hypotheses,
                references=references,
                inputs=source_sequences,
            )
            comet_scores = np.array(comet_scores_per_reference.mean(axis=-1))
            aggregate_scores = np.array(self.aggregate_comet_runner._compute_str_metric(
                samples=hypotheses,
                references=references,
                inputs=source_sequences,
            ))
            self.assertEqual(comet_scores.shape, aggregate_scores.shape)

            print(f"Language pair: {langpair}")
            print(f"Number of segments: {comet_scores.shape[0]}")
            print(f"Number of systems: {comet_scores.shape[1]}")
            print(f"Number of references: {comet_scores_per_reference.shape[2]}")
            print(f"Number of segment scores: {len(comet_scores.flatten())}")
            # Calculate Kendall correlation
            correlation, pvalue = kendalltau(comet_scores.flatten(), aggregate_scores.flatten())
            print(f"{langpair} segment-level kendall correlation: {correlation}")
            # Calculate top-1 accuracy of finding the best (according to COMET) system translation for each segment
            max_indices = np.argmax(comet_scores, axis=1)
            top_1_accuracy = np.mean(np.array(aggregate_scores).argmax(axis=1) == max_indices)
            print(f"Top-1 accuracy: {top_1_accuracy}")
            for top_n in [5]:
                top_n_indices = np.argsort(-np.array(aggregate_scores), axis=1)[:, :top_n]
                top_n_accuracy = np.mean([max_indices[i] in top_n_indices[i] for i in range(len(max_indices))])
                print(f"Top-{top_n} accuracy: {top_n_accuracy}")
                print()
