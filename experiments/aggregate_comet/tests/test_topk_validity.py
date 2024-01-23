from pathlib import Path
from unittest import TestCase

import evaluate
import jsonlines
import sacrebleu
from datasets import load_dataset

from experiments.aggregate_comet.utils import run_all_comet_factors


class TopkValidityTestCase(TestCase):

    def setUp(self):
        language_pair = "en-de"
        wmt = "wmt21"
        epsilon_cutoff = 0.02
        seed_no = 1
        samples_dir = Path(__file__).parent.parent / f"samples_{wmt}"
        samples_name = f"transformer.wmt19.{language_pair}.single_model.1024samples.epsilon{epsilon_cutoff}.seed{seed_no}.jsonl"
        samples_path = samples_dir / samples_name
        assert samples_path.exists(), samples_path
        samples = []  # segments x num_samples
        with jsonlines.open(samples_path) as f:
            for line in f:
                samples.append(line["samples"])
        self.comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        self.comet.scorer = self.comet.scorer.to("cuda:0")
        src_path = sacrebleu.get_source_file(wmt, language_pair)
        ref_path = sacrebleu.get_reference_files(wmt, language_pair)[0]
        dataset = load_dataset("text", data_files={"test": src_path})
        references = Path(ref_path).read_text().splitlines()
        source_sequences = dataset["test"]["text"]
        assert len(dataset["test"]) == len(references) == len(source_sequences)
        self.samples = samples[:16]
        self.source_sequences = source_sequences[:16]
        self.references = references[:16]

    def test_topk(self):
        """
        Top translation of top-k should be identical to the output of direct
        """
        topk_translation_lists, _ = run_all_comet_factors(
            self.comet,
            samples=[[row[i] for row in self.samples] for i in range(len(self.samples[0]))],
            references=[[row[i] for row in self.samples] for i in range(len(self.samples[0]))],
            inputs=self.source_sequences,
            return_top_n=20,
        )
        direct_translation_lists, _ = run_all_comet_factors(
            self.comet,
            samples=[[row[i] for row in self.samples] for i in range(len(self.samples[0]))],
            references=[[row[i] for row in self.samples] for i in range(len(self.samples[0]))],
            inputs=self.source_sequences,
        )
        for topk_translation_list, direct_translation_list in zip(topk_translation_lists, direct_translation_lists):
            for topk_translations, direct_translation in zip(topk_translation_list, direct_translation_list):
                self.assertEqual(topk_translations[0], direct_translation)
