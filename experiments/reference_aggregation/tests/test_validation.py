import logging
from pathlib import Path
from unittest import TestCase

import jsonlines

logging.basicConfig(level=logging.INFO)


class ValidationTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"
        self.test_dir = Path(__file__).parent / "out"
        self.test_dir.mkdir(exist_ok=True)

    def test_run_validation_cometinho(self):
        from experiments.reference_aggregation.validation import main
        jsonl_path = main(self.testset, self.language_pair, seed_no=0, utility_name="cometinho", topk=4, num_samples=8, limit_segments=4, out_dir=self.test_dir)
        self.assertTrue(jsonl_path.exists())
        with jsonlines.open(jsonl_path) as f:
            data = list(f)

        n_by_s_lines = [line for line in data if line["method"] == "n_by_s"]
        s_values = [line["s"] for line in n_by_s_lines]
        self.assertEqual([8, 4, 2, 1], s_values)

        for line in data:
            self.assertEqual(4, len(line["rankings"]))
            self.assertEqual(4, len(line["rankings"][0]))
            self.assertEqual(4, len(set(line["rankings"][0])))

    def test_plot_accuracy(self):
        ...  # TODO
