import logging
from pathlib import Path
from unittest import TestCase

logging.basicConfig(level=logging.INFO)


class ValidationTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"
        self.test_dir = Path(__file__).parent / "out"
        self.test_dir.mkdir(exist_ok=True)

    def test_run_validation_cometinho(self):
        from experiments.reference_aggregation.validation import main
        main(self.testset, self.language_pair, seed_no=0, utility="cometinho", num_samples=8, limit_segments=4, out_dir=self.test_dir)

    def test_plot_accuracy(self):
        ...  # TODO
