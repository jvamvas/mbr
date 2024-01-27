from unittest import TestCase


class GenerateSamplesTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"

    def test_generate_samples(self):
        from experiments.reference_aggregation.generate_samples import main
        out_path = main(self.testset, self.language_pair, seedno=0, num_samples=1024, epsilon_cutoff=0.02, limit_segments=4)