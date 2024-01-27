from unittest import TestCase

from experiments.reference_aggregation.fairseq_utils import load_model


class EpsilonSamplingTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"

    def test_epsilon_sampling(self):
        model = load_model(self.language_pair)
        source_sentence = "This is a test."
        num_samples = 4
        # ε=0.02
        samples = model.sample(num_samples * [source_sentence], seed=42, sampling_epsilon_cutoff=0.02),
        self.assertEqual(len(samples), num_samples)
        self.assertIsInstance(samples[0], str)
        print(samples[0])
        # ε=0
        samples = model.sample(num_samples * [source_sentence], seed=42, sampling_epsilon_cutoff=0),
        self.assertEqual(len(samples), num_samples)
        self.assertIsInstance(samples[0], str)

    def test_extract_translations(self):
        ...  # TODO
