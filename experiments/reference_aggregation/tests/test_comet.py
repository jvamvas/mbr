from unittest import TestCase

from experiments.reference_aggregation.mbr_utils import CometUtility


class CometTestCase(TestCase):

    def setUp(self):
        self.comet = CometUtility("eamt22-cometinho-da")

    def test_compute_features(self):
        self.assertEqual(0, len(self.comet.embeddings))
        self.comet.compute_features({"This is a test.", "Dies ist ein Test."})
        self.assertEqual(2, len(self.comet.embeddings))

    def test_rank_samples_n_by_s(self):
        source_sequence = "This is a sample sentence"
        samples = ["Dies ist ein Beispiel.", "Dies ist ein Beispielsatz", "Dieser Satz macht keinen Sinn.", "Dies ist ein Test."]
        references = samples
        self.comet.compute_features({source_sequence} | set(samples) | set(references))

        indices = self.comet.rank_samples_n_by_s(source_sequence, samples, references, s=4)
        self.assertEqual(len(samples), len(indices))
        self.assertListEqual([1, 0, 3, 2], indices.tolist())
