from unittest import TestCase

from experiments.reference_aggregation.mbr_utils import CometUtility


class CometTestCase(TestCase):

    def setUp(self):
        self.comet = CometUtility("eamt22-cometinho-da")

    def test_compute_features(self):
        self.assertEqual(0, len(self.comet.embeddings))
        self.comet.compute_features({"This is a test.", "Dies ist ein Test."})
        self.assertEqual(2, len(self.comet.embeddings))

    def test_rank_samples_n_by_n(self):
        source_sequence = "This is a sample sentence"
        samples = ["Dies ist ein Beispiel.", "Dies ist ein Beispielsatz", "Dieser Satz macht keinen Sinn.", "Dies ist ein Test."]
        references = samples
        self.comet.compute_features({source_sequence} | set(samples) | set(references))

        indices = self.comet.rank_samples_n_by_s(source_sequence, samples, references, s=4)
        self.assertEqual(len(samples), len(indices))
        self.assertListEqual([1, 0, 3, 2], indices.tolist())

        # Test sample order invariance
        indices = self.comet.rank_samples_n_by_s(source_sequence, samples[::-1], references, s=4)
        self.assertListEqual([2, 3, 0, 1], indices.tolist())

        # Test reference order invariance
        indices = self.comet.rank_samples_n_by_s(source_sequence, samples, references[::-1], s=4)
        self.assertListEqual([1, 0, 3, 2], indices.tolist())

    def test_rank_samples_n_by_1(self):
        source_sequence = "This is a sample sentence"
        samples = ["Dies ist ein Beispiel.", "Dies ist ein Beispielsatz", "Dieser Satz macht keinen Sinn.", "Dies ist ein Test."]
        references = samples
        self.comet.compute_features({source_sequence} | set(samples) | set(references))

        indices = self.comet.rank_samples_n_by_s(source_sequence, samples, references, s=1)
        self.assertEqual(0, indices[0])  # Perfect match with itself
        self.assertListEqual([0, 1, 3, 2], indices.tolist())

        # Test sample order invariance
        indices = self.comet.rank_samples_n_by_s(source_sequence, samples[::-1], references, s=1)
        self.assertListEqual([3, 2, 0, 1], indices.tolist())
