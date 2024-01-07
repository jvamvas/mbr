from unittest import TestCase

from mbr import MBRConfig


class MBRConfigTestCase(TestCase):

    def test_default_config(self):
        config = MBRConfig()
        self.assertEqual(config.num_samples, 10)
        self.assertEqual(config.num_references, 10)

    def test_pruning_schedule(self):
        config = MBRConfig(
            pruning="confidence",
            initial_num_references=8,
            num_references=1024,
        )
        self.assertEqual(config.pruning_schedule, [8, 16, 32, 64, 128, 256, 512, 1024])

    def test_validate(self):
        with self.assertRaises(NotImplementedError):
            MBRConfig(
                pruning="confidence",
                initial_num_references=8,
                num_references=1024,
                output_metric_scores=True,
            )
        with self.assertRaises(ValueError):
            MBRConfig(
                pruning="confidence",
                initial_num_references=1024,
                num_references=8,
            )
