from unittest import TestCase

from mbr import PrunedMBRConfig


class PrunedMBRConfigTestCase(TestCase):

    def test_default_config(self):
        config = PrunedMBRConfig()
        self.assertEqual(config.num_samples, 10)
        self.assertEqual(config.num_references, 10)
        self.assertEqual(config.pruning_alpha, 0.99)

    def test_schedule(self):
        config = PrunedMBRConfig(
            initial_num_references=8,
            num_references=1024,
        )
        self.assertEqual(config.schedule, [8, 16, 32, 64, 128, 256, 512, 1024])
