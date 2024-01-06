import math
from unittest import TestCase

import evaluate

from ..utils import run_all_comet_variants


class AggregatedCometTestCase(TestCase):

    def setUp(self):
        self.comet = evaluate.load("comet", "eamt22-cometinho-da")
        self.inputs = [  # shape: (batch_size,)
            "This is an input sentence.",
            "This is another input sentence.",
        ]
        self.samples = [  # num_samples x batch_size
            ["This is a sample sentence.", "Something totally different."],
            ["This is a sample sentence.", "This a third sample sentence."],
            ["Something totally different.", "This is a fourth sample sentence."],
            ["And finally, this is a fifth sample sentence.", "This is a sample sentence."],
        ]
        self.references = self.samples

    def test_run_all_comet_variants(self):
        translations, durations = run_all_comet_variants(self.comet, self.samples, self.references, self.inputs)
        self.assertEqual(len(translations), math.log2(len(self.samples)))
        self.assertEqual(len(durations), len(translations))
        for translation_list in translations:
            self.assertEqual(len(translation_list), len(self.inputs))
            for i, translation in enumerate(translation_list):
                self.assertIn(translation, [sample[i] for sample in self.samples])
        for i, duration in enumerate(durations):
            self.assertGreater(duration, 0)
            if i > 0:
                self.assertLess(duration, durations[i - 1])
