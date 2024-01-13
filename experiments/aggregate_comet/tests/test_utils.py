import math
from unittest import TestCase

import evaluate

from ..utils import run_all_comet_factors, run_all_comet_n_by_s, run_all_chrf_factors, run_all_chrf_n_by_s


class UtilsTestCase(TestCase):

    def setUp(self):
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

    def test_run_all_comet_factors(self):
        self.comet = evaluate.load("comet", "eamt22-cometinho-da")
        translations, durations = run_all_comet_factors(self.comet, self.samples, self.references, self.inputs)
        self.assertEqual(len(translations), math.log2(len(self.references)) + 1)
        self.assertEqual(len(durations), len(translations))
        for translation_list in translations:
            self.assertEqual(len(translation_list), len(self.inputs))
            for i, translation in enumerate(translation_list):
                self.assertIn(translation, [sample[i] for sample in self.samples])
        for duration in durations:
            self.assertGreater(duration, 0)

    def test_run_all_comet_n_by_s(self):
        self.comet = evaluate.load("comet", "eamt22-cometinho-da")
        translations, durations = run_all_comet_n_by_s(self.comet, self.samples, self.references, self.inputs)
        self.assertEqual(len(translations), math.log2(len(self.references)) + 1)
        self.assertEqual(len(durations), len(translations))
        for translation_list in translations[:-1]:
            self.assertEqual(len(translation_list), len(self.inputs))
            for i, translation in enumerate(translation_list):
                self.assertIn(translation, [sample[i] for sample in self.samples])
        for duration in durations[:-1]:
            self.assertGreater(duration, 0)

    def test_run_all_chrf_factors(self):
        translations, durations = run_all_chrf_factors(
            list(zip(*self.samples)),
            list(zip(*self.references)),
        )
        self.assertEqual(len(translations), math.log2(len(self.references)) + 1)
        self.assertEqual(len(durations), len(translations))
        for translation_list in translations:
            self.assertEqual(len(translation_list), len(self.inputs))
            for i, translation in enumerate(translation_list):
                self.assertIn(translation, [sample[i] for sample in self.samples])
        for duration in durations:
            self.assertGreater(duration, 0)

    def test_run_all_chrf_n_by_s(self):
        translations, durations = run_all_chrf_n_by_s(
            list(zip(*self.samples)),
            list(zip(*self.references)),
        )
        self.assertEqual(len(translations), math.log2(len(self.references)) + 1)
        self.assertEqual(len(durations), len(translations))
        for translation_list in translations[:-1]:
            self.assertEqual(len(translation_list), len(self.inputs))
            for i, translation in enumerate(translation_list):
                self.assertIn(translation, [sample[i] for sample in self.samples])
        for duration in durations[:-1]:
            self.assertGreater(duration, 0)
