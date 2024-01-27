from unittest import TestCase


class BeamSearchTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"

    def test_beam_search(self):
        from experiments.reference_aggregation.baseline_beam_search import main
        out_path = main(self.testset, self.language_pair, limit_segments=4)
        self.assertTrue(out_path.exists())
        translations = out_path.read_text().splitlines()
        self.assertEqual(len(translations), 4)
        print(translations[0])
