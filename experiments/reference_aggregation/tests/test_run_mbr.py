from pathlib import Path
from unittest import TestCase

from experiments.reference_aggregation.run_mbr import main


class MBRTestCase(TestCase):

    def setUp(self):
        self.testset = "wmt21"
        self.language_pair = "en-de"
        self.test_dir = Path(__file__).parent / "out"
        self.test_dir.mkdir(exist_ok=True)

    def test_run_mbr_chrf_pairwise(self):
        out_path = main(method="pairwise", topk=None, testset=self.testset, language_pair=self.language_pair, seed_no=0, utility_name="chrf", num_samples=8, epsilon_cutoff=0.02, limit_segments=4, out_dir=self.test_dir)
        self.assertTrue(out_path.exists())
        self.assertIn(self.test_dir, out_path.parents)
        self.assertTrue(out_path.name.endswith(".de"))
        translations = out_path.read_text().splitlines()
        self.assertEqual(len(translations), 4)
        print(translations[0])

    def test_run_mbr_chrf_aggregate(self):
        out_path = main(method="aggregate", topk=None, testset=self.testset, language_pair=self.language_pair, seed_no=0, utility_name="chrf", num_samples=8, epsilon_cutoff=0.02, limit_segments=4, out_dir=self.test_dir)
        translations = out_path.read_text().splitlines()
        self.assertEqual(len(translations), 4)
        print(translations[0])

    def test_run_mbr_chrf_aggregate_to_fine(self):
        out_path = main(method="aggregate_to_fine", topk=2, testset=self.testset, language_pair=self.language_pair, seed_no=0, utility_name="chrf", num_samples=8, epsilon_cutoff=0.02, limit_segments=4, out_dir=self.test_dir)
        translations = out_path.read_text().splitlines()
        self.assertEqual(len(translations), 4)
        print(translations[0])
