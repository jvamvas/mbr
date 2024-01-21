import os
from copy import deepcopy
from unittest import TestCase

from gemba import gemba, GembaMetric

from lmql.runtime.bopenai import get_stats


class GembaTestCase(TestCase):

    def setUp(self):
        os.environ["GEMBA_CACHE"] = "/tmp/gemba"

    def test_lmql(self):
        score1 = gemba(
            src="Dies ist die perfekte Übersetzung.",
            ref="This is the perfect translation.",
            hyp="This is a near-perfect translation.",
            src_lang="German",
            tgt_lang="English",
        )
        score2 = gemba(
            src="Dies ist die perfekte Übersetzung.",
            ref="This is the perfect translation.",
            hyp="I've never seen a worse translation.",
            src_lang="German",
            tgt_lang="English",
        )
        self.assertGreater(score1, score2)
        print(get_stats())

    def test_class(self):
        metric = GembaMetric()
        score1 = metric.score(
            src="Dies ist die perfekte Übersetzung.",
            ref="This is the perfect translation.",
            hyp="This is a near-perfect translation.",
            src_lang="German",
            tgt_lang="English",
        )
        score2 = metric.score(
            src="Dies ist die perfekte Übersetzung.",
            ref="This is the perfect translation.",
            hyp="I've never seen a worse translation.",
            src_lang="German",
            tgt_lang="English",
        )
        self.assertGreater(score1, score2)
        stats_before_cache = deepcopy(get_stats())
        print(stats_before_cache)
        score3 = metric.score(
            src="Dies ist die perfekte Übersetzung.",
            ref="This is the perfect translation.",
            hyp="This is a near-perfect translation.",
            src_lang="German",
            tgt_lang="English",
        )
        self.assertEqual(score1, score3)
        stats_after_cache = deepcopy(get_stats())
        self.assertEqual(stats_before_cache, stats_after_cache)

        os.remove(metric.cache_path)
        score4 = metric.score(
            src="Dies ist die perfekte Übersetzung.",
            ref="This is the perfect translation.",
            hyp="This is a near-perfect translation.",
            src_lang="German",
            tgt_lang="English",
        )
        self.assertEqual(score1, score4)
        stats_no_cache = deepcopy(get_stats())
        self.assertNotEqual(stats_before_cache, stats_no_cache)
