from unittest import TestCase

from experiments.aggregate_comet.gemba import CREDENTIALS
from experiments.aggregate_comet.gemba.cache import Cache
from experiments.aggregate_comet.gemba.gpt_api import GptApi
from experiments.aggregate_comet.gemba.prompt import prompts


class GembaTestCase(TestCase):

    def setUp(self):
        # self.model = "babbage-002"
        # self.model = "gpt-3.5-turbo-1106"
        self.model = "gpt-4-1106-preview"
        self.cache = Cache(f"{self.model}.jsonl")
        self.gptapi = GptApi(CREDENTIALS.credentials)
        self.gemba_type = "GEMBA-DA_ref"

    def test_score(self):
        source = "Dies ist die perfekte Übersetzung."
        reference = "This is the perfect translation."
        hypothesis1 = "This is a near-perfect translation."
        hypothesis2 = "I've never seen a worse translation."
        data1 = {
            "source_seg": source,
            "target_seg": hypothesis1,
            "reference_seg": reference,
            "source_lang": "German",
            "target_lang": "English",
        }
        prompt1 = prompts[self.gemba_type]["prompt"].format(**data1)
        print(prompt1)
        parsed_answers1 = self.gptapi.request(prompt1, self.model, prompts[self.gemba_type]["validate_answer"], cache=self.cache)
        print(parsed_answers1)
        score1 = parsed_answers1[0]["answer"]
        data2 = {
            "source_seg": source,
            "target_seg": hypothesis2,
            "reference_seg": reference,
            "source_lang": "German",
            "target_lang": "English",
        }
        prompt2 = prompts[self.gemba_type]["prompt"].format(**data2)
        parsed_answers2 = self.gptapi.request(prompt2, self.model, prompts[self.gemba_type]["validate_answer"], cache=self.cache)
        print(parsed_answers2)
        score2 = parsed_answers2[0]["answer"]
        self.assertGreater(score1, score2)
