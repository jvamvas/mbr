from unittest import TestCase

from transformers import GPT2LMHeadModel, AutoTokenizer

from mbr import MBR, MBRConfig


class PruningTestCase(TestCase):

    def setUp(self):
        self.model = MBR(GPT2LMHeadModel).from_pretrained("distilgpt2").eval()
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    def test_generate(self):
        mbr_config = MBRConfig(
            pruning="confidence",
            num_samples=8,
            initial_num_references=2,
        )
        input_sentences = [
            "Hello, my name is",
            "This is a test because",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            progress_bar=True,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(2, len(output))
        self.assertTrue(output[0].startswith("Hello, my name is"))
        self.assertTrue(output[1].startswith("This is a test because"))
