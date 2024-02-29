from unittest import TestCase

from transformers import set_seed, GPT2LMHeadModel, AutoTokenizer, GenerationConfig

from mbr import MBRConfig
from mbr.modeling import PiecewiseMBR


class DecoderOnlyPiecewiseTestCase(TestCase):

    def setUp(self):
        set_seed(42)
        self.model = PiecewiseMBR(GPT2LMHeadModel).from_pretrained("distilgpt2").eval()
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    def test_generate(self):
        mbr_config = MBRConfig(
            num_samples=5,
        )
        generation_config = GenerationConfig.from_pretrained("distilgpt2")
        generation_config.max_length = 20
        input_sentences = [
            "Hello, my name is",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            generation_config=generation_config,
            piece_length=3,
            tokenizer=self.tokenizer,
            do_sample=True,
            progress_bar=True,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        print(output)
        self.assertEqual(1, len(output))
        self.assertTrue(output[0].startswith("Hello, my name is"))
