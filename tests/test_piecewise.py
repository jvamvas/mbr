import os
import unittest
from unittest import TestCase

from transformers import set_seed, GPT2LMHeadModel, AutoTokenizer, GenerationConfig, M2M100ForConditionalGeneration

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
            verbose=True,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        print(output)
        self.assertEqual(1, len(output))
        self.assertTrue(output[0].startswith("Hello, my name is"))


@unittest.skipIf(os.getenv("SKIP_SLOW_TESTS", False), "Requires extra dependencies")
class EncoderDecoderPiecewiseTestCase(TestCase):

    def setUp(self):
        set_seed(42)
        self.model = PiecewiseMBR(M2M100ForConditionalGeneration).from_pretrained("alirezamsh/small100").eval()
        self.tokenizer = AutoTokenizer.from_pretrained("alirezamsh/small100")
        self.tokenizer.tgt_lang = "fr"

    def test_generate(self):
        mbr_config = MBRConfig(
            num_samples=5,
        )
        input_sentences = [
            "Could you translate this for me, please?",
            "This is another sentence.",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt", padding=True)
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            piece_length=3,
            tokenizer=self.tokenizer,
            do_sample=True,
            num_beams=1,
            verbose=True,
        )
        translations = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        print(translations)
        self.assertEqual(2, len(translations))
