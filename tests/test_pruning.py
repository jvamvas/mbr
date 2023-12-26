from unittest import TestCase

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from transformers.generation import SampleDecoderOnlyOutput

from mbr import PrunedMBRConfig, MBR, MBRConfig, MBROutput, MetricOutput
from mbr.modeling import PrunedMBR


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


class PruningTestCase(TestCase):

    def setUp(self):
        self.model = PrunedMBR(GPT2LMHeadModel).from_pretrained("distilgpt2").eval()
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    def test_generate(self):
        mbr_config = PrunedMBRConfig(
            num_samples=5,
        )
        input_sentences = [
            "Hello, my name is",
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
        self.assertEqual(1, len(output))
        self.assertTrue(output[0].startswith("Hello, my name is"))

    def test_model_output(self):
        mbr_config = PrunedMBRConfig(
            num_samples=5,
            return_dict_in_generate=True,
        )
        input_sentences = [
            "Hello, my name is",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
        )
        self.assertIsInstance(output, MBROutput)
        self.assertEqual(1, output.sequences.shape[0])
        str_output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        self.assertTrue(str_output[0].startswith("Hello, my name is"))
        self.assertIsNone(output.all_samples)
        self.assertIsNone(output.selected_samples_indices)
        self.assertIsNone(output.references)
        self.assertIsNone(output.metric_scores)

    def test_model_output_extended(self):
        mbr_config = PrunedMBRConfig(
            num_samples=5,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=True,
            output_hidden_states=True,
            output_all_samples=True,
            output_metric_scores=True,
        )
        input_sentences = [
            "Hello, my name is",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        output = self.model.generate(
            **encoding,
            mbr_config=mbr_config,
            tokenizer=self.tokenizer,
            do_sample=True,
        )
        self.assertIsInstance(output, MBROutput)
        self.assertEqual(1, output.sequences.shape[0])
        self.assertIsNotNone(output.all_samples)
        self.assertEqual(5, len(output.all_samples))
        self.assertIsInstance(output.all_samples[0], SampleDecoderOnlyOutput)
        self.assertEqual(1, output.all_samples[0].sequences.shape[0])
        self.assertIsNotNone(output.selected_samples_indices)
        self.assertEqual(1, len(output.selected_samples_indices))
        self.assertIsNotNone(output.references)
        self.assertEqual(5, len(output.references))
        self.assertIsInstance(output.references[0], SampleDecoderOnlyOutput)
        self.assertIsNotNone(output.metric_scores)
        self.assertIsInstance(output.metric_scores, MetricOutput)
        self.assertTrue(torch.is_floating_point(output.metric_scores.scores))
        self.assertTrue(torch.is_floating_point(output.metric_scores.scores_per_reference))
        self.assertEqual([1, 5], list(output.metric_scores.scores.shape))
        self.assertEqual([1, 5, 5], list(output.metric_scores.scores_per_reference.shape))

        # Test the model output for a selected sample
        sample = output.all_samples[output.selected_samples_indices[0]]
        if output.sequences[0].shape[0] <= sample.sequences[0].shape[0]:
            torch.testing.assert_close(output.sequences[0], sample.sequences[0][:output.sequences[0].shape[0]])
        else:
            torch.testing.assert_close(output.sequences[0][:sample.sequences[0].shape[0]], sample.sequences[0])
        torch.testing.assert_close(output.sequences[0], sample.sequences[0])
        self.assertIsNotNone(sample.scores)
        self.assertEqual(1, len(sample.scores[0]))
        self.assertIsNotNone(sample.attentions)
        self.assertEqual(1, len(sample.attentions[0][0]))
        self.assertIsNotNone(sample.hidden_states)
        self.assertEqual(1, len(sample.hidden_states[0][0]))
