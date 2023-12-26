from unittest import TestCase

import torch.testing
from transformers import GPT2LMHeadModel, AutoTokenizer, set_seed

from mbr import MBR, MBRConfig, MBROutput


class PruningTestCase(TestCase):

    def setUp(self):
        set_seed(42)
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

    def test_alpha_one_equivalent_to_no_pruning(self):
        """
        alpha == 1 means no pruning, so the output should be the same as without pruning.
        """
        input_sentences = [
            "Hello, my name is",
            "This is a test because",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        pruning_config = MBRConfig(
            pruning="confidence",
            pruning_alpha=1.,
            num_samples=8,
            initial_num_references=2,
            return_dict_in_generate=True,
            output_all_samples=True,
            output_reference_sequences=True,
        )
        set_seed(42)
        pruning_output: MBROutput = self.model.generate(
            **encoding,
            mbr_config=pruning_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            progress_bar=True,
        )
        standard_config = MBRConfig(
            num_samples=8,
            return_dict_in_generate=True,
            output_all_samples=True,
            output_reference_sequences=True,
        )
        set_seed(42)
        standard_output: MBROutput = self.model.generate(
            **encoding,
            mbr_config=standard_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            progress_bar=True,
        )
        torch.testing.assert_close(pruning_output.sequences, standard_output.sequences)
        torch.testing.assert_close(pruning_output.all_samples, standard_output.all_samples)
        torch.testing.assert_close(pruning_output.selected_samples_indices, standard_output.selected_samples_indices)
        torch.testing.assert_close(pruning_output.references, standard_output.references)

    def test_no_schedule_equivalent_to_no_pruning(self):
        """
        initial_num_references == num_references means no pruning, so the output should be the same as without pruning.
        """
        input_sentences = [
            "Hello, my name is",
            "This is a test because",
        ]
        encoding = self.tokenizer(input_sentences, return_tensors="pt")
        pruning_config = MBRConfig(
            pruning="confidence",
            num_samples=8,
            initial_num_references=8,
            num_references=8,
            return_dict_in_generate=True,
            output_all_samples=True,
            output_reference_sequences=True,
            output_metric_scores=True,
        )
        self.assertEqual(1, len(pruning_config.pruning_schedule))
        set_seed(42)
        pruning_output: MBROutput = self.model.generate(
            **encoding,
            mbr_config=pruning_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            progress_bar=True,
        )
        standard_config = MBRConfig(
            num_samples=8,
            return_dict_in_generate=True,
            output_all_samples=True,
            output_reference_sequences=True,
            output_metric_scores=True,
        )
        set_seed(42)
        standard_output: MBROutput = self.model.generate(
            **encoding,
            mbr_config=standard_config,
            tokenizer=self.tokenizer,
            do_sample=True,
            progress_bar=True,
        )
        torch.testing.assert_close(pruning_output.sequences, standard_output.sequences)
        torch.testing.assert_close(pruning_output.all_samples, standard_output.all_samples)
        torch.testing.assert_close(pruning_output.selected_samples_indices, standard_output.selected_samples_indices)
        torch.testing.assert_close(pruning_output.references, standard_output.references)
        torch.testing.assert_close(pruning_output.metric_scores.scores, standard_output.metric_scores.scores)
        torch.testing.assert_close(pruning_output.metric_scores.scores_per_reference, standard_output.metric_scores.scores_per_reference)
