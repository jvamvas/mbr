from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.utils import logging, ModelOutput

from mbr import MBRGenerationMixin, MBRConfig, MBROutput
from mbr.metrics.base import MetricRunner

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


@dataclass
class PiecewiseMBROutput(ModelOutput):
    """
    Base class for outputs of generation models when using MBR decoding.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
    """

    sequences: torch.LongTensor = None


class PiecewiseMBRGenerationMixin(MBRGenerationMixin):

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            references_config: Optional[GenerationConfig] = None,
            mbr_config: Optional[MBRConfig] = None,
            piece_length: int = 5,
            tokenizer: Optional["PreTrainedTokenizer"] = None,
            metric_runner: Optional[MetricRunner] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            progress_bar: bool = False,
            verbose: bool = False,
            **kwargs,
    ) -> Union[PiecewiseMBROutput, torch.LongTensor]:
        final_output = PiecewiseMBROutput()

        piece_config = deepcopy(mbr_config)
        piece_config.return_dict_in_generate = True
        piece_config.output_all_samples = verbose

        piece_inputs = deepcopy(inputs)
        piece_kwargs = deepcopy(kwargs)
        del piece_kwargs["attention_mask"]

        i = 0
        while True:
            piece_output: MBROutput = super().generate(
                inputs=piece_inputs,
                generation_config=generation_config,
                references_config=references_config,
                mbr_config=piece_config,
                tokenizer=tokenizer,
                metric_runner=metric_runner,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                assistant_model=assistant_model,
                streamer=streamer,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                progress_bar=progress_bar,
                max_new_tokens=piece_length,
                **piece_kwargs,
            )

            if verbose:
                print(f"Iteration {i}:")
                for batch_idx in range(piece_output.all_samples[0].sequences.shape[0]):
                    for sample_idx in range(len(piece_output.all_samples)):
                        print(tokenizer.decode(piece_output.all_samples[sample_idx].sequences[batch_idx].tolist()))
                    print()

            final_output.sequences = piece_output.sequences

            # Update inputs for next iteration
            if not self.config.is_encoder_decoder:
                if piece_inputs is not None:
                    piece_inputs = piece_output.sequences.to(piece_inputs.device)
                if piece_kwargs.get("input_ids", None) is not None:
                    piece_kwargs["input_ids"] = piece_output.sequences.to(piece_kwargs["input_ids"].device)
            else:
                if "decoder_input_ids" in piece_kwargs:
                    piece_kwargs["decoder_input_ids"] = piece_output.sequences.to(piece_kwargs["decoder_input_ids"].device)
                else:
                    piece_kwargs["decoder_input_ids"] = piece_output.sequences
            # If maximum length is reached, stop
            if generation_config is not None and generation_config.max_length is not None and generation_config.max_length <= final_output.sequences.shape[1]:
                break

            # If all sequences have the eos token, stop
            if tokenizer.eos_token_id and all(tokenizer.eos_token_id in sequence for sequence in final_output.sequences[:, -piece_length:]):
                # Replace everything after the last eos token with padding
                for batch_idx in range(final_output.sequences.shape[0]):
                    for seq_idx in range(1, final_output.sequences.shape[1]):
                        if final_output.sequences[batch_idx, seq_idx] == tokenizer.eos_token_id:
                            final_output.sequences[batch_idx, seq_idx + 1:] = tokenizer.pad_token_id
                            break
                break

            i += 1

        if mbr_config.return_dict_in_generate:
            return final_output
        else:
            return final_output.sequences
