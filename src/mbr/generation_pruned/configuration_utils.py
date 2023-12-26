from typing import List

from transformers.utils import logging

from mbr import MBRConfig

logger = logging.get_logger(__name__)


class PrunedMBRConfig(MBRConfig):
    r"""
    Class that holds a configuration for minimum Bayes risk decoding (MBR) with confidence-based pruning. Pass this
    config when calling`PrunedMBRGenerationMixin.generate()`:

    Arg:
        > Parameters that control confidence-based pruning

        pruning_alpha (`float`, *optional*, defaults to 0.99):
            Confidence threshold for pruning. The lower the value, the more aggressively hypotheses are pruned.
            Needs to be in [0, 1].
        initial_num_references (`int`, *optional*, defaults to min(16, `num_references`)):
            Number of pseudo-references used at the first pruning step. Usually smaller than `num_references`; if equal,
            no pruning is done. Subsequent pruning steps use twice as many references as the previous step until
            `num_references` is reached.
        num_bootstrap_resamples (`int`, *optional*, defaults to 500):
            Number of bootstrap resamples used to estimate the confidence of the metric scores during pruning.

        > Other parameters: See parent class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pruning_alpha = kwargs.pop("pruning_alpha", 0.99)
        self.initial_num_references = kwargs.pop("initial_num_references", min(16, self.num_references))
        self.num_bootstrap_resamples = kwargs.pop("num_bootstrap_resamples", 500)
        self.validate(is_init=True)

    @property
    def schedule(self) -> List[int]:
        schedule = [self.initial_num_references]
        while 2 * schedule[-1] <= self.num_references:
            schedule.append(2 * schedule[-1])
        if schedule[-1] != self.num_references:
            schedule.append(self.num_references)
        return schedule

    def validate(self, is_init=False):
        super().validate(is_init=is_init)
        if self.initial_num_references > self.num_references:
            raise ValueError(
                f"`initial_num_references` ({self.initial_num_references}) must be smaller than or equal to "
                f"`num_references` ({self.num_references})."
            )
