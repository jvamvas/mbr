from typing import List

from transformers import __version__ as transformers_version
from transformers.utils import logging, ExplicitEnum

logger = logging.get_logger(__name__)


class PruningStrategy(ExplicitEnum):
    """
    Possible values for the `pruning` parameter of `MBRConfig`.
    """
    CONFIDENCE = "confidence"


class MBRConfig:
    r"""
    Class that holds a configuration for minimum Bayes risk decoding (MBR). Pass this config when calling
    `MBRGenerationMixin.generate()`:

        Example:

        ```python
        >>> config = MBRConfig(num_samples=10, num_references=10, metric="fastchrf")
        >>> model.generate(..., mbr_config=config)
        ```

    The class is inspired by `transformers.GenerationConfig`.
    Note that `MBRConfig` does not control the sampling strategy. Pass separate `GenerationConfig` objects to control
    sampling:

        ```python
        >>> generation_config = GenerationConfig(do_sample=True, num_beams=1, top_p=0.9)
        >>> references_config = GenerationConfig(do_sample=True, num_beams=1, epsilon_cutoff=0.02)
        >>> model.generate(..., mbr_config=config, generation_config=generation_config, references_config=references_config)
        ```

    Arg:
        num_samples (`int`, *optional*, defaults to 10):
            Number of samples generated. 1 means no MBR decoding.
        num_references (`int`, *optional*, defaults to `num_samples`):
            Number of pseudo-references used for MBR decoding.
        metric (`str` or `~evaluate.Metric`, *optional*, defaults to 'fastchrf'):
            Metric used for MBR decoding.
        metric_config_name (`str`, *optional*, defaults to None):
            Metric configuration to pass to `evaluate.load` (e.g., the model for a trained metric, such as
            "eamt22-cometinho-da"). If not specified, the default configuration is used.
        metric_output_field (`str`, *optional*, defaults to 'score'):
            Field of the metric output that is used
        metric_kwargs (optional):
            Additional arguments for the metric's `compute` method. The default MetricRunner requires it to be hashable.
        metric_cache_size (`int`, *optional*, defaults to `num_samples` * `num_references`):
            Size of the cache for the metric. Set to `None` to disable caching (not recommended).
        lower_is_better (`bool`, *optional*, defaults to `False`):
            Set to true if lower metric scores are better (e.g., perplexity).

        > Parameters that define the output variables of `generate`

        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_all_samples (`bool`, *optional*, defaults to `False`):
            Whether or not to return all sampled sequences. See `all_sampled_sequences` under returned tensors for more
            details.
        output_reference_sequences (`bool`, *optional*, defaults to `False`):
            Whether or not to return the reference sequences. See `reference_sequences` under returned tensors for more
            details.
        output_metric_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the metric scores. See `metric_scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        > Parameters that control pruning
        pruning (`str` or `~mbr.generation.configuration_utils.PruningStrategy`, *optional*, defaults to None):
            Pruning strategy to use. Possible values:
                - `None`: No pruning
                - `"confidence"`: Confidence-based pruning
        pruning_alpha (`float`, *optional*, defaults to 0.99):
            Confidence threshold for confidence-based pruning. The lower the value, the more aggressively hypotheses are
            pruned. Needs to be in [0, 1].
        initial_num_references (`int`, *optional*, defaults to min(16, `num_references`)):
            Number of pseudo-references used at the first step of confidence-based pruning. Usually smaller than
            `num_references`; if equal, no pruning is done. Subsequent pruning steps use twice as many references as the
            previous step until `num_references` is reached.
        num_bootstrap_resamples (`int`, *optional*, defaults to 500):
            Number of bootstrap resamples used to estimate the confidence of the metric scores for confidence-based
            pruning.
    """

    def __init__(self, **kwargs):
        # Parameters that control the generation strategy used
        self.num_samples = kwargs.pop("num_samples", 10)
        self.num_references = kwargs.pop("num_references", self.num_samples)
        self.metric = kwargs.pop("metric", "fastchrf")
        self.metric_config_name = kwargs.pop("metric_config_name", None)
        self.metric_output_field = kwargs.pop("metric_output_field", "score")
        self.metric_kwargs = kwargs.pop("metric_kwargs", {})
        self.metric_cache_size = kwargs.pop("metric_cache_size", self.num_samples * self.num_references)
        self.lower_is_better = kwargs.pop("lower_is_better", False)

        # Parameters that define the output variables of `generate`
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_all_samples = kwargs.pop("output_all_samples", False)
        self.output_reference_sequences = kwargs.pop("output_reference_sequences", False)
        self.output_metric_scores = kwargs.pop("output_metric_scores", False)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        # Parameters that control pruning
        pruning = kwargs.pop("pruning", None)
        if isinstance(pruning, str):
            pruning = PruningStrategy(pruning)
        self.pruning = pruning
        self.pruning_alpha = kwargs.pop("pruning_alpha", 0.99)
        self.initial_num_references = kwargs.pop("initial_num_references", min(16, self.num_references))
        self.num_bootstrap_resamples = kwargs.pop("num_bootstrap_resamples", 500)
        self.validate(is_init=False)

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", transformers_version)
        import mbr
        self.mbr_version = kwargs.pop("mbr_version", mbr.__version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing an `MBRConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        """
        Validates the values of the attributes of the [`GenerationConfig`] instance. Raises exceptions in the presence
        of parameterization that can be detected as incorrect from the configuration instance alone.

        Note that some parameters are best validated at generate runtime, as they may depend on other inputs and/or the
        model, such as parameters related to the generation length.
        """
        if self.metric_cache_size <= 0:
            raise ValueError(f"`metric_cache_size` ({self.metric_cache_size}) must be greater than 0.")
        if self.pruning == PruningStrategy.CONFIDENCE:
            if self.initial_num_references > self.num_references:
                raise ValueError(
                    f"`initial_num_references` ({self.initial_num_references}) must be smaller than or equal to "
                    f"`num_references` ({self.num_references})."
                )
            if self.output_metric_scores and len(self.pruning_schedule) > 1:
                raise NotImplementedError("Pruning does not support output_metric_scores=True, since not all metric "
                                          "scores are calculated.")

    @property
    def pruning_schedule(self) -> List[int]:
        """
        Confidence-based pruning: Returns the number of references used at each pruning iteration.
        """
        schedule = [self.initial_num_references]
        while 2 * schedule[-1] <= self.num_references:
            schedule.append(2 * schedule[-1])
        if schedule[-1] != self.num_references:
            schedule.append(self.num_references)
        return schedule
