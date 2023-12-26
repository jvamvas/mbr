from transformers import GenerationMixin

from mbr.generation.utils import MBRGenerationMixin
from mbr.generation_pruned.utils import PrunedMBRGenerationMixin


def MBR(model_class: type) -> type:
    """
    Utility function for converting a model class into a class that inherits from `~generation.MBRGenerationMixin`.
    """
    if not issubclass(model_class, GenerationMixin):
        raise ValueError(
            f"MBR() can only be applied to classes that inherit from `transformers.GenerationMixin`, "
            f"but got {model_class}."
        )
    return type("MBR" + model_class.__name__, (MBRGenerationMixin, model_class), {})


def PrunedMBR(model_class: type) -> type:
    """
    Utility function for converting a model class into a class that inherits from
    `~generation_pruned.PrunedMBRGenerationMixin`.
    """
    if not issubclass(model_class, GenerationMixin):
        raise ValueError(
            f"PrunedMBR() can only be applied to classes that inherit from `transformers.GenerationMixin`, "
            f"but got {model_class}."
        )
    return type("PrunedMBR" + model_class.__name__, (PrunedMBRGenerationMixin, model_class), {})
