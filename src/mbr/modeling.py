from transformers import GenerationMixin

from mbr.generation.utils import MBRGenerationMixin
from mbr.generation_piecewise.utils import PiecewiseMBRGenerationMixin


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


def PiecewiseMBR(model_class: type) -> type:
    """
    Utility function for converting a model class into a class that inherits from `~generation.MBRGenerationMixin`.
    """
    if not issubclass(model_class, GenerationMixin):
        raise ValueError(
            f"PiecewiseMBR() can only be applied to classes that inherit from `transformers.GenerationMixin`, "
            f"but got {model_class}."
        )
    return type("PiecewiseMBR" + model_class.__name__, (PiecewiseMBRGenerationMixin, model_class), {})
