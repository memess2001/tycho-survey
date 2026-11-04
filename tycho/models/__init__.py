"""Model zoo of the survey objects (SPEC §4.3): :class:`MLP`, :class:`TinyGPT` and :class:`ToyReLU`."""
from tycho.models.mlp import ACTIVATIONS, MLP
from tycho.models.superposition import ToyReLU
from tycho.models.transformer import TinyGPT

__all__ = ["ACTIVATIONS", "MLP", "TinyGPT", "ToyReLU"]
