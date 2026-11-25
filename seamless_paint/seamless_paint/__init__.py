"""Seamless 2D pattern painting: toroidal and general-lattice kernels."""

from .kernel import SeamlessTile, SeamlessInputError, HAVE_NUMPY
from .lattice import Lattice

# Descriptive aliases for callers/tests that name the oblique tile explicitly.
LatticeTile = SeamlessTile
ObliqueTile = SeamlessTile

__all__ = [
    "SeamlessTile",
    "LatticeTile",
    "ObliqueTile",
    "Lattice",
    "SeamlessInputError",
    "HAVE_NUMPY",
]
__version__ = "2.0.0"
