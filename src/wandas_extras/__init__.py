"""Provisional analyses for wandas frames.

New analyses are methods of :class:`ExtrasFrame`, a ChannelFrame subclass, so they keep their
name and arguments when they move into wandas. The remaining functions take a wandas frame,
leave it unchanged, and are being moved onto ExtrasFrame one by one. Analyses that wandas now
has (peak, kurtosis, find_peaks, spectral_kurtosis, integrate, envelope) are not repeated here.
"""

from wandas_extras.frame import AngleFrame, ExtrasFrame, OrderSpectralFrame, recipe_registry
from wandas_extras.modal import modal_parameters

__all__ = [
    "AngleFrame",
    "ExtrasFrame",
    "OrderSpectralFrame",
    "modal_parameters",
    "recipe_registry",
]
__version__ = "0.1.0"
