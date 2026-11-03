"""Provisional analyses for wandas frames.

New analyses are methods of :class:`ExtrasFrame`, a ChannelFrame subclass, so they keep their
name and arguments when they move into wandas. The remaining functions take a wandas frame,
leave it unchanged, and are being moved onto ExtrasFrame one by one.
"""

from wandas_extras.diagnostics import envelope_spectrum, spectral_kurtosis
from wandas_extras.frame import ExtrasFrame, recipe_registry
from wandas_extras.modal import modal_parameters
from wandas_extras.rotating import order_spectrum, order_tracks
from wandas_extras.spectral import spectral_peaks
from wandas_extras.statistics import signal_stats
from wandas_extras.vibration import integrate

__all__ = [
    "ExtrasFrame",
    "envelope_spectrum",
    "integrate",
    "modal_parameters",
    "order_spectrum",
    "order_tracks",
    "signal_stats",
    "spectral_kurtosis",
    "spectral_peaks",
    "recipe_registry",
]
__version__ = "0.1.0"
