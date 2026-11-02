"""Provisional analysis functions for wandas frames.

Each function takes a wandas frame and leaves the frame unchanged. Functions move into wandas
once their interface has settled, and stay here as thin aliases for a while after that.
"""

from wandas_extras.diagnostics import envelope_spectrum, spectral_kurtosis
from wandas_extras.modal import modal_parameters
from wandas_extras.rotating import order_spectrum, order_tracks, pulse_times, rpm_profile
from wandas_extras.spectral import spectral_peaks
from wandas_extras.statistics import signal_stats
from wandas_extras.vibration import integrate

__all__ = [
    "envelope_spectrum",
    "integrate",
    "modal_parameters",
    "order_spectrum",
    "order_tracks",
    "pulse_times",
    "rpm_profile",
    "signal_stats",
    "spectral_kurtosis",
    "spectral_peaks",
]
__version__ = "0.1.0"
