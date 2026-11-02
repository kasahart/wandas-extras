"""Provisional analysis functions for wandas frames.

Each function takes a wandas frame and leaves the frame unchanged. Functions move into wandas
once their interface has settled, and stay here as thin aliases for a while after that.
"""

from wandas_extras.spectral import spectral_peaks
from wandas_extras.statistics import signal_stats

__all__ = ["signal_stats", "spectral_peaks"]
__version__ = "0.1.0"
