"""Helpers shared by the modules: read channel data and metadata from a wandas frame."""

from __future__ import annotations

from typing import Any

import numpy as np


def channel_rows(frame: Any) -> np.ndarray:
    """Frame data as a 2-D array (channels, samples). wandas squeezes a single channel to 1-D."""
    return np.atleast_2d(np.asarray(frame.data))


def channel_info(frame: Any) -> list[tuple[str, str, float]]:
    """(label, unit, dB reference) for each channel."""
    return [(ch.label, ch.unit, float(ch.ref)) for ch in frame.channels]
