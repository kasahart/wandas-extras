"""Waveform statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kurtosis

from wandas_extras._frame import channel_info, channel_rows


def signal_stats(frame: Any, full_scale: float | None = None, histogram_bins: int = 50) -> list[dict[str, Any]]:
    """Statistics of each channel of a time-domain frame.

    Args:
        frame: A wandas ChannelFrame. Trim it first (``frame.trim(start, end)``) to analyse a segment.
        full_scale: Full-scale value of the recording in the channel unit (for example the
            calibrated value of digital full scale). When given, the result also reports how close
            the waveform comes to full scale and an amplitude histogram over [-full_scale, full_scale].
        histogram_bins: Number of histogram bins when ``full_scale`` is given.

    Returns:
        One dict per channel with ``label``, ``unit``, ``rms``, ``peak``, ``mean``, ``crest_factor``
        (None for a silent channel), ``kurtosis`` (Pearson, 3 for a Gaussian), ``rms_db``, ``peak_db``
        and ``db_ref`` (the channel's dB reference). With ``full_scale`` it also has ``full_scale``,
        ``peak_over_full_scale``, ``fraction_near_full_scale`` (share of samples with
        ``|x| >= 0.999 * full_scale``) and ``histogram`` (``edges``, ``counts``).
    """
    if full_scale is not None and full_scale <= 0:
        raise ValueError(f"full_scale must be positive, got {full_scale!r}")
    out = []
    for x, (label, unit, ref) in zip(channel_rows(frame), channel_info(frame)):
        x = np.asarray(x, dtype=float)
        rms = float(np.sqrt(np.mean(x**2)))
        peak = float(np.max(np.abs(x)))
        with np.errstate(divide="ignore"):
            stats: dict[str, Any] = {
                "label": label,
                "unit": unit,
                "rms": rms,
                "peak": peak,
                "mean": float(np.mean(x)),
                "crest_factor": peak / rms if rms > 0 else None,
                "kurtosis": float(kurtosis(x, fisher=False)),
                "rms_db": float(20 * np.log10(rms / ref)),
                "peak_db": float(20 * np.log10(peak / ref)),
                "db_ref": ref,
            }
        if full_scale is not None:
            counts, edges = np.histogram(x, bins=histogram_bins, range=(-full_scale, full_scale))
            stats.update(
                {
                    "full_scale": float(full_scale),
                    "peak_over_full_scale": peak / full_scale,
                    "fraction_near_full_scale": float(np.mean(np.abs(x) >= 0.999 * full_scale)),
                    "histogram": {"edges": edges, "counts": counts},
                }
            )
        out.append(stats)
    return out
