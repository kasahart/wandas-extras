"""Spectrum analysis on wandas SpectralFrame."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.signal import find_peaks

from wandas_extras._frame import channel_info


def spectral_peaks(
    spectrum: Any,
    min_prominence_db: float = 10.0,
    max_peaks: int = 20,
    fmin: float | None = None,
    fmax: float | None = None,
    interpolation: Literal["none", "parabolic"] = "none",
) -> list[dict[str, Any]]:
    """Peaks of each channel of a spectrum.

    Peaks are found on the level in dB (``spectrum.dB``, which uses the channel's reference) with
    ``scipy.signal.find_peaks``. The amplitude has the scaling of the input spectrum (for example
    ``frame.welch()`` gives the peak amplitude of a sinusoid).

    Args:
        spectrum: A wandas SpectralFrame, such as ``frame.welch()`` or ``frame.fft()``.
        min_prominence_db: Minimum prominence of a peak over its surroundings [dB].
        max_peaks: Maximum number of peaks per channel, highest levels first.
        fmin, fmax: Frequency range of the peaks to keep [Hz].
        interpolation: ``"none"`` returns the bin values. ``"parabolic"`` fits a parabola to the
            level in dB at the peak bin and its two neighbours, and corrects the frequency and level.

    Returns:
        One dict per channel with ``label``, ``unit``, ``db_ref``, ``df_hz`` and ``peaks``, a list of
        ``{freq_hz, amplitude, level_db, prominence_db}`` sorted by frequency.
    """
    if interpolation not in ("none", "parabolic"):
        raise ValueError(f"interpolation must be 'none' or 'parabolic', got {interpolation!r}")
    freqs = np.asarray(spectrum.freqs, dtype=float)
    df = float(freqs[1] - freqs[0])
    levels = np.atleast_2d(np.asarray(spectrum.dB, dtype=float))
    out = []
    for lv, (label, unit, ref) in zip(levels, channel_info(spectrum)):
        idx, prop = find_peaks(lv, prominence=min_prominence_db)
        prom = prop["prominences"]
        keep = np.ones(len(idx), dtype=bool)
        if fmin is not None:
            keep &= freqs[idx] >= fmin
        if fmax is not None:
            keep &= freqs[idx] <= fmax
        idx, prom = idx[keep], prom[keep]
        order = np.argsort(lv[idx])[::-1][:max_peaks]
        peaks = []
        for i, pr in zip(idx[order], prom[order]):
            f, level = freqs[i], lv[i]
            if interpolation == "parabolic" and 0 < i < len(freqs) - 1:
                a, b, c = lv[i - 1], lv[i], lv[i + 1]
                d = 0.5 * (a - c) / (a - 2 * b + c)
                f, level = f + d * df, b - 0.25 * (a - c) * d
            peaks.append(
                {
                    "freq_hz": float(f),
                    "amplitude": float(ref * 10 ** (level / 20)),
                    "level_db": float(level),
                    "prominence_db": float(pr),
                }
            )
        peaks.sort(key=lambda p: p["freq_hz"])
        out.append({"label": label, "unit": unit, "db_ref": ref, "df_hz": df, "peaks": peaks})
    return out
