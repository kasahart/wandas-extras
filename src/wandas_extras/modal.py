"""Modal parameters (natural frequency and damping ratio) from a frequency response function."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import optimize
from scipy import signal as sps

# Power of (j 2 pi f) that turns each FRF type into receptance (displacement / force).
_TO_RECEPTANCE = {"receptance": 0, "mobility": 1, "inertance": 2}


def _half_power(f: np.ndarray, mag: np.ndarray, i: int) -> float:
    half = mag[i] / np.sqrt(2)
    lo, hi = i, i
    while lo > 0 and mag[lo] > half:
        lo -= 1
    while hi < len(mag) - 1 and mag[hi] > half:
        hi += 1
    fl = np.interp(half, [mag[lo], mag[lo + 1]], [f[lo], f[lo + 1]])
    fh = np.interp(half, [mag[hi], mag[hi - 1]], [f[hi], f[hi - 1]])
    return float((fh - fl) / (2 * f[i]))


def _sdof_fit(f: np.ndarray, h: np.ndarray, fr0: float, z0: float, lo: float, hi: float) -> tuple[float, float]:
    """Fit one mode plus residual terms (stiffness 1/K and mass -1/(M w^2)) to receptance in [lo, hi]."""
    m = (f >= lo) & (f <= hi) & (f > 0)  # the mass residual diverges at 0 Hz
    if m.sum() < 4:  # too few points: keep the half-power estimate
        return fr0, z0
    w, hb = 2 * np.pi * f[m], h[m]
    wr0 = 2 * np.pi * fr0
    a0 = hb[np.argmin(np.abs(f[m] - fr0))] * (2j * z0 * wr0**2)

    def model(p: np.ndarray) -> np.ndarray:
        return (
            (p[2] + 1j * p[3]) / (p[0] ** 2 - w**2 + 2j * p[1] * p[0] * w)
            + (p[4] + 1j * p[5])
            + (p[6] + 1j * p[7]) / w**2
        )

    def res(p: np.ndarray) -> np.ndarray:
        r = (model(p) - hb) / np.max(np.abs(hb))
        return np.concatenate([r.real, r.imag])

    sol = optimize.least_squares(res, [wr0, z0, a0.real, a0.imag, 0.0, 0.0, 0.0, 0.0], x_scale="jac")
    return float(sol.x[0] / (2 * np.pi)), float(abs(sol.x[1]))


def modal_parameters(
    frf: Any,
    frf_type: Literal["receptance", "mobility", "inertance"],
    method: Literal["half_power", "sdof_fit"] = "half_power",
    coherence: Any = None,
    fmin: float | None = None,
    fmax: float | None = None,
    min_prominence_db: float = 6.0,
    bands_hz: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Natural frequencies and damping ratios of the modes of a driving-point FRF with light damping.

    Peaks of ``|H|`` are found with ``scipy.signal.find_peaks`` (or given as bands). The damping ratio
    comes from the half-power bandwidth. ``method="sdof_fit"`` then fits one mode plus residual
    terms to the receptance in ``fr * (1 +- 3 zeta)`` (or the given band) by complex least squares.

    Args:
        frf: A one-channel frame with complex FRF values and ``freqs``, such as one channel of
            ``frame.transfer_function()``.
        frf_type: What the FRF is: ``receptance`` (displacement/force), ``mobility``
            (velocity/force) or ``inertance`` (acceleration/force). Used to convert to receptance
            for the fit.
        method: ``half_power`` or ``sdof_fit``.
        coherence: Optional one-channel frame with the coherence on the same frequencies; its value
            at each peak is reported.
        fmin, fmax: Frequency range for peak search [Hz].
        min_prominence_db: Minimum prominence of a peak of ``20 log10 |H|`` [dB].
        bands_hz: Bands ``[(lo, hi), ...]``, one per mode, instead of peak search.

    Returns:
        One dict per mode with ``freq_hz``, ``damping_ratio``, ``peak_magnitude`` and
        ``coherence_at_peak`` (None without ``coherence``).
    """
    if method not in ("half_power", "sdof_fit"):
        raise ValueError(f"method must be 'half_power' or 'sdof_fit', got {method!r}")
    if frf_type not in _TO_RECEPTANCE:
        raise ValueError(f"frf_type must be one of {list(_TO_RECEPTANCE)}, got {frf_type!r}")
    f = np.asarray(frf.freqs, dtype=float)
    h = np.asarray(frf.data).reshape(-1)
    if h.shape != f.shape:
        raise ValueError("frf must have exactly one channel; use frame.get_channel(...)")
    coh = None if coherence is None else np.asarray(coherence.data).reshape(-1).real
    mag = np.abs(h)
    if bands_hz:
        idx = []
        for lo, hi in bands_hz:
            band = np.flatnonzero((f >= lo) & (f <= hi))
            idx.append(int(band[np.argmax(mag[band])]))
    else:
        sel = np.ones_like(f, dtype=bool)
        if fmin is not None:
            sel &= f >= fmin
        if fmax is not None:
            sel &= f <= fmax
        i_sel = np.flatnonzero(sel)
        with np.errstate(divide="ignore"):
            pk, _ = sps.find_peaks(20 * np.log10(mag[i_sel]), prominence=min_prominence_db)
        idx = [int(i) for i in i_sel[pk]]
    k = _TO_RECEPTANCE[frf_type]
    with np.errstate(divide="ignore", invalid="ignore"):
        hr = h / (2j * np.pi * f) ** k
    modes = []
    for j, i in enumerate(idx):
        fr, z = float(f[i]), _half_power(f, mag, i)
        if method == "sdof_fit":
            lo, hi = bands_hz[j] if bands_hz else (fr * (1 - 3 * z), fr * (1 + 3 * z))
            fr, z = _sdof_fit(f, hr, fr, max(z, 1e-3), lo, hi)
        modes.append(
            {
                "freq_hz": fr,
                "damping_ratio": z,
                "peak_magnitude": float(mag[i]),
                "coherence_at_peak": None if coh is None else float(coh[i]),
            }
        )
    return modes
