"""Vibration analysis on wandas ChannelFrame."""

from __future__ import annotations

from typing import Any

import numpy as np
from wandas.core.metadata import ChannelCalibration

# SI units along the chain acceleration -> velocity -> displacement.
_SI_CHAIN = ("m/s^2", "m/s", "m")


def _integrate_rfft(data: np.ndarray, fs: float, order: int, fmin: float | None, fmax: float | None) -> np.ndarray:
    n = data.shape[-1]
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = np.fft.rfft(data, axis=-1)
    band = f > 0 if order > 0 else np.ones_like(f, dtype=bool)
    if fmin is not None:
        band &= f >= fmin
    if fmax is not None:
        band &= f <= fmax
    out = np.zeros_like(spec)
    jw = 2j * np.pi * f[band]
    out[..., band] = spec[..., band] * jw ** (-order)
    return np.fft.irfft(out, n, axis=-1).astype(data.dtype, copy=False)


def _output_unit(unit: str, order: int) -> str | None:
    if unit not in _SI_CHAIN:
        return None
    i = _SI_CHAIN.index(unit) + order
    return _SI_CHAIN[i] if 0 <= i < len(_SI_CHAIN) else None


def integrate(
    frame: Any,
    order: int = 1,
    fmin: float | None = None,
    fmax: float | None = None,
    unit: str | None = None,
) -> Any:
    """Integrate (or differentiate) each channel in the frequency domain, limited to a band.

    The spectrum of the whole frame is divided by ``(j 2 pi f) ** order`` and bins outside
    ``[fmin, fmax]`` are set to zero, so the result is periodic over the frame. Integration
    always removes 0 Hz. Typical use: acceleration -> velocity with ``order=1`` and the
    evaluation band of a standard, then ``.rms`` for the band RMS.

    Args:
        frame: A wandas ChannelFrame. Channel calibration is applied first.
        order: Number of integrations; negative values differentiate.
        fmin, fmax: Band kept in the result [Hz]. ``None`` keeps everything on that side.
        unit: Unit of the result. When omitted, SI units are moved along
            ``m/s^2 -> m/s -> m``; other input units need ``unit``.

    Returns:
        A new ChannelFrame with the integrated waveform and the result unit.
    """
    if not isinstance(order, (int, np.integer)) or isinstance(order, bool) or order == 0:
        raise ValueError(f"order must be a non-zero integer, got {order!r}")
    units = []
    for ch in frame.channels:
        u = unit or _output_unit(ch.unit, int(order))
        if u is None:
            raise ValueError(f"cannot infer the result unit from {ch.unit!r} (channel {ch.label!r}); pass unit=")
        units.append(u)
    out = frame.apply(_integrate_rfft, fs=float(frame.sampling_rate), order=int(order), fmin=fmin, fmax=fmax)
    return out.with_calibration([ChannelCalibration(factor=1.0, unit=u) for u in units])
