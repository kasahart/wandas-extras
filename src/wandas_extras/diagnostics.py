"""Fault diagnosis: spectral kurtosis and envelope spectrum (bearing and gear faults)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

from wandas_extras._frame import channel_info, channel_rows


def spectral_kurtosis(
    frame: Any,
    n_fft: int = 256,
    overlap: float = 0.75,
    fmin: float | None = None,
    fmax: float | None = None,
) -> list[dict[str, Any]]:
    """Spectral kurtosis from the STFT (``frame.stft``), to find the band where impulsive content is strongest.

    ``SK(f) = <|X|^4> / <|X|^2>^2 - 2``, which is 0 for Gaussian noise. The 0 Hz and Nyquist bins
    are left out because they are real-valued and have a different reference value.

    Args:
        frame: A wandas ChannelFrame.
        n_fft: Frame length of the STFT. Short frames respond better to impulses.
        overlap: Overlap of the STFT frames (0 to 1).
        fmin, fmax: Frequency range to return [Hz].

    Returns:
        One dict per channel with ``label``, ``freq_hz``, ``kurtosis``, ``max_freq_hz`` and ``df_hz``.
    """
    hop = n_fft - int(n_fft * overlap)
    spec = frame.stft(n_fft=n_fft, hop_length=hop)
    freqs = np.asarray(spec.freqs, dtype=float)
    z = np.asarray(spec.data)
    z = z.reshape(-1, len(freqs), z.shape[-1])
    a2 = np.abs(z) ** 2
    sk = np.mean(a2**2, axis=-1) / np.mean(a2, axis=-1) ** 2 - 2
    m = (freqs > 0) & (freqs < float(frame.sampling_rate) / 2)
    if fmin is not None:
        m &= freqs >= fmin
    if fmax is not None:
        m &= freqs <= fmax
    out = []
    for k, (label, _, _) in zip(sk, channel_info(frame)):
        out.append(
            {
                "label": label,
                "freq_hz": freqs[m],
                "kurtosis": k[m],
                "max_freq_hz": float(freqs[m][np.argmax(k[m])]),
                "df_hz": float(frame.sampling_rate) / n_fft,
            }
        )
    return out


def envelope_spectrum(
    frame: Any,
    band_hz: tuple[float, float],
    filter_order: int = 6,
    window: str = "hann",
    fmax: float | None = None,
) -> list[dict[str, Any]]:
    """Envelope spectrum: band-pass filter, Hilbert envelope, then the spectrum of the envelope without its mean.

    Uses wandas ``band_pass_filter`` (zero-phase Butterworth) and ``envelope``. The spectrum is one
    window over the whole envelope and is scaled to the RMS amplitude of a sinusoid.

    Args:
        frame: A wandas ChannelFrame.
        band_hz: ``(low, high)`` of the band-pass filter [Hz], usually a resonance band.
        filter_order: Butterworth order.
        window: Window over the whole envelope (``scipy.signal.get_window`` name).
        fmax: Highest frequency to return [Hz].

    Returns:
        One dict per channel with ``label``, ``unit``, ``freq_hz``, ``amplitude`` (RMS), ``df_hz``
        and ``band_hz``.
    """
    lo, hi = band_hz
    env = channel_rows(frame.band_pass_filter(lo, hi, order=filter_order).envelope()).astype(float)
    env = env - env.mean(axis=-1, keepdims=True)
    n = env.shape[-1]
    w = sps.get_window(window, n)
    fs = float(frame.sampling_rate)
    f = np.fft.rfftfreq(n, 1 / fs)
    amp = np.sqrt(2) * np.abs(np.fft.rfft(env * w, axis=-1)) / np.sum(w)
    m = np.ones_like(f, dtype=bool) if fmax is None else f <= fmax
    return [
        {
            "label": label,
            "unit": unit,
            "freq_hz": f[m],
            "amplitude": a[m],
            "df_hz": fs / n,
            "band_hz": [float(lo), float(hi)],
        }
        for a, (label, unit, _) in zip(amp, channel_info(frame))
    ]
