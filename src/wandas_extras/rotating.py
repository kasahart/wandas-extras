"""Rotating-machine analysis: speed from a tachometer and order analysis by angular resampling."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

from wandas_extras._frame import channel_info, channel_rows


def _tacho_row(tacho: Any) -> np.ndarray:
    rows = channel_rows(tacho)
    if rows.shape[0] != 1:
        raise ValueError(f"tacho must have one channel, got {rows.shape[0]}; use frame.get_channel(...)")
    return np.asarray(rows[0], dtype=float)


def pulse_times(tacho: Any, pulses_per_rev: int = 1, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Times of the rising edges of a tachometer signal and the revolution count at each edge.

    Args:
        tacho: A one-channel ChannelFrame with the tachometer pulses.
        pulses_per_rev: Pulses per revolution.
        threshold: Level of a rising edge in the channel unit. Defaults to the midpoint of the
            minimum and maximum.

    Returns:
        ``(time_s, revolutions)``. Edge times are refined by linear interpolation between samples.
    """
    x = _tacho_row(tacho)
    fs = float(tacho.sampling_rate)
    thr = 0.5 * (np.max(x) + np.min(x)) if threshold is None else threshold
    edges = np.flatnonzero((x[1:] > thr) & (x[:-1] <= thr)) + 1
    if len(edges) < 3:
        raise ValueError("fewer than 3 rising edges in the tachometer signal; check threshold")
    frac = (thr - x[edges - 1]) / (x[edges] - x[edges - 1])
    t = (edges - 1 + frac) / fs
    return t, np.arange(len(t)) / pulses_per_rev


def rpm_profile(tacho: Any, pulses_per_rev: int = 1, threshold: float | None = None) -> dict[str, Any]:
    """Rotational speed between successive tachometer pulses.

    Returns:
        ``time_s`` (midpoint of each pulse interval), ``rpm``, ``rpm_min`` and ``rpm_max``.
    """
    t, rev = pulse_times(tacho, pulses_per_rev, threshold)
    rpm = 60 * np.diff(rev) / np.diff(t)
    return {"time_s": 0.5 * (t[1:] + t[:-1]), "rpm": rpm, "rpm_min": float(rpm.min()), "rpm_max": float(rpm.max())}


def _angle_resample(frame: Any, tacho: Any, pulses_per_rev: int, samples_per_rev: int, threshold: float | None):
    if float(tacho.sampling_rate) != float(frame.sampling_rate):
        raise ValueError("frame and tacho must have the same sampling rate")
    t, rev = pulse_times(tacho, pulses_per_rev, threshold)
    theta = np.arange(rev[0], rev[-1], 1 / samples_per_rev)
    tg = np.interp(theta, rev, t)
    rows = channel_rows(frame)
    ts = np.arange(rows.shape[-1]) / float(frame.sampling_rate)
    xa = np.vstack([np.interp(tg, ts, np.asarray(r, dtype=float)) for r in rows])
    return theta, tg, xa


def order_spectrum(
    frame: Any,
    tacho: Any,
    pulses_per_rev: int = 1,
    samples_per_rev: int = 64,
    revs_per_block: int = 16,
    window: str = "hann",
    threshold: float | None = None,
    rpm_range: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Average order spectrum (RMS amplitude) by angular resampling.

    Each channel is resampled to ``samples_per_rev`` samples per revolution using the tachometer
    pulses, then averaged with Welch's method over blocks of ``revs_per_block`` revolutions
    (50 % overlap).

    Args:
        frame: A ChannelFrame with the vibration or sound channels.
        tacho: A one-channel ChannelFrame with the tachometer pulses, same sampling rate.
        pulses_per_rev: Pulses per revolution.
        samples_per_rev: Samples per revolution; the highest order is ``samples_per_rev / 2``.
        revs_per_block: Revolutions per block; the order resolution is ``1 / revs_per_block``.
        window: Window name accepted by ``scipy.signal.get_window``.
        threshold: Tachometer threshold (see :func:`pulse_times`).
        rpm_range: Only average the part where the speed is within ``(rpm_lo, rpm_hi)``.

    Returns:
        One dict per channel with ``label``, ``unit``, ``db_ref``, ``order``, ``amplitude_rms``,
        ``level_db``, ``d_order`` and ``n_averages``.
    """
    theta, tg, xa = _angle_resample(frame, tacho, pulses_per_rev, samples_per_rev, threshold)
    if rpm_range is not None:
        rpm_t = 60 / np.gradient(tg, theta)
        xa = xa[:, (rpm_t >= rpm_range[0]) & (rpm_t <= rpm_range[1])]
    nb = samples_per_rev * revs_per_block
    order, p = sps.welch(xa, samples_per_rev, window=window, nperseg=nb, scaling="spectrum", axis=-1)
    n_avg = (xa.shape[-1] - nb) // (nb // 2) + 1
    out = []
    for amp2, (label, unit, ref) in zip(p, channel_info(frame)):
        amp = np.sqrt(amp2)
        with np.errstate(divide="ignore"):
            level = 20 * np.log10(amp / ref)
        out.append(
            {
                "label": label,
                "unit": unit,
                "db_ref": ref,
                "order": order,
                "amplitude_rms": amp,
                "level_db": level,
                "d_order": 1 / revs_per_block,
                "n_averages": int(n_avg),
            }
        )
    return out


def order_tracks(
    frame: Any,
    tacho: Any,
    orders: list[float],
    pulses_per_rev: int = 1,
    samples_per_rev: int = 64,
    revs_per_block: int = 8,
    window: str = "hann",
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """RMS amplitude of the given orders against rotational speed (order tracking).

    Blocks of ``revs_per_block`` revolutions (50 % overlap) of the angle-resampled signal are
    transformed. For each order the largest bin among the nearest bin and its neighbours is taken,
    so an order between bins is not under-read.

    Returns:
        One dict per channel with ``label``, ``unit``, ``rpm`` (mean speed of each block),
        ``tracks`` (``{order: amplitude_rms array}``, keys are ``str(order)``) and ``d_order``.
    """
    _, tg, xa = _angle_resample(frame, tacho, pulses_per_rev, samples_per_rev, threshold)
    nb = samples_per_rev * revs_per_block
    w = sps.get_window(window, nb)
    gain = np.sum(w) / nb
    starts = range(0, xa.shape[-1] - nb + 1, nb // 2)
    oo = np.fft.rfftfreq(nb, 1 / samples_per_rev)
    rpm = np.array([60 * revs_per_block / (tg[i0 + nb - 1] - tg[i0]) for i0 in starts])
    bins = [int(np.argmin(np.abs(oo - o))) for o in orders]
    out = []
    for x, (label, unit, _) in zip(xa, channel_info(frame)):
        spec = np.array([np.fft.rfft(x[i0 : i0 + nb] * w) / (nb * gain) for i0 in starts])
        tracks = {
            str(o): np.sqrt(2) * np.max(np.abs(spec[:, max(k - 1, 0) : k + 2]), axis=1) for o, k in zip(orders, bins)
        }
        out.append({"label": label, "unit": unit, "rpm": rpm, "tracks": tracks, "d_order": 1 / revs_per_block})
    return out
