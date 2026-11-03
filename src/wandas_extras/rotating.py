"""Rotating-machine analysis: speed from a tachometer and order analysis by angular resampling."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

from wandas_extras._frame import channel_info, channel_rows
from wandas_extras.operations import rising_edge_times


def _tacho_row(tacho: Any) -> np.ndarray:
    rows = channel_rows(tacho)
    if rows.shape[0] != 1:
        raise ValueError(f"tacho must have one channel, got {rows.shape[0]}; use frame.get_channel(...)")
    return np.asarray(rows[0], dtype=float)


def _pulse_revs(tacho: Any, pulses_per_rev: int, threshold: float | None) -> tuple[np.ndarray, np.ndarray]:
    t = rising_edge_times(_tacho_row(tacho), float(tacho.sampling_rate), threshold)
    return t, np.arange(len(t)) / pulses_per_rev


def _angle_resample(frame: Any, tacho: Any, pulses_per_rev: int, samples_per_rev: int, threshold: float | None):
    if float(tacho.sampling_rate) != float(frame.sampling_rate):
        raise ValueError("frame and tacho must have the same sampling rate")
    t, rev = _pulse_revs(tacho, pulses_per_rev, threshold)
    theta = np.arange(rev[0], rev[-1], 1 / samples_per_rev)
    tg = np.interp(theta, rev, t)
    rows = channel_rows(frame)
    ts = np.arange(rows.shape[-1]) / float(frame.sampling_rate)
    xa = np.vstack([np.interp(tg, ts, np.asarray(r, dtype=float)) for r in rows])
    return theta, tg, xa


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
