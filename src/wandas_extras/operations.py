"""Numerical kernels of the ExtrasFrame methods, as wandas AudioOperations.

Each operation works on the ``(channels, samples)`` array that wandas passes to ``_process``. It
does not build Frames or touch Frame metadata; that is the job of the ExtrasFrame method.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from wandas.processing.base import AudioOperation, ChannelIndependentAudioOperation, precision_real_dtype
from wandas.utils.types import NDArrayReal


def validate_rpm_params(pulses_per_rev: Any, threshold: Any) -> None:
    """Reject arguments that the tachometer kernel cannot use."""
    if isinstance(pulses_per_rev, bool) or not isinstance(pulses_per_rev, (int, np.integer)) or pulses_per_rev < 1:
        raise ValueError(f"pulses_per_rev must be a positive integer, got {pulses_per_rev!r}")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))):
        raise ValueError(f"threshold must be a number or None, got {threshold!r}")


def rising_edge_times(x: np.ndarray, fs: float, threshold: float | None = None) -> np.ndarray:
    """Times [s] of the rising edges of a pulse train, refined by linear interpolation."""
    thr = 0.5 * (np.max(x) + np.min(x)) if threshold is None else threshold
    edges = np.flatnonzero((x[1:] > thr) & (x[:-1] <= thr)) + 1
    if len(edges) < 3:
        raise ValueError("fewer than 3 rising edges in the tachometer signal; check threshold")
    frac = (thr - x[edges - 1]) / (x[edges] - x[edges - 1])
    return (edges - 1 + frac) / fs


class TachoRpm(ChannelIndependentAudioOperation[NDArrayReal, NDArrayReal]):
    """Rotational speed [rpm] from tachometer pulses, interpolated onto the sample times."""

    name = "wandas_extras_tacho_rpm"
    _display = "rpm"
    _keeps_input_precision = True

    def __init__(self, sampling_rate: float, pulses_per_rev: int = 1, threshold: float | None = None) -> None:
        super().__init__(sampling_rate, pulses_per_rev=pulses_per_rev, threshold=threshold)

    def validate_params(self) -> None:
        validate_rpm_params(self._config_value("pulses_per_rev"), self._config_value("threshold"))

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return precision_real_dtype(input_dtype)

    def _process(self, data: NDArrayReal) -> NDArrayReal:  # ty: ignore[invalid-mutable-override]
        fs = float(self.sampling_rate)
        ppr = int(self._config_value("pulses_per_rev"))
        threshold = self._config_value("threshold")
        rows = np.atleast_2d(np.asarray(data, dtype=float))
        t_samples = np.arange(rows.shape[-1]) / fs
        out = np.empty_like(rows)
        for i, x in enumerate(rows):
            t = rising_edge_times(x, fs, threshold)
            rpm = 60.0 / (ppr * np.diff(t))
            out[i] = np.interp(t_samples, 0.5 * (t[1:] + t[:-1]), rpm)
        return out.reshape(np.shape(data))


def angle_sample_times(
    tacho: NDArrayReal,
    fs: float,
    pulses_per_rev: int,
    samples_per_rev: int,
    threshold: float | None = None,
    rpm_range: tuple[float, float] | None = None,
) -> NDArrayReal:
    """Times [s] of equally spaced shaft angles, from the rising edges of a tachometer signal.

    The revolution count is ``k / pulses_per_rev`` at the k-th edge and linear between edges. The
    angles run from the first to the last edge in steps of ``1 / samples_per_rev`` revolution. With
    ``rpm_range``, only the angles where the speed is within ``(rpm_lo, rpm_hi)`` are kept.
    """
    t = rising_edge_times(np.asarray(tacho, dtype=float), fs, threshold)
    rev = np.arange(len(t)) / pulses_per_rev
    theta = np.arange(rev[0], rev[-1], 1 / samples_per_rev)
    times = np.interp(theta, rev, t)
    if rpm_range is not None:
        rpm = 60 / np.gradient(times, theta)
        times = times[(rpm >= rpm_range[0]) & (rpm <= rpm_range[1])]
    return times


class AngleResample(AudioOperation[NDArrayReal, NDArrayReal]):
    """Linear interpolation of each channel at given sample times (angle-domain resampling).

    Drops the tachometer channel, so the channel count changes: a cross-channel AudioOperation.
    """

    name = "wandas_extras_angle_resample"
    _display = "angle"
    _keeps_input_precision = True

    def __init__(self, sampling_rate: float, sample_times: NDArrayReal, drop_channel: int) -> None:
        super().__init__(sampling_rate, sample_times=np.asarray(sample_times, dtype=float), drop_channel=drop_channel)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0] - 1, len(self._config_value("sample_times")))

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return precision_real_dtype(input_dtype)

    def _process(self, data: NDArrayReal) -> NDArrayReal:  # ty: ignore[invalid-mutable-override]
        times = self._config_value("sample_times")
        rows = np.delete(np.atleast_2d(np.asarray(data, dtype=float)), int(self._config_value("drop_channel")), axis=0)
        t_samples = np.arange(rows.shape[-1]) / float(self.sampling_rate)
        return np.vstack([np.interp(times, t_samples, x) for x in rows]).reshape(len(rows), len(times))
