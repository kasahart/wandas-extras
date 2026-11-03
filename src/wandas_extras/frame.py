"""ExtrasFrame: a ChannelFrame with the provisional analyses as methods.

The methods follow the wandas extension guide: the numerical kernel is an ``AudioOperation`` in
``wandas_extras.operations``, the method builds the result Frame lazily with lineage, and
``@recipe_operation`` makes the step replayable from a Recipe built with :func:`recipe_registry`.
When a method moves into wandas, it keeps its name and arguments.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dask.array as da
import numpy as np
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import RecipeRegistry, default_recipe_registry, recipe_definition, recipe_operation
from wandas.processing import Welch

from wandas_extras.operations import AngleResample, TachoRpm, angle_sample_times, validate_rpm_params


def _validate_rpm(params: Mapping[str, Any]) -> None:
    validate_rpm_params(params.get("pulses_per_rev", 1), params.get("threshold"))


def _validate_angle_resample(params: Mapping[str, Any]) -> None:
    validate_rpm_params(params.get("pulses_per_rev", 1), params.get("threshold"))
    spr = params.get("samples_per_rev", 64)
    if isinstance(spr, bool) or not isinstance(spr, int) or spr < 2:
        raise ValueError(f"samples_per_rev must be an integer of at least 2, got {spr!r}")


class ExtrasFrame(ChannelFrame):
    """ChannelFrame with the analyses of wandas-extras.

    Create it like a ChannelFrame (``ExtrasFrame.read_wav(path)``, ``ExtrasFrame.from_numpy(...)``).
    Processing methods return ExtrasFrame again, so the methods stay available along a chain.
    """

    @recipe_operation("wandas_extras.channel.rpm", validate_params=_validate_rpm)
    def rpm(self, pulses_per_rev: int = 1, threshold: float | None = None) -> ExtrasFrame:
        """Rotational speed from tachometer pulses, as a waveform in rpm.

        The speed between two rising edges is ``60 / (pulses_per_rev * interval)``. It is placed at
        the middle of the interval and linearly interpolated onto the sample times (held constant
        before the first and after the last interval). Edge times are refined by linear
        interpolation between samples.

        Args:
            pulses_per_rev: Pulses per revolution.
            threshold: Level of a rising edge in the channel unit. Defaults to the midpoint of each
                channel's minimum and maximum.

        Returns:
            An ExtrasFrame with one rpm channel per tachometer channel, at the same sampling rate.
            ``frame.rpm().data.min()`` and ``.max()`` give the speed range.
        """
        operation = TachoRpm(self.sampling_rate, pulses_per_rev=pulses_per_rev, threshold=threshold)
        channel_metadata = self._metadata_after_analysis()
        for descriptor, channel in zip(channel_metadata, self.channels, strict=True):
            descriptor["label"] = f"rpm({channel.label})"
            descriptor["calibration"] = ChannelCalibration(factor=1.0, unit="rpm")
        return self._apply_operation_instance(
            operation,
            frame_metadata_updates={"channel_metadata": channel_metadata},
        )

    def full_scale_fraction(self, full_scale: float, level: float = 0.999) -> np.ndarray:
        """Share of samples at or above ``level * full_scale`` in magnitude, for each channel.

        A scalar reduction like ``rms``: it computes now, returns a NumPy array of shape
        ``(n_channels,)`` and adds no lineage. Use it with ``peak / full_scale`` to judge clipping.

        Args:
            full_scale: Full-scale value of the recording in the channel unit (for example the
                calibrated value of digital full scale).
            level: Fraction of full scale that counts as "at full scale".
        """
        if full_scale <= 0:
            raise ValueError(f"full_scale must be positive, got {full_scale!r}")
        near = da.absolute(self._float_data) >= level * full_scale
        return np.asarray(near.mean(axis=-1).compute(), dtype=float)

    def amplitude_histogram(self, full_scale: float, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Histogram of the sample values over ``[-full_scale, full_scale]``, for each channel.

        A scalar reduction like ``rms``: it computes now and adds no lineage. Samples outside the
        range are not counted.

        Returns:
            ``(counts, edges)``: counts of shape ``(n_channels, bins)`` and the ``bins + 1`` edges.
        """
        if full_scale <= 0:
            raise ValueError(f"full_scale must be positive, got {full_scale!r}")
        rows = self._float_data
        counts = [da.histogram(rows[i], bins=bins, range=(-full_scale, full_scale))[0] for i in range(self.n_channels)]
        edges = np.linspace(-full_scale, full_scale, bins + 1)
        return np.asarray(da.stack(counts).compute()), edges

    @recipe_operation("wandas_extras.channel.angle_resample", validate_params=_validate_angle_resample)
    def angle_resample(
        self,
        tacho: int | str = -1,
        pulses_per_rev: int = 1,
        samples_per_rev: int = 64,
        threshold: float | None = None,
        rpm_range: tuple[float, float] | None = None,
    ) -> AngleFrame:
        """Resample the channels at equal shaft angles, using a tachometer channel of this frame.

        Like ``trigger()``, only the tachometer channel is computed to find the pulses; the other
        channels stay lazy. The angles run from the first to the last rising edge in steps of
        ``1 / samples_per_rev`` revolution, and each channel is linearly interpolated at the times
        of those angles. A Recipe records these arguments, so replaying it finds the pulses of its
        own input again.

        Args:
            tacho: Index or label of the tachometer channel. It is left out of the result.
            pulses_per_rev: Pulses per revolution.
            samples_per_rev: Samples per revolution; the highest order is ``samples_per_rev / 2``.
            threshold: Level of a rising edge (see :meth:`rpm`).
            rpm_range: Keep only the angles where the speed is within ``(rpm_lo, rpm_hi)``. The kept
                parts are joined end to end.

        Returns:
            An AngleFrame whose sample axis is revolutions (``sampling_rate`` is samples per
            revolution) with the other channels.
        """
        (index,) = self._channel_indices(tacho)
        index %= self.n_channels
        if self.n_channels < 2:
            raise ValueError("angle_resample needs a tachometer channel and at least one other channel")
        times = angle_sample_times(
            np.asarray(self._float_data[index].compute()),
            float(self.sampling_rate),
            pulses_per_rev,
            samples_per_rev,
            threshold,
            rpm_range,
        )
        keep = [i for i in range(self.n_channels) if i != index]
        channel_metadata = self._metadata_after_analysis()
        operation = AngleResample(self.sampling_rate, sample_times=times, drop_channel=index)
        return self._apply_operation_instance(
            operation,
            output_frame_class=AngleFrame,
            frame_metadata_updates={
                "sampling_rate": float(samples_per_rev),
                "channel_metadata": [channel_metadata[i] for i in keep],
                "channel_ids": [self._channel_ids[i] for i in keep],
                "source_time_offset": 0.0,
            },
        )


class AngleFrame(ExtrasFrame):
    """Channels sampled at equal shaft angles, from :meth:`ExtrasFrame.angle_resample`.

    ``sampling_rate`` is samples per revolution, so ``time`` counts revolutions from the start of
    the resampled signal, and every ChannelFrame method works per revolution instead of per
    second. A spectrum of this frame is over orders.
    """

    @property
    def revolutions(self) -> np.ndarray:
        """Revolutions of each sample from the start of the resampled signal (same as ``time``)."""
        return self.time

    def plot(self, *args: Any, xlabel: str | None = None, **kwargs: Any) -> Any:
        """Plot like ChannelFrame, with revolutions on the x-axis."""
        return super().plot(*args, xlabel=xlabel or "Revolutions [rev]", **kwargs)

    @recipe_operation("wandas_extras.angle.order_spectrum")
    def order_spectrum(self, revs_per_block: int = 16, window: str = "hann") -> OrderSpectralFrame:
        """Welch-averaged order spectrum (peak amplitude, like ``welch()``).

        Blocks of ``revs_per_block`` revolutions overlap by half. The order resolution is
        ``1 / revs_per_block``.
        """
        n_fft = int(self.sampling_rate) * revs_per_block
        operation = Welch(self.sampling_rate, n_fft=n_fft, hop_length=n_fft // 2, window=window)
        channel_metadata = self._metadata_after_analysis()
        return self._apply_operation_instance(
            operation,
            output_frame_class=OrderSpectralFrame,
            output_frame_kwargs={"n_fft": n_fft, "window": window, "win_length": n_fft},
            frame_metadata_updates={"channel_metadata": channel_metadata},
        )


class OrderSpectralFrame(SpectralFrame):
    """A spectrum over orders (cycles per revolution), from :meth:`AngleFrame.order_spectrum`.

    ``freqs`` holds the orders; ``orders`` is the same array under its proper name.
    """

    @property
    def orders(self) -> np.ndarray:
        """Orders of the bins [cycles per revolution]."""
        return np.asarray(self.freqs)

    def plot(self, *args: Any, xlabel: str | None = None, **kwargs: Any) -> Any:
        """Plot like SpectralFrame, with orders on the x-axis."""
        return super().plot(*args, xlabel=xlabel or "Order", **kwargs)


def recipe_registry() -> RecipeRegistry:
    """The default wandas Recipe registry plus the ExtrasFrame methods.

    Pass it to ``RecipePlan.from_frame``, ``RecipePlan.load`` and ``RecipePlan.apply``.
    """
    registry = default_recipe_registry()
    for method in (ExtrasFrame.rpm, ExtrasFrame.angle_resample, AngleFrame.order_spectrum):
        registry = registry.with_operation(recipe_definition(method))
    return registry
