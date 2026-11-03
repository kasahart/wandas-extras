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
from wandas.pipeline import RecipeRegistry, default_recipe_registry, recipe_definition, recipe_operation

from wandas_extras.operations import TachoRpm, validate_rpm_params


def _validate_rpm(params: Mapping[str, Any]) -> None:
    validate_rpm_params(params.get("pulses_per_rev", 1), params.get("threshold"))


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


def recipe_registry() -> RecipeRegistry:
    """The default wandas Recipe registry plus the ExtrasFrame methods.

    Pass it to ``RecipePlan.from_frame``, ``RecipePlan.load`` and ``RecipePlan.apply``.
    """
    return default_recipe_registry().with_operation(recipe_definition(ExtrasFrame.rpm))
