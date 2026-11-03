from typing import cast

import numpy as np
import pytest
import wandas as wd
from wandas.pipeline import RecipePlan

from wandas_extras import ExtrasFrame, recipe_registry

FS = 8192


def _run_up_tacho(rpm0=600.0, rpm1=1800.0, seconds=8.0, pulses_per_rev=1):
    """Square tachometer pulses for a linear run-up, and the true speed at each sample."""
    t = np.arange(int(FS * seconds)) / FS
    rev = (rpm0 * t + 0.5 * (rpm1 - rpm0) / seconds * t**2) / 60
    tacho = (np.mod(rev * pulses_per_rev, 1.0) < 0.1).astype(float) * 5.0
    return tacho, rpm0 + (rpm1 - rpm0) / seconds * t


def _frame(data, **kwargs) -> ExtrasFrame:
    # wandas 0.13 types from_numpy as returning ChannelFrame; the object is an ExtrasFrame.
    return cast(ExtrasFrame, ExtrasFrame.from_numpy(data, FS, **kwargs))


def test_rpm_follows_the_run_up_as_a_lazy_frame():
    tacho, true_rpm = _run_up_tacho()
    frame = _frame(tacho, ch_labels=["tacho"], ch_units="V")

    rpm = frame.rpm()

    assert type(rpm) is ExtrasFrame
    assert rpm.sampling_rate == FS
    assert rpm.channels[0].unit == "rpm"
    assert rpm.channels[0].label == "rpm(tacho)"
    inner = slice(FS, -FS)  # held constant outside the first and last pulse intervals
    np.testing.assert_allclose(rpm.data[inner], true_rpm[inner], rtol=5e-3)  # edges quantised to 1 sample
    np.testing.assert_allclose(frame.data, tacho)  # input unchanged


def test_rpm_counts_pulses_per_revolution_and_threshold():
    tacho, true_rpm = _run_up_tacho(pulses_per_rev=4)
    frame = _frame(tacho, ch_units="V")

    rpm = frame.rpm(pulses_per_rev=4, threshold=2.0)

    np.testing.assert_allclose(rpm.data[FS:-FS], true_rpm[FS:-FS], rtol=2e-2)


def test_rpm_channels_are_independent():
    tacho, _ = _run_up_tacho()
    slow, _ = _run_up_tacho(rpm0=300.0, rpm1=300.0)
    frame = _frame(np.vstack([tacho, slow]))

    rpm = frame.rpm().data

    np.testing.assert_allclose(rpm[0], _frame(tacho).rpm().data)
    np.testing.assert_allclose(rpm[1, FS:-FS], 300.0, rtol=5e-3)


def test_rpm_rejects_bad_arguments_and_signals_without_pulses():
    frame = _frame(np.zeros(FS))

    with pytest.raises(ValueError, match="pulses_per_rev"):
        frame.rpm(pulses_per_rev=0)
    with pytest.raises(ValueError, match="rising edges"):
        frame.rpm().data


def test_rpm_replays_from_a_recipe():
    tacho, _ = _run_up_tacho()
    other, _ = _run_up_tacho(rpm0=900.0, rpm1=1200.0)
    registry = recipe_registry()
    processed = _frame(tacho).low_pass_filter(cutoff=2000.0).rpm(threshold=2.5)

    plan = RecipePlan.from_dict(
        RecipePlan.from_frame(processed, input_names=("tacho",), registry=registry).to_dict(), registry=registry
    )
    replayed = plan.apply({"tacho": _frame(other)}, registry=registry)

    expected = _frame(other).low_pass_filter(cutoff=2000.0).rpm(threshold=2.5)
    np.testing.assert_allclose(replayed.data, expected.data)


def test_plain_channel_frame_does_not_change():
    assert not hasattr(wd.from_numpy(np.zeros(8), FS), "rpm")
