from typing import cast

import numpy as np
import pytest
import wandas as wd
from wandas.pipeline import RecipePlan

from wandas_extras import AngleFrame, ExtrasFrame, OrderSpectralFrame, recipe_registry

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


def test_full_scale_fraction_and_histogram_per_channel():
    x = np.array([[0.0, 0.5, -1.0, 1.0, 0.2], [0.1, 0.1, 0.1, 0.1, 0.1]])
    frame = _frame(x)

    np.testing.assert_allclose(frame.full_scale_fraction(1.0), [0.4, 0.0])
    counts, edges = frame.amplitude_histogram(1.0, bins=4)

    np.testing.assert_array_equal(counts, [np.histogram(row, bins=4, range=(-1.0, 1.0))[0] for row in x])
    np.testing.assert_allclose(edges, [-1.0, -0.5, 0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="full_scale"):
        frame.full_scale_fraction(0.0)


def test_envelope_spectrum_is_a_wandas_chain():
    """The README recipe for the envelope spectrum: band-pass, envelope, remove DC, FFT."""
    t = np.arange(FS * 2) / FS
    carrier = np.sin(2 * np.pi * 3000 * t)
    x = (1 + 0.5 * np.sin(2 * np.pi * 100 * t)) * carrier
    frame = _frame(x)

    spec = frame.band_pass_filter(2000.0, 4000.0, order=6).envelope().remove_dc().fft(window="hann")

    amp = np.abs(np.asarray(spec.data))
    i = int(np.argmin(np.abs(spec.freqs - 100.0)))
    assert spec.freqs[np.argmax(amp[1:]) + 1] == pytest.approx(100.0)
    assert amp[i] == pytest.approx(0.5, rel=0.02)  # peak amplitude of the modulation


def _run_up_frame(seconds=8.0, orders=((1.0, 1.0), (3.0, 0.5))):
    """Vibration made of fixed orders (peak amplitudes) during a run-up, with the tacho as channel 'tacho'."""
    tacho, _ = _run_up_tacho(seconds=seconds)
    t = np.arange(len(tacho)) / FS
    rev = (600.0 * t + 0.5 * 1200.0 / seconds * t**2) / 60
    vib = sum(a * np.sin(2 * np.pi * o * rev) for o, a in orders)
    return _frame(np.vstack([vib, tacho]), ch_labels=["vib", "tacho"], ch_units=["m/s^2", "V"])


def test_angle_resample_drops_the_tacho_and_counts_revolutions():
    frame = _run_up_frame()

    angle = frame.angle_resample("tacho", samples_per_rev=64)

    assert type(angle) is AngleFrame
    assert angle.sampling_rate == 64
    assert [ch.label for ch in angle.channels] == ["vib"]
    assert angle.channels[0].unit == "m/s^2"
    assert angle.revolutions[1] == pytest.approx(1 / 64)
    # 600 -> 1800 rpm in 8 s is 160 revolutions; the angles span first to last pulse
    assert angle.n_samples / 64 == pytest.approx(159, abs=1)


def test_order_spectrum_peaks_at_the_orders_with_peak_amplitude():
    angle = _run_up_frame().angle_resample("tacho")

    spec = angle.order_spectrum(revs_per_block=16, window="flattop")

    assert type(spec) is OrderSpectralFrame
    amp = np.abs(np.asarray(spec.data))
    for order, a in ((1.0, 1.0), (3.0, 0.5)):
        assert amp[int(np.argmin(np.abs(spec.orders - order)))] == pytest.approx(a, rel=0.01)
    assert spec.orders[1] == pytest.approx(1 / 16)


def test_rpm_range_keeps_only_that_speed():
    frame = _run_up_frame()

    full = frame.angle_resample("tacho")
    part = frame.angle_resample("tacho", rpm_range=(1000.0, 1400.0))

    # revolutions between 1000 and 1400 rpm in a linear run-up: (1400^2 - 1000^2) / (2 * 150 * 60)
    assert part.n_samples / 64 == pytest.approx((1400**2 - 1000**2) / (2 * 150 * 60), rel=0.02)
    assert part.n_samples < full.n_samples


def test_angle_resample_needs_another_channel():
    tacho, _ = _run_up_tacho()
    with pytest.raises(ValueError, match="at least one other channel"):
        _frame(tacho).angle_resample(0)


def test_order_spectrum_replays_from_a_recipe():
    registry = recipe_registry()
    frame = _run_up_frame()
    processed = frame.angle_resample("tacho", samples_per_rev=32).order_spectrum(revs_per_block=8)
    plan = RecipePlan.from_dict(
        RecipePlan.from_frame(processed, input_names=("signal",), registry=registry).to_dict(), registry=registry
    )

    other = _run_up_frame(seconds=6.0, orders=((2.0, 1.0),))
    replayed = plan.apply({"signal": other}, registry=registry)

    expected = other.angle_resample("tacho", samples_per_rev=32).order_spectrum(revs_per_block=8)
    assert type(replayed) is OrderSpectralFrame
    np.testing.assert_allclose(np.asarray(replayed.data), np.asarray(expected.data))
