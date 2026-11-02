import numpy as np
import pytest
import wandas as wd

from wandas_extras import integrate


def _accel(freqs, amps, fs=4096, seconds=2.0, unit="m/s^2"):
    t = np.arange(int(fs * seconds)) / fs
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
    return wd.from_numpy(x, fs, ch_units=unit), t


def test_acceleration_to_velocity_of_a_sine():
    f0, a0 = 50.0, 2.0
    frame, t = _accel([f0], [a0])

    vel = integrate(frame)

    expected = -a0 / (2 * np.pi * f0) * np.cos(2 * np.pi * f0 * t)
    np.testing.assert_allclose(np.asarray(vel.data), expected, atol=1e-9)
    assert vel.channels[0].unit == "m/s"
    assert float(vel.rms[0]) == pytest.approx(a0 / (2 * np.pi * f0) / np.sqrt(2), rel=1e-9)


def test_band_limits_remove_components_outside():
    frame, _ = _accel([5.0, 100.0], [1.0, 1.0])

    vel = integrate(frame, fmin=10.0, fmax=1000.0)

    assert float(vel.rms[0]) == pytest.approx(1 / (2 * np.pi * 100) / np.sqrt(2), rel=1e-9)


def test_double_integration_and_differentiation_round_trip():
    frame, _ = _accel([40.0, 120.0], [1.0, 0.3])

    disp = integrate(frame, order=2)
    back = integrate(disp, order=-2)

    assert disp.channels[0].unit == "m"
    assert back.channels[0].unit == "m/s^2"
    np.testing.assert_allclose(np.asarray(back.data), np.asarray(frame.data), atol=1e-9)


def test_calibration_is_applied_and_unit_can_be_given():
    frame, _ = _accel([50.0], [1.0], unit="g")
    frame = frame.with_calibration([9.80665])  # samples in g, factor to m/s^2

    vel = integrate(frame, unit="m/s")

    assert vel.channels[0].unit == "m/s"
    assert float(vel.rms[0]) == pytest.approx(9.80665 / (2 * np.pi * 50) / np.sqrt(2), rel=1e-9)


def test_input_frame_is_unchanged():
    frame, _ = _accel([50.0], [1.0])
    before = np.asarray(frame.data).copy()

    integrate(frame)

    np.testing.assert_array_equal(np.asarray(frame.data), before)
    assert frame.channels[0].unit == "m/s^2"


def test_rejects_unknown_unit_and_zero_order():
    frame, _ = _accel([50.0], [1.0], unit="V")
    with pytest.raises(ValueError, match="unit"):
        integrate(frame)
    with pytest.raises(ValueError, match="order"):
        integrate(frame, order=0, unit="V")
