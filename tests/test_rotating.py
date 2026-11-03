import numpy as np
import pytest
import wandas as wd

from wandas_extras import order_tracks

FS = 8192


def _run_up(rpm0=600.0, rpm1=1800.0, seconds=8.0, orders=((1.0, 1.0), (3.0, 0.5))):
    """Linear run-up with a once-per-rev tacho and a vibration made of fixed orders (peak amplitudes)."""
    t = np.arange(int(FS * seconds)) / FS
    rev = (rpm0 * t + 0.5 * (rpm1 - rpm0) / seconds * t**2) / 60
    vib = sum(a * np.sin(2 * np.pi * o * rev) for o, a in orders)
    tacho = (np.mod(rev, 1.0) < 0.1).astype(float) * 5.0
    return wd.from_numpy(vib, FS, ch_units="m/s^2"), wd.from_numpy(tacho, FS, ch_units="V"), rev


def test_order_tracks_are_flat_for_constant_order_amplitudes():
    vib, tacho, _ = _run_up()

    (ch,) = order_tracks(vib, tacho, orders=[1.0, 3.0], window="flattop")

    assert np.all(np.diff(ch["rpm"]) > 0)
    np.testing.assert_allclose(ch["tracks"]["1.0"], 1 / np.sqrt(2), rtol=0.01)
    np.testing.assert_allclose(ch["tracks"]["3.0"], 0.5 / np.sqrt(2), rtol=0.01)


def test_tacho_must_be_one_channel_with_pulses():
    vib, tacho, _ = _run_up()
    with pytest.raises(ValueError, match="one channel"):
        order_tracks(vib, wd.from_numpy(np.vstack([np.asarray(tacho.data)] * 2), FS), orders=[1.0])
    with pytest.raises(ValueError, match="rising edges"):
        order_tracks(vib, wd.from_numpy(np.zeros(len(vib.data)), FS), orders=[1.0])
