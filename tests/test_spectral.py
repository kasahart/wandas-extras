import numpy as np
import pytest
import wandas as wd

from wandas_extras import spectral_peaks


def _tones(freqs, amps, fs=8000, seconds=2.0, unit=None):
    t = np.arange(int(fs * seconds)) / fs
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
    return wd.from_numpy(x, fs, ch_units=unit)


def test_finds_tones_sorted_by_frequency():
    sp = _tones([1000.0, 500.0], [1.0, 0.5]).welch(n_fft=1024)

    (ch,) = spectral_peaks(sp)

    assert [p["freq_hz"] for p in ch["peaks"]] == [500.0, 1000.0]
    assert ch["peaks"][1]["amplitude"] == pytest.approx(1.0, rel=1e-3)
    assert ch["df_hz"] == pytest.approx(8000 / 1024)


def test_parabolic_interpolation_moves_off_bin_tone_toward_true_frequency():
    f0 = 1003.0  # between bins (df = 7.8125 Hz)
    sp = _tones([f0], [1.0]).welch(n_fft=1024)

    (raw,) = spectral_peaks(sp)
    (fit,) = spectral_peaks(sp, interpolation="parabolic")

    err_raw = abs(raw["peaks"][0]["freq_hz"] - f0)
    err_fit = abs(fit["peaks"][0]["freq_hz"] - f0)
    assert err_fit < err_raw
    assert fit["peaks"][0]["level_db"] >= raw["peaks"][0]["level_db"]


def test_level_uses_channel_reference_and_limits_apply():
    sp = _tones([250.0, 1000.0, 2000.0], [1.0, 1.0, 0.1], unit="Pa").welch(n_fft=1024)

    (ch,) = spectral_peaks(sp, fmin=500, max_peaks=1)

    assert ch["db_ref"] == pytest.approx(2e-5)
    assert [p["freq_hz"] for p in ch["peaks"]] == [1000.0]
    assert ch["peaks"][0]["level_db"] == pytest.approx(20 * np.log10(1 / 2e-5), abs=0.01)


def test_one_result_per_channel():
    fs = 8000
    t = np.arange(fs) / fs
    frame = wd.from_numpy(np.vstack([np.sin(2 * np.pi * 500 * t), np.sin(2 * np.pi * 1500 * t)]), fs)

    a, b = spectral_peaks(frame.welch(n_fft=1024))

    assert [p["freq_hz"] for p in a["peaks"]] == [500.0]
    assert [p["freq_hz"] for p in b["peaks"]] == [1500.0]


def test_rejects_unknown_interpolation():
    with pytest.raises(ValueError, match="interpolation"):
        spectral_peaks(_tones([1000.0], [1.0]).welch(n_fft=256), interpolation="cubic")  # ty: ignore[invalid-argument-type]
