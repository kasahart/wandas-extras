import numpy as np
import pytest
import wandas as wd

from wandas_extras import signal_stats


def test_sine_statistics_use_channel_reference():
    fs = 8000
    t = np.arange(fs) / fs
    amp = 2e-5 * 10 ** (94 / 20) * np.sqrt(2)  # 94 dB SPL
    frame = wd.from_numpy(amp * np.sin(2 * np.pi * 1000 * t), fs, ch_units="Pa")

    (s,) = signal_stats(frame)

    assert s["unit"] == "Pa"
    assert s["db_ref"] == pytest.approx(2e-5)
    assert s["rms_db"] == pytest.approx(94.0, abs=1e-6)
    assert s["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-6)
    assert s["kurtosis"] == pytest.approx(1.5, rel=1e-3)  # Pearson kurtosis of a sine
    assert "histogram" not in s


def test_full_scale_reports_clipping():
    x = np.array([0.0, 0.5, -1.0, 1.0, 0.2])
    (s,) = signal_stats(wd.from_numpy(x, 10), full_scale=1.0, histogram_bins=4)

    assert s["peak_over_full_scale"] == pytest.approx(1.0)
    assert s["fraction_near_full_scale"] == pytest.approx(0.4)
    assert s["histogram"]["counts"].sum() == len(x)
    assert s["histogram"]["edges"][[0, -1]].tolist() == [-1.0, 1.0]


def test_one_result_per_channel_and_silent_channel():
    frame = wd.from_numpy(np.vstack([np.ones(10), np.zeros(10)]), 10, ch_labels=["a", "b"])

    a, b = signal_stats(frame)

    assert (a["label"], b["label"]) == ("a", "b")
    assert a["crest_factor"] == pytest.approx(1.0)
    assert b["crest_factor"] is None


def test_frame_is_unchanged_and_trim_selects_segment():
    frame = wd.from_numpy(np.r_[np.zeros(100), np.ones(100)], 100)
    before = np.asarray(frame.data).copy()

    (s,) = signal_stats(frame.trim(1.0, 2.0))

    assert s["mean"] == pytest.approx(1.0)
    np.testing.assert_array_equal(np.asarray(frame.data), before)


def test_rejects_non_positive_full_scale():
    with pytest.raises(ValueError, match="full_scale"):
        signal_stats(wd.from_numpy(np.ones(10), 10), full_scale=0)
