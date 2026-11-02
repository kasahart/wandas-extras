import numpy as np
import pytest
import wandas as wd

from wandas_extras import envelope_spectrum, spectral_kurtosis

FS = 16000


def _bearing_like(seconds=2.0, carrier=3000.0, rate=87.0, seed=0):
    """Gaussian noise plus decaying bursts at a resonance, repeating at ``rate`` Hz."""
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    t = np.arange(n) / FS
    x = 0.1 * rng.standard_normal(n)
    burst_t = np.arange(int(0.004 * FS)) / FS
    burst = np.exp(-burst_t / 0.0008) * np.sin(2 * np.pi * carrier * burst_t)
    for t0 in np.arange(0.01, seconds - 0.01, 1 / rate):
        i = int(t0 * FS)
        x[i : i + len(burst)] += burst
    return wd.from_numpy(x, FS, ch_units="m/s^2"), t


def test_spectral_kurtosis_is_near_zero_for_gaussian_noise():
    rng = np.random.default_rng(3)
    (ch,) = spectral_kurtosis(wd.from_numpy(rng.standard_normal(FS * 4), FS))

    assert abs(float(np.median(ch["kurtosis"]))) < 0.1
    assert ch["freq_hz"][0] > 0 and ch["freq_hz"][-1] < FS / 2


def test_spectral_kurtosis_peaks_in_the_resonance_band():
    frame, _ = _bearing_like(rate=10.0)  # sparse impacts: most STFT frames hold none

    (ch,) = spectral_kurtosis(frame)

    assert 2000 <= ch["max_freq_hz"] <= 4000
    (limited,) = spectral_kurtosis(frame, fmin=5000, fmax=7000)
    assert limited["freq_hz"].min() >= 5000 and limited["freq_hz"].max() <= 7000


def test_envelope_spectrum_shows_the_repetition_rate():
    frame, _ = _bearing_like()

    (ch,) = envelope_spectrum(frame, (2000.0, 4000.0), fmax=500.0)

    peak = ch["freq_hz"][np.argmax(ch["amplitude"])]
    assert peak == pytest.approx(87.0, abs=2 * ch["df_hz"])
    assert ch["freq_hz"][-1] <= 500.0
    assert ch["unit"] == "m/s^2"


def test_envelope_spectrum_amplitude_of_a_modulated_tone():
    t = np.arange(FS * 2) / FS
    x = (1 + 0.5 * np.cos(2 * np.pi * 50 * t)) * np.sin(2 * np.pi * 3000 * t)

    (ch,) = envelope_spectrum(wd.from_numpy(x, FS), (2000.0, 4000.0), window="flattop")

    i = int(np.argmin(np.abs(ch["freq_hz"] - 50)))
    assert ch["amplitude"][i] == pytest.approx(0.5 / np.sqrt(2), rel=0.02)
