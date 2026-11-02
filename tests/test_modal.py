import numpy as np
import pytest
import wandas as wd
from scipy import signal as sps

from wandas_extras import modal_parameters

MODES = [(120.0, 0.02, 1.0), (310.0, 0.01, 0.6)]  # (natural frequency, damping ratio, modal mass inverse)


class _FRF:
    """Minimal one-channel FRF frame (freqs + complex data) for analytic tests."""

    def __init__(self, f, h):
        self.freqs, self.data = f, h


def _receptance(f):
    w = 2 * np.pi * f
    return sum(a / ((2 * np.pi * fr) ** 2 - w**2 + 2j * z * (2 * np.pi * fr) * w) for fr, z, a in MODES)


@pytest.mark.parametrize("method", ["half_power", "sdof_fit"])
def test_two_modes_from_analytic_receptance(method):
    f = np.arange(0, 600, 0.25)
    modes = modal_parameters(_FRF(f, _receptance(f)), "receptance", method=method)

    assert len(modes) == 2
    for m, (fr, z, _) in zip(modes, MODES):
        assert m["freq_hz"] == pytest.approx(fr, rel=2e-3)
        assert m["damping_ratio"] == pytest.approx(z, rel=0.15 if method == "half_power" else 0.02)
        assert m["coherence_at_peak"] is None


def test_inertance_is_converted_for_the_fit_and_bands_select_modes():
    f = np.arange(0, 600, 0.25)
    inert = _receptance(f) * (2j * np.pi * f) ** 2

    (m,) = modal_parameters(_FRF(f, inert), "inertance", method="sdof_fit", bands_hz=[(250.0, 370.0)])

    assert m["freq_hz"] == pytest.approx(310.0, rel=1e-3)
    assert m["damping_ratio"] == pytest.approx(0.01, rel=0.02)


def test_from_wandas_transfer_function_with_coherence():
    fs, n = 8192, 8192 * 30  # high fs keeps the bilinear frequency warping small
    rng = np.random.default_rng(0)
    force = rng.standard_normal(n)
    fr, z = 200.0, 0.02
    wr = 2 * np.pi * fr
    b, a = sps.bilinear([1.0], [1.0, 2 * z * wr, wr**2], fs)
    disp = sps.lfilter(b, a, force)
    frame = wd.from_numpy(np.vstack([force, disp]), fs, ch_labels=["F", "x"])
    tf = frame.transfer_function(n_fft=8192, hop_length=4096).get_channel(2)
    coh = frame.coherence(n_fft=8192, hop_length=4096).get_channel(2)

    (m,) = modal_parameters(tf, "receptance", coherence=coh, fmin=50, fmax=500, min_prominence_db=10)

    assert m["freq_hz"] == pytest.approx(fr, rel=0.01)
    assert m["damping_ratio"] == pytest.approx(z, rel=0.3)
    assert m["coherence_at_peak"] > 0.9


def test_rejects_unknown_type_and_multichannel():
    f = np.arange(0, 10.0)
    with pytest.raises(ValueError, match="frf_type"):
        modal_parameters(_FRF(f, f.astype(complex)), "compliance")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="one channel"):
        modal_parameters(_FRF(f, np.ones((2, 10), dtype=complex)), "receptance")
