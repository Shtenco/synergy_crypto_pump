import numpy as np

from crypto_pump.v2.bocd import bocd_change_probability


def test_bocd_reacts_to_regime_shift():
    x = np.r_[np.zeros(80), np.ones(8) * 6.0]
    p = bocd_change_probability(x, hazard_bars=40, max_run_length=64)
    baseline = np.nanmedian(p[30:70])
    shifted = np.nanmax(p[80:84])
    assert shifted > baseline
    assert shifted > 0.05
