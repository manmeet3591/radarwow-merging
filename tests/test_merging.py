import numpy as np

from radarwow.merging import krige, leave_one_out, merge, radar_at

X = np.arange(-50, 50) + 0.5
Y = X[::-1]
XX, YY = np.meshgrid(X, Y)
RNG = np.random.default_rng(0)
GX, GY = RNG.uniform(-45, 45, 40), RNG.uniform(-45, 45, 40)


def truth():
    return 5 + 3 * np.sin(XX / 15) + 2 * np.cos(YY / 20)


def test_kriging_is_exact_at_gauges_and_smooth():
    vals = np.sin(GX / 10) + GY / 50
    est = krige(np.c_[GX, GY], vals, np.c_[GX, GY])
    assert np.allclose(est, vals, atol=1e-5)


def test_ked_reproduces_linear_drift():
    # gauge = 2 * radar + 1 everywhere: KED with the radar as drift recovers it exactly
    r = RNG.uniform(0, 10, 40)
    t = RNG.uniform(0, 10, 5)
    trg = np.c_[RNG.uniform(-40, 40, 5), RNG.uniform(-40, 40, 5)]
    est = krige(np.c_[GX, GY], 2 * r + 1, trg, src_drift=r, trg_drift=t)
    assert np.allclose(est, 2 * t + 1, atol=1e-4)


def test_mfb_removes_a_uniform_bias():
    radar = 0.5 * truth()
    g = radar_at(truth(), X, Y, GX, GY)
    out = merge("MFB", radar, X, Y, GX, GY, g)
    assert np.allclose(out, truth(), rtol=0.02)


def test_merging_beats_radar_on_leave_one_out():
    radar = 0.6 * truth() + RNG.normal(0, 0.3, truth().shape)
    g = radar_at(truth(), X, Y, GX, GY)
    err = {m: np.nanmean(np.abs(leave_one_out(m, radar, X, Y, GX, GY, g) - g)) for m in ["radar only", "MFB", "conditional merging", "KED"]}
    assert err["MFB"] < 0.5 * err["radar only"] and err["KED"] < err["radar only"] and err["conditional merging"] < err["radar only"]
