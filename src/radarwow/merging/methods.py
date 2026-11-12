"""Radar-gauge merging methods as compared by Goudenhoofdt & Delobbe 2009, Hydrol. Earth Syst. Sci. 13, 195-203,
doi:10.5194/hess-13-195-2009 (their codes in brackets):

- `mean_field_bias` [MFB]: one factor sum(G) / sum(R) for the whole field.
- `brandes` [BRA]: Brandes 1975 (J. Appl. Meteor. 14, 1339-1345, doi:10.1175/1520-0450(1975)014<1339:OROBAR>2.0.CO;2);
  gauge/radar factors spread with Barnes weights exp(-d^2 / k), k = 1 / (2 * gauge density).
- `ordinary_kriging` [KRI]: gauges only.
- `conditional_merging` [KRE]: Sinclair & Pegram 2005 (Atmos. Sci. Lett. 6, 19-22, doi:10.1002/asl.85); kriged
  gauges plus the radar's deviation from its own kriged values at the gauges.
- `kriging_external_drift` [KED]: radar as the external drift of the gauge field.

As in the paper: a linear variogram (with no nugget the kriging weights do not depend on its slope, so nothing is
fitted per hour), the 20 nearest gauges for kriging, and the radar value at a gauge taken as the 3 x 3 cell mean
(`radar_at`). Negative merged values are set to 0. Coordinates are in km.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def radar_at(field, x, y, gx, gy, half: int = 1):
    """Mean of the (2 half + 1)^2 cells around each gauge (NaN cells ignored); ``x``, ``y`` are the grid axes."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    j = np.round((np.asarray(gx) - x[0]) / dx).astype(int)
    i = np.round((np.asarray(gy) - y[0]) / dy).astype(int)
    out = np.full(len(j), np.nan)
    for k, (ii, jj) in enumerate(zip(i, j)):
        if 0 <= ii < field.shape[0] and 0 <= jj < field.shape[1]:
            block = field[max(ii - half, 0):ii + half + 1, max(jj - half, 0):jj + half + 1]
            if np.isfinite(block).any():
                out[k] = np.nanmean(block)
    return out


def krige(src_xy, src_val, trg_xy, src_drift=None, trg_drift=None, n_nearest: int = 20, chunk: int = 20000):
    """Ordinary kriging (or kriging with one external drift) with a linear variogram, local neighbourhood."""
    src_xy, trg_xy = np.asarray(src_xy, float), np.asarray(trg_xy, float)
    src_val = np.asarray(src_val, float)
    n = min(n_nearest, len(src_val))
    drift = src_drift is not None
    tree = cKDTree(src_xy)
    out = np.empty(len(trg_xy))
    m = n + 1 + int(drift)
    for s in range(0, len(trg_xy), chunk):
        t = trg_xy[s:s + chunk]
        _, idx = tree.query(t, k=n)
        idx = idx.reshape(len(t), n)
        p = src_xy[idx]  # (T, n, 2)
        a = np.zeros((len(t), m, m))
        a[:, :n, :n] = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1)  # gamma(h) = h
        a[:, :n, n] = a[:, n, :n] = 1.0
        b = np.zeros((len(t), m))
        b[:, :n] = np.linalg.norm(p - t[:, None, :], axis=-1)
        b[:, n] = 1.0
        if drift:
            d = np.asarray(src_drift, float)[idx]
            a[:, :n, n + 1] = a[:, n + 1, :n] = d
            b[:, n + 1] = np.asarray(trg_drift, float)[s:s + chunk]
        # tiny jitter keeps the system solvable when two gauges share a location
        a[:, np.arange(n), np.arange(n)] += 1e-9
        w = np.linalg.solve(a, b[..., None])[..., 0][:, :n]
        out[s:s + chunk] = np.sum(w * src_val[idx], axis=1)
    return out


def mean_field_bias(radar_trg, g, r_g, min_mm: float = 0.1, **_):
    use = (g >= min_mm) | (r_g >= min_mm)
    f = np.sum(g[use]) / max(np.sum(r_g[use]), 1e-6)
    return radar_trg * f


def brandes(radar_trg, g, r_g, g_xy, trg_xy, area_km2, min_mm: float = 0.5, **_):
    use = (g >= min_mm) & (r_g >= min_mm)
    if use.sum() == 0:
        return radar_trg.copy()
    f = g[use] / r_g[use]
    k = 1.0 / (2 * len(g) / area_km2)  # km^2
    d2 = np.sum((trg_xy[:, None, :] - g_xy[use][None, :, :]) ** 2, axis=-1)
    w = np.exp(-d2 / k)
    ws = w.sum(axis=1)
    fx = np.where(ws > 1e-12, (w * f).sum(axis=1) / np.maximum(ws, 1e-12), np.sum(g[use]) / np.sum(r_g[use]))
    return radar_trg * fx


def ordinary_kriging(radar_trg, g, r_g, g_xy, trg_xy, **_):
    return np.clip(krige(g_xy, g, trg_xy), 0, None)


def conditional_merging(radar_trg, g, r_g, g_xy, trg_xy, **_):
    return np.clip(krige(g_xy, g, trg_xy) + (radar_trg - krige(g_xy, r_g, trg_xy)), 0, None)


def kriging_external_drift(radar_trg, g, r_g, g_xy, trg_xy, **_):
    return np.clip(krige(g_xy, g, trg_xy, src_drift=r_g, trg_drift=radar_trg), 0, None)


METHODS = {"radar only": lambda radar_trg, **_: radar_trg, "MFB": mean_field_bias, "Brandes": brandes,
           "kriging (gauges only)": ordinary_kriging, "conditional merging": conditional_merging, "KED": kriging_external_drift}


def merge(method: str, field, x, y, gx, gy, g, area_km2=None):
    """Merged field on the grid (``field`` with axes ``x``, ``y``) from gauges at (gx, gy) with values ``g``."""
    xx, yy = np.meshgrid(x, y)
    ok = np.isfinite(field)
    trg_xy = np.c_[xx[ok], yy[ok]]
    r_g = radar_at(field, x, y, gx, gy)
    keep = np.isfinite(r_g) & np.isfinite(g)
    area_km2 = area_km2 or ok.sum() * abs((x[1] - x[0]) * (y[1] - y[0]))
    est = METHODS[method](radar_trg=field[ok], g=np.asarray(g)[keep], r_g=r_g[keep], g_xy=np.c_[gx, gy][keep], trg_xy=trg_xy,
                          area_km2=area_km2)
    out = np.full(field.shape, np.nan)
    out[ok] = est
    return out


def leave_one_out(method: str, field, x, y, gx, gy, g, area_km2=None):
    """Estimate at each gauge from all the others (radar value at the left-out gauge = its 3 x 3 cell mean)."""
    gx, gy, g = map(np.asarray, (gx, gy, g))
    r_g = radar_at(field, x, y, gx, gy)
    ok = np.isfinite(field)
    area_km2 = area_km2 or ok.sum() * abs((x[1] - x[0]) * (y[1] - y[0]))
    est = np.full(len(g), np.nan)
    for k in range(len(g)):
        if not np.isfinite(r_g[k]):
            continue
        others = np.ones(len(g), bool)
        others[k] = False
        others &= np.isfinite(r_g) & np.isfinite(g)
        est[k] = METHODS[method](radar_trg=np.array([r_g[k]]), g=g[others], r_g=r_g[others], g_xy=np.c_[gx, gy][others],
                                 trg_xy=np.array([[gx[k], gy[k]]]), area_km2=area_km2)[0]
    return est
