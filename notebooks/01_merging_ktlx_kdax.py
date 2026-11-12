# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Radar-gauge merging, verified at gauges left out: KTLX and KDAX
#
# **Question.** The radar-only hourly totals of radarwow-verify notebook 05 are within a few per cent of the
# gauge-corrected reference on KTLX but still ~20 % low on KDAX. How much do rain gauges improve them, and which
# merging method is best?
#
# **Methods** (`radarwow.merging`), the set compared by Goudenhoofdt & Delobbe (2009, HESS,
# doi:10.5194/hess-13-195-2009): mean field bias (MFB), Brandes (1975) spatially varying factors, ordinary kriging of
# the gauges alone, conditional merging (Sinclair & Pegram 2005) and kriging with the radar as external drift (KED).
#
# **Verification.** Leave-one-out: each gauge in turn is removed, the field is merged from the others, and the
# estimate at the removed gauge is compared with what it measured. The MRMS gauge-corrected product cannot be the
# reference here: it was built with these same gauges.
#
# **Data.** Radar: `radarwow.pipeline.hourly_totals` (the chain of verify notebooks 01-05). Gauges: hourly HADS
# reports from the Iowa Environmental Mesonet archive (free). HADS data come with little quality control, so the
# results are shown twice: with basic checks only, and with an extra gross-error check (see section 1).

# %%
# Colab: uncomment to install
# !pip install -q cfgrib eccodes pysteps opencv-python-headless rasterio "git+https://github.com/manmeet3591/radarwow-pipeline" "git+https://github.com/manmeet3591/radarwow-merging"

# %%
import os
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from radarwow.merging import METHODS, leave_one_out, merge, radar_at
from radarwow.pipeline import hourly_totals
from radarwow.verify import basic_qc, fetch_hads, fetch_mrms, gross_error_qc

FIG = "../docs/figures"
CACHE = os.path.expanduser("~/.cache/radarwow")
CASES = {"KTLX": dict(network="OK_DCP", start="2022-05-05 07:00", end="2022-05-05 11:00", n_tilts=1, blockage=False),
         "KDAX": dict(network="CA_DCP", start="2021-10-24 12:00", end="2021-10-24 16:00", n_tilts=3, blockage=True)}
MAX_KM = 145  # keep gauges a 3 x 3 window inside the 150 km radar disk

# %% [markdown]
# ## 1. Radar totals and gauges

# %%
data = {}
qc_rows = []
for site, c in CASES.items():
    rad = hourly_totals(site, c["start"], c["end"], n_tilts=c["n_tilts"], blockage=c["blockage"])
    lat0, lon0 = rad.attrs["radar_lat"], rad.attrs["radar_lon"]
    g = fetch_hads(c["network"], lat0, lon0, MAX_KM, c["start"], c["end"], f"{CACHE}/gauges")
    g["gx"] = (g["lon"] - lon0) * 111.2 * np.cos(np.deg2rad(lat0))
    g["gy"] = (g["lat"] - lat0) * 111.2
    x, y = rad["x"].values, rad["y"].values
    g["radar_mm"] = np.nan
    for t, idx in g.groupby("time").groups.items():
        field = rad["RAIN"].sel(time=t).values
        g.loc[idx, "radar_mm"] = radar_at(field, x, y, g.loc[idx, "gx"].values, g.loc[idx, "gy"].values)
    g = gross_error_qc(basic_qc(g, radar_col="radar_mm"), radar_col="radar_mm")
    for reason, n in g["qc"].value_counts().items():
        qc_rows.append(dict(site=site, check=reason, gauge_hours=n))
    basic_ok = ~g["qc"].isin(["impossible value", "gauge dry under radar rain", "gauge rain with no radar echo", "no radar value"])
    data[site] = (rad, {"basic QC": g[basic_ok].reset_index(drop=True), "basic + gross-error QC": g[g["qc"] == "ok"].reset_index(drop=True)})
display(Markdown("**Gauge quality control** (gauge-hours). The gross-error check flags gauge-hours that differ from the radar by "
                 "more than 5 mm and a factor of 3 after scaling the radar by the hour's median gauge/radar ratio, and drops "
                 "stations flagged twice. It uses the radar, so it leans the comparison towards the radar; both versions "
                 "are shown below.\n\n" + pd.DataFrame(qc_rows).pivot_table(index="check", columns="site", values="gauge_hours", fill_value=0).to_markdown()))

# %% [markdown]
# ## 2. Leave-one-out at the gauges

# %%
loo = []
for site, (rad, gsets) in data.items():
  x, y = rad["x"].values, rad["y"].values
  for qc_name, g in gsets.items():
    for t, gt in g.groupby("time"):
        field = rad["RAIN"].sel(time=t).values
        for m in METHODS:
            est = leave_one_out(m, field, x, y, gt["gx"].values, gt["gy"].values, gt["gauge_mm"].values)
            loo.append(pd.DataFrame(dict(site=site, qc=qc_name, time=t, method=m, station=gt["station"].values, gauge=gt["gauge_mm"].values, est=est)))
        # MRMS radar-only at the gauges: a fair peer (no gauges in it); the gauge-corrected product is not independent
        for prod, label in [("radar_only", "MRMS radar-only"), ("gauge_corrected", "MRMS gauge-corrected (uses these gauges)")]:
            mr = fetch_mrms(prod, pd.Timestamp(t) + pd.Timedelta(hours=1), f"{CACHE}/mrms")
            v = mr.sel(latitude=gt["lat"].to_xarray(), longitude=gt["lon"].to_xarray(), method="nearest").values
            loo.append(pd.DataFrame(dict(site=site, qc=qc_name, time=t, method=label, station=gt["station"].values, gauge=gt["gauge_mm"].values, est=v)))
loo = pd.concat(loo, ignore_index=True)


def summary(d):
    d = d.dropna(subset=["est"])
    e, g = d["est"].values, d["gauge"].values
    return pd.Series(dict(n=len(d), bias=e.sum() / g.sum(), MAE=np.mean(np.abs(e - g)), RMSE=np.sqrt(np.mean((e - g) ** 2)),
                          corr=np.corrcoef(e, g)[0, 1]))


order = list(METHODS) + ["MRMS radar-only", "MRMS gauge-corrected (uses these gauges)"]
for site in CASES:
  for qc_name in ["basic QC", "basic + gross-error QC"]:
    s = loo[(loo.site == site) & (loo.qc == qc_name)].groupby("method").apply(summary).loc[order]
    s["MAE vs radar only"] = (s["MAE"] / s.loc["radar only", "MAE"] - 1).map(lambda v: f"{v:+.0%}")
    display(Markdown(f"**{site}, {qc_name}**: all gauge-hours (bias = sum of estimates / sum of gauges; MAE, RMSE in mm)\n\n"
                     + s.round({"bias": 2, "MAE": 2, "RMSE": 2, "corr": 2}).to_markdown()))

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for row, site in zip(axes, CASES):
    d = loo[(loo.site == site) & (loo.qc == "basic + gross-error QC")]
    lim = np.nanpercentile(d["gauge"], 99.5) * 1.1
    for ax, m in zip(row, ["radar only", "MFB", "KED"]):
        dm = d[d.method == m]
        ax.scatter(dm["gauge"], dm["est"], s=6, alpha=0.5)
        ax.plot([0, lim], [0, lim], "r")
        ax.set(xlim=(0, lim), ylim=(0, lim), title=f"{site}: {m} (gross-error QC)", xlabel="gauge (mm/h)", ylabel="estimate at the left-out gauge (mm/h)")
fig.tight_layout()
fig.savefig(f"{FIG}/01_loo_scatter.png", dpi=100)
plt.show()

# %% [markdown]
# ## 3. Merged maps

# %%
fig, axes = plt.subplots(2, 4, figsize=(21, 10))
for row, (site, hour) in zip(axes, [("KTLX", "2022-05-05 08:00"), ("KDAX", "2021-10-24 13:00")]):
    rad, gsets = data[site]
    g = gsets["basic + gross-error QC"]
    t = pd.Timestamp(hour)
    x, y = rad["x"].values, rad["y"].values
    field = rad["RAIN"].sel(time=t).values
    gt = g[g.time == t]
    vmax = 20 if site == "KTLX" else 12
    panels = [("radar only", field), ("MFB", merge("MFB", field, x, y, gt.gx.values, gt.gy.values, gt.gauge_mm.values)),
              ("KED", merge("KED", field, x, y, gt.gx.values, gt.gy.values, gt.gauge_mm.values))]
    for ax, (title, f) in zip(row[:3], panels):
        p = ax.pcolormesh(x, y, np.ma.masked_invalid(f), cmap="Blues", vmin=0, vmax=vmax, shading="auto")
        ax.scatter(gt.gx, gt.gy, c=gt.gauge_mm, cmap="Blues", vmin=0, vmax=vmax, s=18, edgecolors="k", linewidths=0.5)
        ax.set(title=f"{site} {t:%H}-{t.hour + 1:02d} UTC: {title}", aspect=1)
    ref = fetch_mrms("gauge_corrected", t + pd.Timedelta(hours=1), f"{CACHE}/mrms")
    lat0, lon0 = rad.attrs["radar_lat"], rad.attrs["radar_lon"]
    box = ref.sel(latitude=slice(lat0 + 1.36, lat0 - 1.36), longitude=slice(lon0 - 1.7, lon0 + 1.7))
    bx = (box.longitude - lon0) * 111.2 * np.cos(np.deg2rad(lat0))
    by = (box.latitude - lat0) * 111.2
    row[3].pcolormesh(bx, by, np.ma.masked_invalid(box.values), cmap="Blues", vmin=0, vmax=vmax, shading="auto")
    row[3].set(title="MRMS gauge-corrected", aspect=1, xlim=(-150, 150), ylim=(-150, 150))
    fig.colorbar(p, ax=row, label="mm in the hour", shrink=0.8)
fig.savefig(f"{FIG}/02_merged_maps.png", dpi=85, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Takeaways
# - **Gauge quality control decides the outcome on KTLX.** With basic checks, every merging method is worse than the
#   radar alone at the left-out gauges. Six bad gauge-hours (of 303) cause that: one gauge reporting 46 mm under
#   4 mm of radar and MRMS rain, one reporting an hour late, and gauges stuck at 0 under 9-13 mm. Without them the
#   radar-gauge correlation goes from 0.60 to 0.92, and conditional merging cuts the error by 19 % (KED 11 %, MFB 9 %).
# - **KTLX needs the radar.** Kriging the gauges alone is 33 % worse than the radar: convective rain with about one
#   gauge per 700 km^2. This is the summer result of Goudenhoofdt & Delobbe 2009.
# - **KDAX gains a lot from gauges, with either QC:** KED, conditional merging and kriging cut the error by 34 %
#   (basic QC) or about 50 % (with the gross-error check, which removes 11 % of gauge-hours, so the 34 % is the safer
#   number). Bias goes from 0.46-0.51 to 1.0. Kriging the gauges alone does as well as KED: 230 gauges in steady rain
#   carry most of the information (the paper's winter result). Mean field bias alone gains little (0 to 15 %): it
#   doubles the whole field, overshooting in the lightly raining south.
# - **Against MRMS.** With the gross-error check, KED's leave-one-out error on KDAX (1.77 mm) is slightly lower than
#   MRMS gauge-corrected's (1.89 mm), although MRMS uses these same gauges in its analysis. MRMS radar-only, which
#   uses no gauges, is worse than our radar-only on both cases.
# - **Recommendation:** conditional merging or KED, always behind gauge QC; MFB only where gauges are very sparse.
# - **Limits:** two cases, four hours each; leave-one-out at gauges is the only independent check here (no separate
#   gauge network); the gross-error check uses the radar, which favours the radar slightly.
