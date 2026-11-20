#!/usr/bin/env python3
"""Characteristic variability timescale of the AGN as a function of black-hole mass,
from ensemble structure functions of the rest-frame r-band light curves.

For each mass bin, a random sample of light curves (each thinned to at most
--max-points observations, so long curves do not dominate) contributes every pair of
observations: rest-frame lag dt and magnitude difference dm. Per lag bin,

    SF(dt)^2 = (0.7413 * IQR(dm))^2 - <err_i^2 + err_j^2>

i.e. a robust (outlier-resistant) spread of dm with the photometric noise removed.
A damped random walk has SF(dt) = SF_inf * sqrt(1 - exp(-dt / tau)): it rises as
dt^0.5 and flattens at dt ~ tau, the characteristic timescale. That form is fitted
over --fit-range days per mass bin; uncertainties are jackknife over 10 groups of
light curves.

Two samples: every redshift, and --zmin <= z < --zmax. Mass rises with redshift in
this catalogue, and redshift changes both the rest-frame baseline (higher z = shorter
rest-frame coverage, which biases tau low) and the rest-frame wavelength the r band
sees; the narrow-z sample removes most of that.

Caveats (qualitative use only): tau is biased low unless the rest-frame baseline is
>~ 10 tau (median ~820 rest-frame days here); long lags come only from long-baseline
curves; luminosity is not controlled.

    python plot_sf_by_mass.py --out cluster_results/sf_by_mass.png
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

from plot_span_window_contour import INK, INK2, MUTED, RULE  # noqa: E402

MASS_EDGES = [7.0, 8.0, 8.4, 8.8, 9.2, 9.6, 10.5]
SHADES = ["#b9d4f3", "#8fbbec", "#5b9be2", "#2a78d6", "#1c5aa6", "#123c70"]  # light -> dark
SAMPLE_COL = {"all": "#2a78d6", "zcut": "#eb6834"}
DT_EDGES = np.logspace(0, np.log10(2500), 32)          # rest-frame days
DM_EDGES = np.linspace(-3, 3, 1201)                     # 0.005 mag bins
NGROUP = 10


def drw(dt, sf_inf, tau, floor=0.0):
    """Damped random walk SF, plus a lag-independent floor for noise the quoted
    photometric errors miss (it shows as a flat SF at the shortest lags)."""
    return np.sqrt(sf_inf ** 2 * (1 - np.exp(-dt / tau)) + floor ** 2)


def iqr_sigma(h):
    """0.7413 * IQR from a histogram over DM_EDGES (the Gaussian-equivalent sigma)."""
    c = np.cumsum(h)
    if c[-1] < 40:
        return np.nan
    q1, q3 = np.interp([0.25 * c[-1], 0.75 * c[-1]], c, DM_EDGES[1:])
    return 0.7413 * (q3 - q1)


def sf(hist, noise, cnt):
    s = np.array([iqr_sigma(h) for h in hist])
    n = np.divide(noise, cnt, out=np.full(len(cnt), np.nan), where=cnt > 0)
    v = s ** 2 - n
    return np.sqrt(np.where(v > 0, v, np.nan))


def fit_tau(dt, y, lo, hi):
    ok = np.isfinite(y) & (dt >= lo) & (dt <= hi)
    if ok.sum() < 5:
        return np.nan, np.nan, np.nan
    try:
        (s_inf, tau, fl), _ = curve_fit(drw, dt[ok], y[ok], p0=(np.nanmax(y[ok]), 100.0, 0.02),
                                        bounds=([0, 1, 0], [5, 1e5, 0.3]), maxfev=20000)
        return s_inf, tau, fl
    except RuntimeError:
        return np.nan, np.nan, np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/records/agn_test_rest/fold_0/test")
    ap.add_argument("--band", default="r")
    ap.add_argument("--per-bin", type=int, default=3000, help="light curves per mass bin")
    ap.add_argument("--max-points", type=int, default=150, help="observations kept per curve")
    ap.add_argument("--min-points", type=int, default=50)
    ap.add_argument("--zmin", type=float, default=1.0)
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--fit-range", type=float, nargs=2, default=[1.0, 1000.0],
                    help="lags fitted (rest-frame days); a tau above a third of the upper end "
                         "is shown as a lower limit, not a measurement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("cluster_results/sf_by_mass.png"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    idx = pd.read_csv(os.path.join(a.src, "lc_index.csv"), dtype={"lcid": str, "source_id": str})
    idx = idx[(idx["band_name"] == a.band) & idx["mbh_valid"].astype(bool) & (idx["LOGMBH"] > 0)
              & (idx["n_points"] >= a.min_points)]
    idx = idx.assign(mbin=pd.cut(idx["LOGMBH"], MASS_EDGES, right=False, labels=False)).dropna(subset=["mbin"])
    pick = (idx.groupby("mbin", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), a.per_bin), random_state=a.seed)))
    meta = pick.set_index("lcid")[["mbin", "Z_FIT", "LOGMBH"]]
    meta["group"] = rng.integers(0, NGROUP, len(meta))
    print(f"{len(meta):,} light curves sampled:", meta["mbin"].value_counts().sort_index().tolist())

    nm, nt, nd = len(MASS_EDGES) - 1, len(DT_EDGES) - 1, len(DM_EDGES) - 1
    H = {s: np.zeros((nm, NGROUP, nt, nd), np.int32) for s in SAMPLE_COL}
    N = {s: np.zeros((nm, NGROUP, nt)) for s in SAMPLE_COL}
    C = {s: np.zeros((nm, NGROUP, nt)) for s in SAMPLE_COL}
    seen = 0
    for p in sorted(glob.glob(os.path.join(a.src, "*.record"))):
        for raw in tf.data.TFRecordDataset(p).as_numpy_iterator():
            ex = tf.train.SequenceExample.FromString(raw)
            lc = ex.context.feature["ID"].bytes_list.value[0].decode()
            if lc not in meta.index:
                continue
            fl = ex.feature_lists.feature_list
            t, m, e = (np.asarray(fl[k].feature[0].float_list.value, float) for k in ("mjd", "mag", "err"))
            if len(t) > a.max_points:
                keep = np.sort(rng.choice(len(t), a.max_points, replace=False))
                t, m, e = t[keep], m[keep], e[keep]
            i, j = np.triu_indices(len(t), 1)
            dt, dm, nz = t[j] - t[i], m[j] - m[i], e[i] ** 2 + e[j] ** 2
            ti = np.digitize(dt, DT_EDGES) - 1
            di = np.clip(np.digitize(dm, DM_EDGES) - 1, 0, nd - 1)
            ok = (ti >= 0) & (ti < nt)
            ti, di, nz = ti[ok], di[ok], nz[ok]
            row = meta.loc[lc]
            mb, g, z = int(row["mbin"]), int(row["group"]), row["Z_FIT"]
            samples = ["all"] + (["zcut"] if a.zmin <= z < a.zmax else [])
            hc = np.bincount(ti * nd + di, minlength=nt * nd).reshape(nt, nd)
            nc = np.bincount(ti, weights=nz, minlength=nt)
            cc = np.bincount(ti, minlength=nt)
            for s in samples:
                H[s][mb, g] += hc.astype(np.int32)
                N[s][mb, g] += nc
                C[s][mb, g] += cc
            seen += 1
    print(f"{seen:,} light curves read")

    dtc = np.sqrt(DT_EDGES[:-1] * DT_EDGES[1:])
    curves, rows = {}, []
    for s in SAMPLE_COL:
        for mb in range(nm):
            full = sf(H[s][mb].sum(0), N[s][mb].sum(0), C[s][mb].sum(0))
            sinf, tau, floor = fit_tau(dtc, full, *a.fit_range)
            jk = []
            for g in range(NGROUP):                       # leave one group of curves out
                keepg = [k for k in range(NGROUP) if k != g]
                y = sf(H[s][mb][keepg].sum(0), N[s][mb][keepg].sum(0), C[s][mb][keepg].sum(0))
                jk.append(fit_tau(dtc, y, *a.fit_range)[1])
            jk = np.array(jk)
            jk = jk[np.isfinite(jk)]
            tau_err = np.sqrt((len(jk) - 1) / len(jk) * np.sum((jk - jk.mean()) ** 2)) if len(jk) > 2 else np.nan
            sub = meta[(meta["mbin"] == mb)]
            if s == "zcut":
                sub = sub[(sub["Z_FIT"] >= a.zmin) & (sub["Z_FIT"] < a.zmax)]
            curves[(s, mb)] = full
            rows.append({"sample": s, "mass_lo": MASS_EDGES[mb], "mass_hi": MASS_EDGES[mb + 1],
                         "mass_median": sub["LOGMBH"].median(), "z_median": sub["Z_FIT"].median(),
                         "curves": len(sub), "sf_inf": sinf, "tau_days": tau, "tau_err": tau_err,
                         "noise_floor": floor,
                         "measured": bool(np.isfinite(tau) and tau <= a.fit_range[1] / 3)})
    st = pd.DataFrame(rows)
    print(st.round(3).to_string(index=False))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"wspace": 0.28,
                                                                 "width_ratios": [1, 1, 0.9]})

    def frame(ax):
        ax.grid(True, which="major", color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0, which="both")

    for ax, s, ttl in ((axes[0], "all", "All redshifts"),
                       (axes[1], "zcut", f"{a.zmin:g} ≤ z < {a.zmax:g} only")):
        for mb in range(nm):
            r = st[(st["sample"] == s)].iloc[mb]
            y = curves[(s, mb)]
            ax.plot(dtc, y, "o", color=SHADES[mb], ms=3.5, mec="white", mew=0.4)
            if np.isfinite(r["tau_days"]):
                xx = np.logspace(0, np.log10(2500), 200)
                tl = (f"τ ≈ {r['tau_days']:.0f} d" if r["measured"]
                      else f"τ > {a.fit_range[1] / 3:.0f} d")
                ax.plot(xx, drw(xx, r["sf_inf"], r["tau_days"], r["noise_floor"]),
                        color=SHADES[mb], lw=1.4,
                        label=f"{MASS_EDGES[mb]:g}–{MASS_EDGES[mb + 1]:g}  "
                              f"({tl}, {int(r['curves']):,} curves)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rest-frame lag Δt (days)")
        ax.set_ylabel("structure function, noise removed (mag)")
        ax.set_title(f"{ttl}: ensemble structure function by log M_BH", loc="left", fontsize=10)
        ax.axvspan(*a.fit_range, color=RULE, alpha=0.25, lw=0, zorder=0)
        ax.legend(frameon=False, fontsize=7, loc="lower right", title="log M_BH bin",
                  title_fontsize=7.5)
        frame(ax)

    ax = axes[2]
    for s, lab in (("all", "all redshifts"), ("zcut", f"{a.zmin:g} ≤ z < {a.zmax:g}")):
        g = st[st["sample"] == s]
        lim = a.fit_range[1] / 3
        ok = g["measured"].to_numpy(bool)
        ax.plot(g["mass_median"], np.minimum(g["tau_days"], lim), color=SAMPLE_COL[s], lw=1.2,
                alpha=0.6)
        ax.errorbar(g.loc[ok, "mass_median"], g.loc[ok, "tau_days"], yerr=g.loc[ok, "tau_err"],
                    fmt="o", color=SAMPLE_COL[s], ms=5.5, mec="white", mew=0.8, capsize=0, label=lab)
        if (~ok).any():                                    # lower limits: open, with arrows
            ax.errorbar(g.loc[~ok, "mass_median"], np.full((~ok).sum(), lim), yerr=lim * 0.4,
                        lolims=True, fmt="o", mfc="white", color=SAMPLE_COL[s], ms=5.5, capsize=3)
    g = st[(st["sample"] == "all")].dropna(subset=["tau_days"])
    if len(g):
        x0, y0 = g["mass_median"].median(), np.nanmedian(g["tau_days"])
        xx = np.linspace(7.6, 10.0, 50)
        ax.plot(xx, y0 * 10 ** (0.38 * (xx - x0)), color=MUTED, lw=1, ls=(0, (4, 3)),
                label="slope 0.38 (Burke et al. 2021), for shape only")
    ax.axhline(a.fit_range[1] / 3, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.text(7.6, a.fit_range[1] / 3 * 1.1, "above: lower limits only (open, arrows)",
            fontsize=7, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(10, 2000)
    ax.set_xlabel(r"median $\log M_{\rm BH}$ of the bin")
    ax.set_ylabel("damped-random-walk τ (rest-frame days)")
    ax.set_title("Fitted timescale vs mass (jackknife bars)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    frame(ax)
    fig.suptitle("Characteristic variability timescale vs black-hole mass, ZTF r band, rest frame",
                 x=0.01, ha="left", y=1.03, fontsize=11.5)
    fig.text(0.01, 0.975, f"SF² = (0.74·IQR Δm)² − ⟨σᵢ²+σⱼ²⟩; "
             f"fit SF∞√(1−e^(−Δt/τ)) over the shaded {a.fit_range[0]:g}–"
             f"{a.fit_range[1]:g} d with a free noise floor. Qualitative: τ is biased low unless "
             "the rest-frame baseline (median ~820 d) is ≫ τ.", fontsize=8, color=MUTED, ha="left")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")
    st.to_csv(a.out.with_suffix(".csv"), index=False)
    print(f"wrote {a.out} and {a.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
