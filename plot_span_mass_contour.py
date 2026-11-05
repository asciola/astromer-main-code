#!/usr/bin/env python3
"""Contour maps of the gain over the reference model across black-hole mass and time
span, one panel per window size.

Pavlos' layout: x = true black-hole mass (LOGMBH), y = time span, one panel per
window size, one row per attention
type (--families: standard attention = base, MHLA at each latent size). Every
(mass bin, span bin) cell gets

    MSE(reference) / MSE(model)          > 1 = the model beats the reference

on the sources in that cell. The reference is the base model at window 200 (--base),
so the standard-attention window-200 panel is 1 everywhere by construction. Mass and
span are close to independent (Spearman ~0), so the two axes do not explain each
other; window size and span are related only through what each window can see (see
lc_time_features.py). Both axes are evenly spaced bins labelled with their ranges,
because the sources are concentrated in the top ~150 days of span and the middle of
the mass range. Dots mark the cells actually computed; colour between them is
interpolated. Cells with fewer than --min-n sources are left blank.

WHICH SPAN (--span-of)
  curve   (default) days from first to last --band observation of the whole curve.
  window  days covered by ONE window of the panel's own window size (wspan_<W> from
          lc_time_features.py, averaged over the curve's windows), i.e. the time range
          the encoder sees at once. Needs --time-features. The bins are in days and
          shared by every panel, so a window-800 panel fills only the top rows (one
          window is nearly the whole curve) while window 200 spreads over 600-2100 days.
          At window 200 this axis is almost the inverse of point count (Spearman -0.98
          with r-band points), so read it as "sparse vs dense curves" there.

NULL (target m only)
  Splitting by TRUE mass rewards any model that shrinks less toward the mean than the
  reference (see length_mass_gain.py): it looks better at both mass ends and worse in
  the middle even if its gain has nothing to do with mass. For --targets m the script
  also writes contour_span_mass_excess_m.png, the same map divided by what a uniform
  improvement would give in each cell (null_tercile_ratios from length_mass_gain.py,
  run per span bin with that bin's real R^2 values). 1 = no mass-specific effect. For
  z and e, mass is not the target, so no such correction is needed.

Reads the out-of-fold predictions.csv from an EMBEDDINGS-ONLY fit_regressor.py run and
lc_index.csv. Writes, to --out-dir:

  contour_span_mass_<t>.png         one figure per target
  contour_span_mass_excess_m.png    target m divided by the uniform-improvement null
  contour_span_mass.csv             every cell: family, window, span bin, mass bin, n,
                                    MSE, reference MSE, ratio, null, excess

--cells redraws from a saved contour_span_mass.csv without reading the predictions.

    python plot_span_mass_contour.py --pred cluster_results/agn_mlp/predictions.csv \\
        --index $IDX --out-dir cluster_results/contour_mass_mlp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from length_mass_gain import null_tercile_ratios
from plot_span_window_contour import CONTROL, DIVERGE, INK, INK2, MUTED, RULE, TLABEL, \
    fam_title, parse


SPAN_BINS = {"curve": [0, 2000, 2150, 2200, 2250, 2275, np.inf],
             # shared by every panel so the day scale is the same; windows of 600+ points
             # cover most of the curve, so their panels fill only the top rows
             "window": [0, 600, 800, 1000, 1200, 1500, 1800, 2100, 2250, np.inf]}


def bin_label(lo, hi, unit, fmt):
    if lo == 0:
        return f"< {hi:{fmt}}{unit}"
    if not np.isfinite(hi):
        return f"{lo:{fmt}}+{unit}"
    return f"{lo:{fmt}}–{hi:{fmt}}{unit}"


def idx_lcids(index, band):
    """lcid -> source_id for one band (to join lc_time_features.csv onto sources)."""
    idx = pd.read_csv(index, dtype={"source_id": str, "lcid": str},
                      usecols=["lcid", "source_id", "band_name"])
    return idx[idx["band_name"] == band].drop_duplicates("source_id")[["lcid", "source_id"]]


def cells(a):
    """MSE ratio over the reference for every (target, family, window, span, mass) cell."""
    rng = np.random.default_rng(a.seed)
    idx = pd.read_csv(a.index, dtype={"source_id": str},
                      usecols=["source_id", "band_name", "mjd_min", "mjd_max",
                               "LOGMBH", "mbh_valid"])
    b = idx[idx["band_name"] == a.band].drop_duplicates("source_id").set_index("source_id")
    if b.empty:
        print(f"no band {a.band!r} in {a.index}", file=sys.stderr)
        return None
    if a.span_of == "curve":
        curve_span = b["mjd_max"] - b["mjd_min"]
        span = lambda ws: curve_span                                     # noqa: E731
    else:
        if a.time_features is None:
            print("--span-of window needs --time-features", file=sys.stderr)
            return None
        tf_ = pd.read_csv(a.time_features, dtype={"lcid": str})
        tf_ = idx_lcids(a.index, a.band).merge(tf_, on="lcid").set_index("source_id")
        span = lambda ws: tf_[f"wspan_{ws}"]                             # noqa: E731
    # LOGMBH == 0 is the catalogue's "not measured" sentinel; mbh_valid excludes it
    mass = b["LOGMBH"].where(b["mbh_valid"].astype(bool))

    pred = pd.read_csv(a.pred, dtype={"source_id": str})
    pred = pred[pred["model"] != CONTROL]
    names = sorted(pred["model"].unique())
    base_hits = [n for n in names if a.base in n]
    if len(base_hits) != 1:
        print(f"--base {a.base!r} matched {base_hits or 'nothing'}; models are:\n  " +
              "\n  ".join(names), file=sys.stderr)
        return None
    base = base_hits[0]
    models = {parse(n): n for n in names if parse(n)[0] in a.families}
    nm = len(a.mass_bins) - 1

    rows = []
    for t in a.targets:
        wide = pred.pivot_table(index="source_id", columns="model", values=f"{t}_pred")
        y = pred.drop_duplicates("source_id").set_index("source_id")[f"{t}_true"] \
            .reindex(wide.index)
        mt = mass.reindex(wide.index)
        ok = y.notna() & mt.notna() & wide[base].notna()
        wide, y = wide[ok], y[ok].to_numpy(float)
        mb = pd.cut(mt[ok].to_numpy(float), a.mass_bins, right=False, labels=False)
        eb = (wide[base].to_numpy(float) - y) ** 2
        for (lat, ws), mdl in models.items():
            # with --span-of window the span is that of THIS model's windows
            sb = pd.cut(span(ws).reindex(wide.index).to_numpy(float), a.span_bins,
                        right=False, labels=False)
            em = (wide[mdl].to_numpy(float) - y) ** 2
            for k in range(len(a.span_bins) - 1):
                sk = (sb == k) & np.isfinite(em)
                null = np.full(nm, np.nan)
                if t == "m" and sk.sum() >= 50:
                    v = y[sk].var()
                    r2b, r2m = 1 - eb[sk].mean() / v, 1 - em[sk].mean() / v
                    null = null_tercile_ratios(y[sk], mb[sk], nm, r2b, r2m, rng, a.null_draws)
                for j in range(nm):
                    sel = sk & (mb == j)
                    if sel.sum() < a.min_n:
                        continue
                    ratio = eb[sel].mean() / em[sel].mean()
                    rows.append({"target": t, "span_of": a.span_of, "family": lat, "window": ws,
                                 "span_bin": k, "span_lo": a.span_bins[k],
                                 "span_hi": a.span_bins[k + 1],
                                 "mass_bin": j, "mass_lo": a.mass_bins[j],
                                 "mass_hi": a.mass_bins[j + 1], "n": int(sel.sum()),
                                 "mse": em[sel].mean(), "mse_base": eb[sel].mean(),
                                 "mse_ratio": ratio, "null_mse_ratio": null[j],
                                 "excess": ratio / null[j] if np.isfinite(null[j]) else np.nan})
        print(f"{t}: {len(y):,} sources with a valid mass", flush=True)
    return pd.DataFrame(rows)


def draw(res, t, value, fams, wins, title, cbar_label, path, span_of="curve"):
    rt = res[res["target"] == t]
    sbins = sorted(rt["span_bin"].unique())
    mbins = sorted(rt["mass_bin"].unique())
    ylab, xlab = [], []
    for k in sbins:
        c = rt[rt["span_bin"] == k].iloc[0]
        lab = bin_label(c["span_lo"], c["span_hi"], " d", ".0f")
        if span_of == "curve":      # with window spans the row counts differ per panel
            lab += f"\n{rt[rt['span_bin'] == k].groupby('mass_bin')['n'].max().sum():,}"
        ylab.append(lab)
    for j in mbins:
        c = rt[rt["mass_bin"] == j].iloc[0]
        xlab.append(bin_label(c["mass_lo"], c["mass_hi"], "", ".1f"))
    # one colour scale per figure, symmetric around 1, shared by every panel; set by the
    # 98th percentile of the well-populated cells so a few extreme cells cannot wash out
    # the rest (extend="both" still shows anything beyond it)
    big = rt[(rt["n"] >= 200) & (np.abs(rt["mse_ratio"] - 1) > 1e-9)][value].dropna()
    lim = float(np.quantile(np.abs((big if len(big) else rt[value].dropna()) - 1), 0.98))
    lim = max(0.02, np.ceil(lim / 0.02) * 0.02)
    levels = np.linspace(1 - lim, 1 + lim, 17)
    norm = TwoSlopeNorm(1.0, 1 - lim, 1 + lim)

    fig, axes = plt.subplots(len(fams), len(wins), figsize=(2.75 * len(wins) + 1.6,
                                                            2.75 * len(fams) + 0.9),
                             sharex=True, sharey=True, squeeze=False)
    cf = None
    for r, fam in enumerate(fams):
        for c, ws in enumerate(wins):
            ax = axes[r][c]
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            g = rt[(rt["family"] == fam) & (rt["window"] == ws)]
            if r == 0:
                ax.set_title(f"window {ws}", fontsize=10.5, pad=8)
            if c == 0:
                ax.set_ylabel(fam_title(fam), fontsize=10.5, labelpad=8)
            if g.empty:
                continue
            Z = g.pivot_table(index="span_bin", columns="mass_bin", values=value) \
                .reindex(index=sbins, columns=mbins).to_numpy(float)
            X, Y = np.meshgrid(np.arange(len(mbins)), np.arange(len(sbins)))
            if np.abs(g["mse_ratio"] - 1).max() < 1e-9:      # the reference itself
                ax.text(0.5, 0.5, "reference\n(= 1 everywhere)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8.5, color=MUTED)
                ax.set_facecolor("#F6F7F7")
                continue
            cf = ax.contourf(X, Y, Z, levels=levels, cmap=DIVERGE, norm=norm, extend="both")
            ax.contour(X, Y, Z, levels=[1.0], colors=INK, linewidths=1.2,
                       linestyles=[(0, (4, 3))])
            ax.scatter(X[np.isfinite(Z)], Y[np.isfinite(Z)], s=8, color=INK, alpha=0.45, lw=0)
            # the best cell in this panel, as the one selective label, kept inside the axes
            jy, jx = np.unravel_index(np.nanargmax(Z), Z.shape)
            right, top = jx == len(mbins) - 1, jy == len(sbins) - 1
            ax.annotate(f"{Z[jy, jx]:.3f}", (X[jy, jx], Y[jy, jx]),
                        xytext=(-5 if right else 5, -5 if top else 5),
                        textcoords="offset points", fontsize=7.5, color=INK,
                        ha="right" if right else "left", va="top" if top else "bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    for ax in axes[-1]:
        ax.set_xticks(range(len(mbins)), xlab, rotation=35, ha="right", fontsize=7.5)
        ax.tick_params(axis="x", pad=4)
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(sbins)), ylab, fontsize=7)
        ax.tick_params(axis="y", pad=10)
    fig.supxlabel(r"true black-hole mass, $\log M_{\rm BH}$", fontsize=10, color=INK2, y=0.0)
    fig.supylabel("r-band time span (first to last observation); sources per row"
                  if span_of == "curve" else
                  "days covered by one r-band window of that panel's size",
                  fontsize=9.5, color=INK2, x=0.0)
    if cf is not None:
        cb = fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015,
                          ticks=np.linspace(1 - lim, 1 + lim, 5))
        cb.set_label(cbar_label, fontsize=8.5)
        cb.outline.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", y=1.04, fontsize=11.5)
    fig.text(0.02, 0.995, "teal = better, brown = worse; dashed line = 1.00; dots = computed "
             "cells, colour between them interpolated; label = best cell in the panel; "
             "blank = too few sources in that cell",
             ha="left", fontsize=8, color=MUTED)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path,
                    help="predictions.csv from an embeddings-only run (agn_ridge or agn_mlp)")
    ap.add_argument("--index", type=Path, help="lc_index.csv")
    ap.add_argument("--cells", type=Path,
                    help="redraw from this contour_span_mass.csv instead of --pred/--index")
    ap.add_argument("--targets", nargs="+", default=["m", "z", "e"], choices=["m", "z", "e"])
    ap.add_argument("--base", default="base_256_200_K_baseline",
                    help="substring naming the reference model (default: base, window 200)")
    ap.add_argument("--families", nargs="+", default=["base", "16"],
                    help="one row each: 'base' and/or MHLA latent sizes")
    ap.add_argument("--band", default="r")
    ap.add_argument("--span-of", default="curve", choices=["curve", "window"],
                    help="y axis: span of the whole light curve, or days covered by one "
                         "window of each panel's window size")
    ap.add_argument("--time-features", type=Path,
                    help="lc_time_features.csv (needed for --span-of window)")
    ap.add_argument("--span-bins", type=float, nargs="+",
                    help="span bin edges in days (default depends on --span-of)")
    ap.add_argument("--mass-bins", type=float, nargs="+",
                    default=[0, 8.0, 8.4, 8.8, 9.2, 9.6, np.inf])
    ap.add_argument("--min-n", type=int, default=100, help="smallest cell drawn (default 100)")
    ap.add_argument("--null-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", help="regression head named in the title (default: from --pred)")
    ap.add_argument("--out-dir", type=Path, default=Path("contour_span_mass"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.span_bins is None:
        a.span_bins = SPAN_BINS[a.span_of]

    if a.cells is not None:
        res = pd.read_csv(a.cells, dtype={"family": str})
        res = res[res["target"].isin(a.targets) & res["family"].isin(a.families)]
        head = a.head or a.cells.parent.name.split("_")[-1]
        a.span_of = res["span_of"].iloc[0] if "span_of" in res else "curve"
        res = res[res["n"] >= a.min_n]
    elif a.pred is None or a.index is None:
        print("need --pred and --index, or --cells", file=sys.stderr)
        return 1
    else:
        res = cells(a)
        if res is None:
            return 1
        head = a.head or a.pred.parent.name.replace("agn_", "")
        res.to_csv(a.out_dir / "contour_span_mass.csv", index=False)
        print(f"wrote {a.out_dir}/contour_span_mass.csv")
    head = {"mlp": "MLP"}.get(head, head)
    fams = [f for f in a.families if (res["family"] == f).any()]
    wins = sorted(res["window"].unique())

    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    ref = "standard attention, window 200"
    for t in a.targets:
        draw(res, t, "mse_ratio", fams, wins,
             f"{TLABEL[t]}, {head} head: gain over {ref}, by black-hole mass and time span",
             f"MSE({ref}) / MSE(model)\n> 1 = better than the reference",
             a.out_dir / f"contour_span_mass_{t}.png", a.span_of)
        if t == "m" and res.loc[res["target"] == "m", "excess"].notna().any():
            draw(res, t, "excess", fams, wins,
                 f"{TLABEL[t]}, {head} head: gain over {ref} BEYOND a uniform improvement",
                 "observed / expected MSE ratio\n1 = no mass-specific effect",
                 a.out_dir / "contour_span_mass_excess_m.png", a.span_of)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
