#!/usr/bin/env python3
"""Contour maps of the gain over the reference model across window size and time span.

For each attention type (standard attention = base, MHLA at each --families latent
size), every (window size, span bin) cell gets

    MSE(reference) / MSE(model)          > 1 = the model beats the reference

computed on the sources whose --band light curve spans that many days (first to last
observation). The reference is the base model at window 200 (--base), so its own cell
column is 1 by construction. The cells are drawn as filled contours: x = window size,
y = span bin. The span bins are evenly spaced on y (labelled with their day ranges
and source counts), because ~80% of sources fall in the top ~150 days and true day
spacing would squash those rows. Dots mark the cells actually computed; everything
between them is interpolated. --cells redraws from a saved contour_span_window.csv
without reading the predictions again.

Spans are not matched for point count here: short-span curves also have fewer points
(see span_within_length.py for the matched comparison). Nearly all curves span the
full ~6 yr ZTF baseline, so the default bins are fine at the top and coarse below
2000 days, where only ~2.5% of sources fall.

Reads the out-of-fold predictions.csv from an EMBEDDINGS-ONLY fit_regressor.py run and
lc_index.csv. Writes, to --out-dir:

  contour_span_window_<t>.png   one panel per attention type, one file per target
  contour_span_window.csv       every cell: family, window, span bin, n, median span,
                                MSE, reference MSE, ratio

    python plot_span_window_contour.py --pred cluster_results/agn_mlp/predictions.csv \\
        --index $IDX --out-dir cluster_results/contour_mlp
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, INK2, MUTED, RULE = "#101A19", "#384645", "#5F6B6A", "#DBE1E0"
DIVERGE = LinearSegmentedColormap.from_list(
    "gain", ["#7A4E0B", "#C9A36A", "#EEEFEE", "#7CC3B9", "#06574E"])
TLABEL = {"m": r"black-hole mass ($\log M_{\rm BH}$)", "z": "redshift ($z$)",
          "e": r"Eddington ratio ($\log L/L_{\rm Edd}$)"}
PAT_LATENT = re.compile(r"_K_01_(\d+)_(\d+)$")
PAT_BASE = re.compile(r"_base_\d+_(\d+)_K_baseline$")
CONTROL = "CONTROL: summary stats only"


def parse(name):
    m = PAT_LATENT.search(name)
    if m:
        return m.group(1), int(m.group(2))
    m = PAT_BASE.search(name)
    if m:
        return "base", int(m.group(1))
    return None, None


def fam_title(lat):
    return "standard attention" if lat == "base" else f"MHLA, latent {lat}"


def cells(a):
    """MSE ratio over the reference for every (target, family, window, span bin)."""
    # ---- span per source ---------------------------------------------------------
    idx = pd.read_csv(a.index, dtype={"source_id": str},
                      usecols=["source_id", "band_name", "mjd_min", "mjd_max"])
    b = idx[idx["band_name"] == a.band].drop_duplicates("source_id").set_index("source_id")
    if b.empty:
        print(f"no band {a.band!r} in {a.index}", file=sys.stderr)
        return None
    span = b["mjd_max"] - b["mjd_min"]

    # ---- predictions ------------------------------------------------------------
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

    rows = []
    for t in a.targets:
        wide = pred.pivot_table(index="source_id", columns="model", values=f"{t}_pred")
        y = pred.drop_duplicates("source_id").set_index("source_id")[f"{t}_true"] \
            .reindex(wide.index)
        s = span.reindex(wide.index)
        ok = y.notna() & s.notna() & wide[base].notna()
        wide, y, s = wide[ok], y[ok].to_numpy(float), s[ok].to_numpy(float)
        sb = pd.cut(s, a.span_bins, right=False, labels=False)
        eb = (wide[base].to_numpy(float) - y) ** 2
        for (lat, ws), mdl in models.items():
            em = (wide[mdl].to_numpy(float) - y) ** 2
            for k in range(len(a.span_bins) - 1):
                sel = (sb == k) & np.isfinite(em)
                if sel.sum() < 20:
                    continue
                rows.append({"target": t, "family": lat, "window": ws, "span_bin": k,
                             "span_lo": a.span_bins[k], "span_hi": a.span_bins[k + 1],
                             "n": int(sel.sum()), "span_median": float(np.median(s[sel])),
                             "mse": em[sel].mean(), "mse_base": eb[sel].mean(),
                             "mse_ratio": eb[sel].mean() / em[sel].mean()})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path,
                    help="predictions.csv from an embeddings-only run (agn_ridge or agn_mlp)")
    ap.add_argument("--index", type=Path, help="lc_index.csv")
    ap.add_argument("--cells", type=Path,
                    help="redraw from this contour_span_window.csv instead of --pred/--index")
    ap.add_argument("--targets", nargs="+", default=["z", "m", "e"], choices=["m", "z", "e"])
    ap.add_argument("--base", default="base_256_200_K_baseline",
                    help="substring naming the reference model (default: base, window 200)")
    ap.add_argument("--families", nargs="+", default=["base", "4", "16", "128"],
                    help="attention types to draw: 'base' and/or MHLA latent sizes")
    ap.add_argument("--band", default="r")
    ap.add_argument("--span-bins", type=float, nargs="+",
                    default=[0, 2000, 2150, 2200, 2250, 2275, np.inf])
    ap.add_argument("--head", help="regression head named in the title (default: from --pred)")
    ap.add_argument("--out-dir", type=Path, default=Path("contour_span_window"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.cells is not None:
        res = pd.read_csv(a.cells, dtype={"family": str})
        res = res[res["target"].isin(a.targets) & res["family"].isin(a.families)]
        head = a.head or a.cells.parent.name.split("_")[-1]
    elif a.pred is None or a.index is None:
        print("need --pred and --index, or --cells", file=sys.stderr)
        return 1
    else:
        res = cells(a)
        if res is None:
            return 1
        head = a.head or a.pred.parent.name.replace("agn_", "")
    res.to_csv(a.out_dir / "contour_span_window.csv", index=False)
    head = {"mlp": "MLP"}.get(head, head)
    fams = [f for f in a.families if (res["family"] == f).any()]
    wins = sorted(res["window"].unique())

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    for t in a.targets:
        rt = res[res["target"] == t]
        # one colour scale per target, symmetric around 1, shared by every panel
        lim = float(np.nanmax(np.abs(rt["mse_ratio"] - 1)))
        lim = np.ceil(lim * 100) / 100
        levels = np.linspace(1 - lim, 1 + lim, 17)
        norm = TwoSlopeNorm(1.0, 1 - lim, 1 + lim)
        fig, axes = plt.subplots(1, len(fams), figsize=(3.6 * len(fams) + 1.2, 3.9),
                                 sharey=True, squeeze=False)
        # span bins sit at even steps: nearly all sources are in the top ~150 days, so
        # true day spacing would squash the rows that hold most of the data
        bins = sorted(rt["span_bin"].unique())
        ylab = []
        for k in bins:
            c = rt[rt["span_bin"] == k].iloc[0]
            lo, hi = c["span_lo"], c["span_hi"]
            rng_ = (f"< {hi:.0f}" if lo == 0 else f"{lo:.0f}+" if not np.isfinite(hi)
                    else f"{lo:.0f}–{hi:.0f}")
            n = rt[rt["span_bin"] == k]["n"].max()
            ylab.append(f"{rng_} d\n{n:,} sources")
        for ax, fam in zip(axes[0], fams):
            g = rt[rt["family"] == fam].pivot_table(index="span_bin", columns="window",
                                                     values="mse_ratio") \
                .reindex(index=bins, columns=wins)
            Z = g.to_numpy(float)
            X, Y = np.meshgrid(wins, np.arange(len(bins)))
            cf = ax.contourf(X, Y, Z, levels=levels, cmap=DIVERGE, norm=norm, extend="both")
            ax.contour(X, Y, Z, levels=[1.0], colors=INK, linewidths=1.3,
                       linestyles=[(0, (4, 3))])
            ax.scatter(X[np.isfinite(Z)], Y[np.isfinite(Z)], s=10, color=INK, alpha=0.5, lw=0)
            # the best cell in this panel, as the one selective label, kept inside the axes
            j = np.nanargmax(Z)
            jy, jx = np.unravel_index(j, Z.shape)
            ax.annotate(f"{Z[jy, jx]:.3f}", (X[jy, jx], Y[jy, jx]),
                        xytext=(-6 if jx == len(wins) - 1 else 6, -6 if jy == len(bins) - 1 else 6),
                        textcoords="offset points", fontsize=8.5, color=INK,
                        ha="right" if jx == len(wins) - 1 else "left",
                        va="top" if jy == len(bins) - 1 else "bottom",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
            ax.set_xticks(wins)
            ax.set_yticks(range(len(bins)), ylab)
            ax.set_xlabel("window size (points)")
            ax.set_title(fam_title(fam), loc="left", fontsize=10.5, pad=8)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            ax.tick_params(axis="y", labelsize=8, pad=14)
            ax.tick_params(axis="x", pad=7)
        axes[0][0].set_ylabel(f"{a.band}-band time span, first to last observation")
        cb = fig.colorbar(cf, ax=axes[0].tolist(), fraction=0.025, pad=0.015,
                          ticks=np.linspace(1 - lim, 1 + lim, 5))
        cb.set_label("MSE(standard, window 200) / MSE(model)\n> 1 = better than standard, window 200",
                     fontsize=8.5)
        cb.outline.set_visible(False)
        fig.suptitle(f"{TLABEL[t]}, {head} head: gain over standard attention (window 200) "
                     f"by window size and time span", x=0.02, ha="left", y=1.06, fontsize=11.5)
        fig.text(0.02, 0.985, "teal = better, brown = worse; dashed line = no gain (1.00); "
                 "dots = computed cells, colour between them interpolated; "
                 "label = best cell in the panel",
                 ha="left", fontsize=8, color=MUTED)
        p = a.out_dir / f"contour_span_window_{t}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {p}")
    print(f"wrote {a.out_dir}/contour_span_window.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
