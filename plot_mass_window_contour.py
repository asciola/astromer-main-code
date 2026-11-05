#!/usr/bin/env python3
"""Contour maps of the gain over the reference model across black-hole mass and window
size, for long light curves only.

Keeps the sources whose --band light curve has at least --min-points observations
(default 800 r-band points, ~5% of sources), then for each attention type (--families,
one panel each: standard attention = base, MHLA at each latent size) gives every
(mass bin, window size) cell

    MSE(reference) / MSE(model)          > 1 = the model beats the reference

on the sources in that cell. The reference is the base model at window 200 (--base) on
the same sources, so the standard-attention row at window 200 is 1 by construction.
x = true black-hole mass (LOGMBH) bins, y = window size. Dots mark the cells actually
computed; colour between them is interpolated. Cells with fewer than --min-n sources
are left blank.

The predictions average the g, r and i embeddings of a source (fit_regressor.py
--agg mean), so the cut is on one band's length but the model saw all bands.

NULL (target m only)
  Splitting by TRUE mass rewards any model that shrinks less toward the mean than the
  reference (see length_mass_gain.py), so for target m the script also writes
  contour_mass_window_excess_m.png: each cell divided by what a uniform improvement
  would give there (null_tercile_ratios, run per model with its R^2 on the kept
  sources). 1 = no mass-specific effect.

Reads the out-of-fold predictions.csv from an EMBEDDINGS-ONLY fit_regressor.py run and
lc_index.csv. Writes, to --out-dir:

  contour_mass_window_<t>.png         one figure per target
  contour_mass_window_excess_m.png    target m divided by the uniform-improvement null
  contour_mass_window.csv             every cell: family, window, mass bin, n, MSE,
                                      reference MSE, ratio, null, excess

    python plot_mass_window_contour.py --pred cluster_results/agn_mlp/predictions.csv \\
        --index $IDX --out-dir cluster_results/contour_mw800_mlp
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
from plot_span_mass_contour import bin_label
from plot_span_window_contour import CONTROL, DIVERGE, INK, INK2, MUTED, RULE, TLABEL, \
    fam_title, parse


def cells(a):
    """MSE ratio over the reference for every (target, family, window, mass) cell."""
    rng = np.random.default_rng(a.seed)
    idx = pd.read_csv(a.index, dtype={"source_id": str},
                      usecols=["source_id", "band_name", "n_points", "LOGMBH", "mbh_valid"])
    b = idx[idx["band_name"] == a.band].drop_duplicates("source_id").set_index("source_id")
    if b.empty:
        print(f"no band {a.band!r} in {a.index}", file=sys.stderr)
        return None
    # LOGMBH == 0 is the catalogue's "not measured" sentinel; mbh_valid excludes it
    keep = b[(b["n_points"] >= a.min_points) & b["mbh_valid"].astype(bool)]
    mass = keep["LOGMBH"]

    pred = pd.read_csv(a.pred, dtype={"source_id": str})
    pred = pred[(pred["model"] != CONTROL) & pred["source_id"].isin(mass.index)]
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
        ok = y.notna() & wide[base].notna()
        wide, y = wide[ok], y[ok].to_numpy(float)
        mb = pd.cut(mass.reindex(wide.index).to_numpy(float), a.mass_bins, right=False,
                    labels=False)
        eb = (wide[base].to_numpy(float) - y) ** 2
        for (lat, ws), mdl in models.items():
            em = (wide[mdl].to_numpy(float) - y) ** 2
            good = np.isfinite(em)
            null = np.full(nm, np.nan)
            if t == "m":
                v = y[good].var()
                r2b, r2m = 1 - eb[good].mean() / v, 1 - em[good].mean() / v
                null = null_tercile_ratios(y[good], mb[good], nm, r2b, r2m, rng, a.null_draws)
            for j in range(nm):
                sel = good & (mb == j)
                if sel.sum() < a.min_n:
                    continue
                ratio = eb[sel].mean() / em[sel].mean()
                rows.append({"target": t, "band": a.band, "min_points": a.min_points,
                             "family": lat, "window": ws, "mass_bin": j,
                             "mass_lo": a.mass_bins[j], "mass_hi": a.mass_bins[j + 1],
                             "n": int(sel.sum()), "mse": em[sel].mean(),
                             "mse_base": eb[sel].mean(), "mse_ratio": ratio,
                             "null_mse_ratio": null[j],
                             "excess": ratio / null[j] if np.isfinite(null[j]) else np.nan})
        print(f"{t}: {len(y):,} sources with >= {a.min_points} {a.band}-band points "
              "and a valid mass", flush=True)
    return pd.DataFrame(rows)


def draw(res, t, value, fams, title, cbar_label, note, path):
    rt = res[res["target"] == t]
    mbins = sorted(rt["mass_bin"].unique())
    wins = sorted(rt["window"].unique())
    xlab = []
    for j in mbins:
        c = rt[rt["mass_bin"] == j].iloc[0]
        n = rt[rt["mass_bin"] == j]["n"].max()
        xlab.append(bin_label(c["mass_lo"], c["mass_hi"], "", ".1f") + f"\n{n:,}")
    # one colour scale per figure, symmetric around 1, shared by every panel; set by the
    # 98th percentile of the non-reference cells (extend="both" shows anything beyond)
    v = rt[np.abs(rt["mse_ratio"] - 1) > 1e-9][value].dropna()
    lim = float(np.quantile(np.abs(v - 1), 0.98)) if len(v) else 0.02
    lim = max(0.02, np.ceil(lim / 0.02) * 0.02)
    levels = np.linspace(1 - lim, 1 + lim, 17)
    norm = TwoSlopeNorm(1.0, 1 - lim, 1 + lim)

    fig, axes = plt.subplots(1, len(fams), figsize=(3.9 * len(fams) + 1.4, 3.9),
                             sharey=True, squeeze=False)
    cf = None
    for ax, fam in zip(axes[0], fams):
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        g = rt[rt["family"] == fam]
        Z = g.pivot_table(index="window", columns="mass_bin", values=value) \
            .reindex(index=wins, columns=mbins).to_numpy(float)
        R = g.pivot_table(index="window", columns="mass_bin", values="mse_ratio") \
            .reindex(index=wins, columns=mbins).to_numpy(float)
        Z[np.abs(R - 1) < 1e-9] = 1.0           # the reference row is exactly 1
        X, Y = np.meshgrid(np.arange(len(mbins)), np.arange(len(wins)))
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=DIVERGE, norm=norm, extend="both")
        ax.contour(X, Y, Z, levels=[1.0], colors=INK, linewidths=1.2,
                   linestyles=[(0, (4, 3))])
        ax.scatter(X[np.isfinite(Z)], Y[np.isfinite(Z)], s=10, color=INK, alpha=0.5, lw=0)
        if fam == "base" and 200 in wins:
            ax.annotate("reference", (len(mbins) / 2 - 0.5, wins.index(200)), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8,
                        color=MUTED)
        # the best cell in this panel, as the one selective label, kept inside the axes
        jy, jx = np.unravel_index(np.nanargmax(Z), Z.shape)
        right, top = jx == len(mbins) - 1, jy == len(wins) - 1
        ax.annotate(f"{Z[jy, jx]:.3f}", (X[jy, jx], Y[jy, jx]),
                    xytext=(-6 if right else 6, -6 if top else 6),
                    textcoords="offset points", fontsize=8, color=INK,
                    ha="right" if right else "left", va="top" if top else "bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
        ax.set_xticks(range(len(mbins)), xlab, fontsize=7.5)
        ax.tick_params(axis="x", pad=6)
        ax.set_yticks(range(len(wins)), [str(w) for w in wins])
        ax.set_title(fam_title(fam), loc="left", fontsize=10.5, pad=8)
        ax.set_xlabel(r"true black-hole mass, $\log M_{\rm BH}$ (sources per bin below)")
    axes[0][0].set_ylabel("window size (points)")
    cb = fig.colorbar(cf, ax=axes[0].tolist(), fraction=0.03, pad=0.02,
                      ticks=np.linspace(1 - lim, 1 + lim, 5))
    cb.set_label(cbar_label, fontsize=8.5)
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", y=1.07, fontsize=11.5)
    fig.text(0.02, 0.995, note, ha="left", fontsize=8, color=MUTED)
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
                    help="redraw from this contour_mass_window.csv instead of --pred/--index")
    ap.add_argument("--targets", nargs="+", default=["m", "z", "e"], choices=["m", "z", "e"])
    ap.add_argument("--base", default="base_256_200_K_baseline",
                    help="substring naming the reference model (default: base, window 200)")
    ap.add_argument("--families", nargs="+", default=["base", "16"],
                    help="one panel each: 'base' and/or MHLA latent sizes")
    ap.add_argument("--band", default="r", help="band whose length is cut on")
    ap.add_argument("--min-points", type=int, default=800,
                    help="keep sources with at least this many --band points (default 800)")
    ap.add_argument("--mass-bins", type=float, nargs="+",
                    default=[0, 8.0, 8.4, 8.8, 9.2, 9.6, np.inf])
    ap.add_argument("--min-n", type=int, default=100, help="smallest cell drawn (default 100)")
    ap.add_argument("--null-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", help="regression head named in the title (default: from --pred)")
    ap.add_argument("--out-dir", type=Path, default=Path("contour_mass_window"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.cells is not None:
        res = pd.read_csv(a.cells, dtype={"family": str})
        res = res[res["target"].isin(a.targets) & res["family"].isin(a.families) &
                  (res["n"] >= a.min_n)]
        head = a.head or a.cells.parent.name.split("_")[-1]
        a.band, a.min_points = res["band"].iloc[0], int(res["min_points"].iloc[0])
    elif a.pred is None or a.index is None:
        print("need --pred and --index, or --cells", file=sys.stderr)
        return 1
    else:
        res = cells(a)
        if res is None:
            return 1
        head = a.head or a.pred.parent.name.replace("agn_", "")
        res.to_csv(a.out_dir / "contour_mass_window.csv", index=False)
        print(f"wrote {a.out_dir}/contour_mass_window.csv")
    head = {"mlp": "MLP"}.get(head, head)
    fams = [f for f in a.families if (res["family"] == f).any()]

    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    ref = "standard attention, window 200"
    cut = f"light curves with ≥ {a.min_points} {a.band}-band points"
    note = ("teal = better, brown = worse; dashed line = 1.00; dots = computed cells, "
            "colour between them interpolated; label = best cell in the panel")
    for t in a.targets:
        draw(res, t, "mse_ratio", fams,
             f"{TLABEL[t]}, {head} head, {cut}: gain over {ref}",
             f"MSE({ref}) / MSE(model)\n> 1 = better than the reference", note,
             a.out_dir / f"contour_mass_window_{t}.png")
        if t == "m" and res.loc[res["target"] == "m", "excess"].notna().any():
            draw(res, t, "excess", fams,
                 f"{TLABEL[t]}, {head} head, {cut}: gain BEYOND a uniform improvement",
                 "observed / expected MSE ratio\n1 = no mass-specific effect", note,
                 a.out_dir / "contour_mass_window_excess_m.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
