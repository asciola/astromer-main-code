#!/usr/bin/env python3
"""Maps of MHLA's gain over standard attention across black-hole mass and the
REST-FRAME duration of the encoder's window, on the paired window sets from
restframe_windows.py (the same sources at every duration).

Takes one fit_regressor.py predictions.csv per duration (as plot_rest_duration.py
does) and, for every model and every (duration, mass bin) cell, computes

    MSE(reference) / MSE(model)          > 1 = the model beats the reference

on the sources in that cell. --reference picks the reference:

  same   (default) standard attention at the SAME duration. One panel per MHLA
         latent; answers "where in mass, and at which duration, does MHLA beat
         standard attention?"
  first  standard attention at the SHORTEST duration. Adds a standard-attention
         panel (1 in its first row by construction); answers "how does each model
         change with duration, by mass?"

x = true black-hole mass (LOGMBH) bins, y = rest-frame window duration. Each cell is
drawn as a block with its value printed in it; bold = the 95% paired bootstrap
interval (sources resampled within the cell, the same draws for every model and
duration) excludes 1. Only three durations exist, so the map is cells, not an
interpolated contour.

Every source has one window per duration and its prediction is out-of-fold, so
cells in one column (one mass bin) hold the same sources at every duration.

NULL (target m only)
  Splitting by TRUE mass rewards any model that shrinks less toward the mean than
  the reference (see length_mass_gain.py), so for target m the script also draws the
  EXCESS: each cell divided by what a uniform improvement would give there
  (null_tercile_ratios, from the model's and the reference's overall R^2 at that
  duration). 1 = no mass-specific effect. Its interval treats the null as fixed.
  The Eddington ratio is split by mass, not by itself, so it gets no null, but it
  is not free of the artifact: log L/L_Edd = log L_bol - log M_BH - 38.1, so mass
  bins are partly Eddington-ratio bins.

Writes, to --out-dir:
  rest_duration_mass_<head>_<ref>_<t>.png          one figure per target
  rest_duration_mass_<head>_<ref>_excess_m.png     target m over the null
  rest_duration_mass_<head>_<ref>.csv              every cell: model, duration, mass
                                                   bin, n, MSEs, ratio and interval,
                                                   null, excess and interval

    python plot_rest_duration_mass.py --runs 200:cluster_results/agn_ridge_rw200/predictions.csv \\
        300:cluster_results/agn_ridge_rw300/predictions.csv \\
        400:cluster_results/agn_ridge_rw400/predictions.csv \\
        --out-dir cluster_results/rest_duration
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
from matplotlib.colors import TwoSlopeNorm, to_rgb

from length_mass_gain import null_tercile_ratios
from plot_span_mass_contour import bin_label
from plot_span_window_contour import CONTROL, DIVERGE, INK, INK2, MUTED, RULE, fam_title, parse

TLABEL = {"m": r"$\log M_{\rm BH}$", "e": r"$\log L/L_{\rm Edd}$"}


def load(runs):
    """{days: predictions} restricted to the sources every run has."""
    preds = {d: pd.read_csv(p, dtype={"source_id": str}) for d, p in runs}
    sids = set.intersection(*(set(p["source_id"]) for p in preds.values()))
    if any(set(p["source_id"]) != sids for p in preds.values()):
        print(f"note: the runs do not cover identical sources; using the intersection "
              f"({len(sids):,})")
    return {d: p[p["source_id"].isin(sids)] for d, p in preds.items()}


def cells(a, preds):
    days = sorted(preds)
    first = preds[days[0]]
    models = [m for m in first["model"].unique() if m != CONTROL and parse(m)[0] is not None]
    base = [m for m in models if parse(m)[0] == "base"]
    if len(base) != 1:
        print(f"need exactly one standard-attention model, found {base}", file=sys.stderr)
        return None
    base = base[0]
    # mass of every source (the same in every run); sources without one are dropped
    mass = first.drop_duplicates("source_id").set_index("source_id")["m_true"].dropna()
    sids = mass.index.to_numpy()
    nm = len(a.mass_bins) - 1
    mb = pd.cut(mass.to_numpy(float), a.mass_bins, right=False, labels=False)
    rng = np.random.default_rng(a.seed)
    # one set of bootstrap weights per mass bin, shared by every model, duration and
    # target, so the intervals are paired
    W = {j: np.stack([np.bincount(rng.integers(0, (mb == j).sum(), (mb == j).sum()),
                                  minlength=(mb == j).sum())
                      for _ in range(a.bootstrap)]).astype(np.float64)
         for j in range(nm) if (mb == j).sum() >= a.min_n}

    rows = []
    for t in a.targets:
        sq = {}                                    # (days, model) -> squared errors
        for d in days:
            p = preds[d]
            wide = p.pivot_table(index="source_id", columns="model", values=f"{t}_pred") \
                .reindex(sids)
            y = p.drop_duplicates("source_id").set_index("source_id")[f"{t}_true"] \
                .reindex(sids).to_numpy(float)
            for m in models:
                sq[(d, m)] = (wide[m].to_numpy(float) - y) ** 2
        ok = np.isfinite(y) & np.all([np.isfinite(v) for v in sq.values()], axis=0)
        var = y[ok].var()
        for d in days:
            dref = d if a.reference == "same" else days[0]
            eb = sq[(dref, base)]
            for m in models:
                if a.reference == "same" and m == base:
                    continue
                em = sq[(d, m)]
                null = np.full(nm, np.nan)
                if t == "m":
                    r2b, r2m = 1 - eb[ok].mean() / var, 1 - em[ok].mean() / var
                    null = null_tercile_ratios(y[ok], mb[ok], nm, r2b, r2m, rng, a.null_draws)
                for j, Wj in W.items():
                    sel = (mb == j)
                    if not ok[sel].all():          # keep the bootstrap weights aligned
                        sel = sel & ok
                        Wj = Wj[:, ok[mb == j]]
                    e_b, e_m = eb[sel], em[sel]
                    ratio = e_b.mean() / e_m.mean()
                    boot = (Wj @ e_b) / (Wj @ e_m)
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    nj = null[j]
                    rows.append({
                        "target": t, "reference": a.reference, "days": d, "model": m,
                        "family": parse(m)[0], "mass_bin": j, "mass_lo": a.mass_bins[j],
                        "mass_hi": a.mass_bins[j + 1], "n": int(sel.sum()),
                        "mse": e_m.mean(), "mse_ref": e_b.mean(), "mse_ratio": ratio,
                        "ratio_lo": lo, "ratio_hi": hi, "null_mse_ratio": nj,
                        "excess": ratio / nj if np.isfinite(nj) else np.nan,
                        "excess_lo": lo / nj if np.isfinite(nj) else np.nan,
                        "excess_hi": hi / nj if np.isfinite(nj) else np.nan})
        print(f"{t}: {ok.sum():,} sources with a valid mass and {t}", flush=True)
    return pd.DataFrame(rows)


def text_color(rgba):
    r, g, b = to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.45 else INK


def draw(res, t, value, lo_col, hi_col, fams, title, cbar_label, note, path):
    rt = res[res["target"] == t]
    mbins = sorted(rt["mass_bin"].unique())
    days = sorted(rt["days"].unique())
    xlab = []
    for j in mbins:
        c = rt[rt["mass_bin"] == j].iloc[0]
        xlab.append(bin_label(c["mass_lo"], c["mass_hi"], "", ".1f") + f"\n{int(c['n']):,}")
    # one colour scale per figure, symmetric around 1, set by the 98th percentile of the
    # non-reference cells
    v = rt[np.abs(rt["mse_ratio"] - 1) > 1e-12][value].dropna()
    lim = float(np.quantile(np.abs(v - 1), 0.98)) if len(v) else 0.02
    lim = max(0.02, np.ceil(lim / 0.02) * 0.02)
    norm = TwoSlopeNorm(1.0, 1 - lim, 1 + lim)

    ncols = 3 if len(fams) == 9 else min(5, len(fams))
    nrows = int(np.ceil(len(fams) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols + 1.2, 2.55 * nrows + 0.6),
                             squeeze=False, sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    mesh = None
    for k, ax in enumerate(axes.flat):
        if k >= len(fams):
            ax.set_visible(False)
            continue
        fam = fams[k]
        g = rt[rt["family"] == fam]
        Z = g.pivot_table(index="days", columns="mass_bin", values=value) \
            .reindex(index=days, columns=mbins).to_numpy(float)
        R = g.pivot_table(index="days", columns="mass_bin", values="mse_ratio") \
            .reindex(index=days, columns=mbins).to_numpy(float)
        LO = g.pivot_table(index="days", columns="mass_bin", values=lo_col) \
            .reindex(index=days, columns=mbins).to_numpy(float)
        HI = g.pivot_table(index="days", columns="mass_bin", values=hi_col) \
            .reindex(index=days, columns=mbins).to_numpy(float)
        is_ref = np.abs(R - 1) < 1e-12                 # the reference row itself
        Z[is_ref] = 1.0
        mesh = ax.pcolormesh(np.arange(len(mbins) + 1) - 0.5, np.arange(len(days) + 1) - 0.5,
                             Z, cmap=DIVERGE, norm=norm, edgecolors="white", linewidth=2)
        for iy in range(len(days)):
            for ix in range(len(mbins)):
                z = Z[iy, ix]
                if not np.isfinite(z):
                    continue
                if is_ref[iy, ix]:
                    s, wt = "ref", "normal"
                else:
                    sig = LO[iy, ix] > 1 or HI[iy, ix] < 1
                    s, wt = f"{z:.3f}", "bold" if sig else "normal"
                ax.text(ix, iy, s, ha="center", va="center", fontsize=7.5, fontweight=wt,
                        color=text_color(mesh.cmap(mesh.norm(z))))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(range(len(mbins)), xlab, fontsize=7.5)
        ax.set_yticks(range(len(days)), [f"{d:.0f} d" for d in days])
        ax.set_title(fam_title(fam), loc="left", fontsize=10, pad=6)
    for ax in axes[-1]:
        ax.set_xlabel(r"true $\log M_{\rm BH}$ (sources per bin)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("rest-frame window\nduration", fontsize=8.5)
    cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015,
                      ticks=np.linspace(1 - lim, 1 + lim, 5))
    cb.set_label(cbar_label, fontsize=8.5)
    cb.outline.set_visible(False)
    # title and note at fixed distances (in inches) above the top row's panel titles
    top = max(ax.get_position().y1 for ax in axes[0] if ax.get_visible())
    inch = 1 / fig.get_figheight()
    fig.text(0.02, top + 0.62 * inch, title, ha="left", va="bottom", fontsize=11.5)
    fig.text(0.02, top + 0.40 * inch, note, ha="left", va="bottom", fontsize=8, color=MUTED)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+",
                    help="DAYS:predictions.csv, one per window duration")
    ap.add_argument("--cells", type=Path,
                    help="redraw from this rest_duration_mass_*.csv instead of --runs")
    ap.add_argument("--targets", nargs="+", default=["m", "e"], choices=["m", "e"])
    ap.add_argument("--reference", choices=["same", "first"], default="same",
                    help="standard attention at the same duration (default) or at the "
                         "shortest one")
    ap.add_argument("--families", nargs="+",
                    help="MHLA latents (and 'base') to draw, one panel each "
                         "(default: all present)")
    ap.add_argument("--mass-bins", type=float, nargs="+",
                    default=[0, 8.0, 8.4, 8.8, 9.2, 9.6, np.inf])
    ap.add_argument("--min-n", type=int, default=200, help="smallest mass bin drawn")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--null-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", help="regression head named in titles and files "
                                   "(default: from the first --runs path)")
    ap.add_argument("--out-dir", type=Path, default=Path("rest_duration"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.cells is not None:
        res = pd.read_csv(a.cells, dtype={"family": str})
        res = res[res["target"].isin(a.targets)]
        a.reference = res["reference"].iloc[0]
        head = a.head or a.cells.stem.split("_")[3]
    elif a.runs:
        runs = []
        for r in a.runs:
            d, _, p = r.partition(":")
            runs.append((float(d), Path(p)))
        head = a.head or runs[0][1].parent.name.split("_")[1]
        res = cells(a, load(runs))
        if res is None:
            return 1
    else:
        print("need --runs or --cells", file=sys.stderr)
        return 1
    stem = f"rest_duration_mass_{head}_{a.reference}"
    if a.cells is None:
        res.to_csv(a.out_dir / f"{stem}.csv", index=False)
        print(f"wrote {a.out_dir / f'{stem}.csv'}")

    present = list(dict.fromkeys(res["family"]))
    order = sorted(present, key=lambda f: -1 if f == "base" else int(f))
    fams = [f for f in order if a.families is None or f in a.families]
    head_t = {"mlp": "MLP"}.get(head, head)
    ref = ("standard attention at the same duration" if a.reference == "same" else
           f"standard attention at {res['days'].min():.0f} d")

    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    note = ("teal = better, brown = worse than the reference; bold = 95% paired bootstrap "
            "interval excludes 1; every column holds the same sources at every duration")
    for t in a.targets:
        draw(res, t, "mse_ratio", "ratio_lo", "ratio_hi", fams,
             f"{TLABEL[t]}, {head_t} head: gain over {ref}, by mass and rest-frame duration",
             f"MSE(reference) / MSE(model)\n> 1 = better than the reference", note,
             a.out_dir / f"{stem}_{t}.png")
        if t == "m" and res.loc[res["target"] == "m", "excess"].notna().any():
            draw(res, t, "excess", "excess_lo", "excess_hi", fams,
                 f"{TLABEL[t]}, {head_t} head: gain over {ref} BEYOND a uniform improvement",
                 "observed / expected MSE ratio\n1 = no mass-specific effect", note,
                 a.out_dir / f"{stem}_excess_m.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
