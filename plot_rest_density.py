#!/usr/bin/env python3
"""Gain over standard attention at window 200, for every encoder, in each cell of the
rest-frame window size / density design (restframe_windows.py --cells,
cluster_results/run_rest_density.sh):

  p200_d300      200 points over 300 rest-frame days
  p400_d600      400 points over 600 days     same density, twice the time
  p400_d600_t2   those 400 thinned to 200     same time, half the density

The same sources are in every cell. Takes one fit_regressor.py predictions.csv per
cell. In every cell the REFERENCE is standard attention pretrained at window 200 on
that cell's data (on p400_d600 it embeds the 400 points as two 200-point windows,
averaged), and every encoder in the run is scored against it:

    R^2(encoder) - R^2(standard attention, window 200)      > 0 = beats it

Window-200 encoders (blue) appear in every cell; window-400 encoders (orange),
including standard attention at 400, wherever they were run (the 400-point cell).
One figure, rows = targets (default m, e), columns = cells; x = attention type and
latent size. Error bars are 95% paired bootstrap intervals over sources: one set of
draws is shared by every model and cell, so comparisons between cells are paired
too.

The console prints, per target, R^2 of every encoder in every cell, and the paired
differences between cells for the encoders that ran in both:
  time       p400_d600    - p200_d300      twice the time, same density
  density    p400_d600    - p400_d600_t2   twice the density, same time
  thinned    p400_d600_t2 - p200_d300      twice the time, half the density
(a window-200 encoder on p400_d600 sees two windows; there, "time" and "density"
also change how many windows it averages). Writes rest_density_<head>.png,
rest_density_<head>.csv (every model x cell x target) and
rest_density_<head>_contrasts.csv.

    python plot_rest_density.py --runs p200_d300:agn_ridge_rc_p200_d300/predictions.csv \\
        p400_d600:agn_ridge_rc_p400_d600/predictions.csv \\
        p400_d600_t2:agn_ridge_rc_p400_d600_t2/predictions.csv \\
        --head ridge --out-dir rest_density
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

from plot_span_window_contour import CONTROL, INK, INK2, MUTED, RULE, parse

TLABEL = {"m": r"$\log M_{\rm BH}$", "e": r"$\log L/L_{\rm Edd}$", "z": "redshift z"}
WIN_COLOR = {200: "#2a78d6", 400: "#eb6834"}          # categorical slots 1 and 2
CELL_RE = re.compile(r"p(\d+)_d(\d+)(?:_t(\d+))?$")
CONTRASTS = [("time", "p400_d600", "p200_d300", "2x the time, same density"),
             ("density", "p400_d600", "p400_d600_t2", "2x the density, same time"),
             ("thinned", "p400_d600_t2", "p200_d300", "2x the time, half the density")]


def cell_label(name):
    m = CELL_RE.match(name)
    if not m:
        return name
    p, d, t = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
    n = p // t
    how = f"{p} thinned to {n}" if t > 1 else f"{n}"
    return f"{how} points over {d} d\n{n / d:.2f} observations / day"


def fam_key(lat):
    return -1 if lat == "base" else int(lat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="CELL:predictions.csv, one per cell (CELL as p<P>_d<D>[_t<T>])")
    ap.add_argument("--targets", nargs="+", default=["m", "e"])
    ap.add_argument("--ref-window", type=int, default=200,
                    help="window size of the standard-attention reference (default 200)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", help="regression head for titles and files (default: from paths)")
    ap.add_argument("--out-dir", type=Path, default=Path("rest_density"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for r in a.runs:
        c, _, p = r.partition(":")
        runs.append((c, Path(p)))
    cells = [c for c, _ in runs]
    head = a.head or runs[0][1].parent.name.split("_")[1]

    preds = {c: pd.read_csv(p, dtype={"source_id": str}) for c, p in runs}
    sids = sorted(set.intersection(*(set(p["source_id"]) for p in preds.values())))
    if any(set(p["source_id"]) != set(sids) for p in preds.values()):
        print(f"note: the cells do not cover identical sources; using the intersection "
              f"({len(sids):,})")
    models = {c: {m: parse(m) for m in p["model"].unique()
                  if m != CONTROL and parse(m)[0] is not None} for c, p in preds.items()}
    ref = {}
    for c in cells:
        hits = [m for m, v in models[c].items() if v == ("base", a.ref_window)]
        if len(hits) != 1:
            print(f"{c}: need exactly one standard-attention window-{a.ref_window} model, "
                  f"found {hits}", file=sys.stderr)
            return 1
        ref[c] = hits[0]

    rng = np.random.default_rng(a.seed)
    n = len(sids)
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n)
                  for _ in range(a.bootstrap)]).astype(np.float64)

    rows, boot = [], {}
    for t in a.targets:
        for c in cells:
            p = preds[c]
            y = p.drop_duplicates("source_id").set_index("source_id")[f"{t}_true"] \
                .reindex(sids).to_numpy(float)
            wide = p.pivot_table(index="source_id", columns="model", values=f"{t}_pred") \
                .reindex(sids)
            ok = np.isfinite(y)
            Wk, yy = W[:, ok], y[ok]
            sw = Wk.sum(1)
            var_b = (Wk @ (yy * yy)) / sw - ((Wk @ yy) / sw) ** 2

            def r2(name):
                e = (wide[name].to_numpy(float)[ok] - yy) ** 2
                return 1 - e.mean() / yy.var(), 1 - ((Wk @ e) / sw) / var_b

            r0, b0 = r2(ref[c])
            for m, (lat, win) in models[c].items():
                v, vb = r2(m)
                boot[(t, c, lat, win)] = vb
                g = vb - b0
                rows.append({"target": t, "cell": c, "model": m, "family": lat,
                             "window": win, "n": int(ok.sum()), "r2": v,
                             "r2_lo": np.percentile(vb, 2.5), "r2_hi": np.percentile(vb, 97.5),
                             "gain_vs_ref": v - r0, "gain_lo": np.percentile(g, 2.5),
                             "gain_hi": np.percentile(g, 97.5), "ref_r2": r0})
    res = pd.DataFrame(rows)
    res.to_csv(a.out_dir / f"rest_density_{head}.csv", index=False)

    # ---- figure: rows = targets, columns = cells ------------------------------------
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fams = sorted(res["family"].unique(), key=fam_key)
    xpos = {f: i for i, f in enumerate(fams)}
    xt = ["standard" if f == "base" else str(f) for f in fams]
    fig, axes = plt.subplots(len(a.targets), len(cells),
                             figsize=(4.3 * len(cells) + 0.6, 3.3 * len(a.targets) + 0.5),
                             squeeze=False, sharey="row", sharex=True,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.06})
    for i, t in enumerate(a.targets):
        for j, c in enumerate(cells):
            ax = axes[i][j]
            rc = res[(res["target"] == t) & (res["cell"] == c)]
            ax.axhline(0, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
            wins = sorted(rc["window"].unique())
            for k, win in enumerate(wins):
                g = rc[rc["window"] == win]
                g = g[~((g["family"] == "base") & (g["window"] == a.ref_window))]
                g = g.assign(x=g["family"].map(xpos)).sort_values("x")
                off = (k - (len(wins) - 1) / 2) * 0.18
                col = WIN_COLOR.get(win, INK2)
                mh = g[g["family"] != "base"]
                ax.plot(mh["x"] + off, mh["gain_vs_ref"], color=col, lw=1.0, alpha=0.6, zorder=2)
                ax.errorbar(g["x"] + off, g["gain_vs_ref"],
                            yerr=[g["gain_vs_ref"] - g["gain_lo"], g["gain_hi"] - g["gain_vs_ref"]],
                            fmt="o", color=col, ms=6, mec="white", mew=1.2, elinewidth=1.4,
                            capsize=0, zorder=3, label=f"pretrained at window {win}")
            ax.set_title(f"reference R$^2$ {rc['ref_r2'].iloc[0]:.3f}", loc="right",
                         fontsize=8, color=MUTED)
            ax.set_xticks(range(len(fams)), xt, fontsize=8)
            ax.grid(True, axis="y", color=RULE, lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(length=0)
            if i == 0:
                ax.set_title(cell_label(c), loc="left", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{TLABEL.get(t, t)}\nR$^2$ $-$ R$^2$(reference)")
            if i == len(a.targets) - 1:
                ax.set_xlabel("attention: standard, or MHLA latent size")
    hl = {}                                     # one entry per window size, from every panel
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    hl = dict(sorted(hl.items(), key=lambda kv: int(kv[0].split()[-1])))
    fig.legend(hl.values(), hl.keys(), frameon=False, fontsize=8.5, ncol=len(hl),
               loc="upper right", bbox_to_anchor=(0.98, 1.0))
    n0 = int(res["n"].iloc[0])
    fig.suptitle(f"{head} head: gain over standard attention pretrained at window "
                 f"{a.ref_window}, run on the same data, in each cell",
                 x=0.02, ha="left", y=1.035, fontsize=11.5)
    fig.text(0.02, 0.985, f"{n0:,} sources, the same in every cell; dashed line = the "
             "reference; bars = 95% paired bootstrap intervals", ha="left", fontsize=8,
             color=MUTED)
    p = a.out_dir / f"rest_density_{head}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")

    # ---- console: R^2 table and paired differences between cells --------------------
    con = []
    for t in a.targets:
        rt = res[res["target"] == t]
        tab = rt.pivot_table(index=["window", "family"], columns="cell", values="r2") \
            .reindex(columns=cells)
        tab = tab.loc[sorted(tab.index, key=lambda k: (k[0], fam_key(k[1])))]
        print(f"\n{t}: out-of-fold R^2 by encoder (window, attention) and cell")
        print(tab.round(4).to_string())
        print(f"{t}: paired differences between cells, R^2(first) - R^2(second) "
              "[95%]  * = interval excludes 0")
        for name, c1, c2, what in CONTRASTS:
            if c1 not in cells or c2 not in cells:
                continue
            print(f"  {name}: {c1} - {c2}  ({what})")
            for (win, lat) in tab.index:
                k1, k2 = (t, c1, lat, win), (t, c2, lat, win)
                if k1 not in boot or k2 not in boot:
                    continue
                d = boot[k1] - boot[k2]
                pt = tab.loc[(win, lat), c1] - tab.loc[(win, lat), c2]
                lo, hi = np.percentile(d, [2.5, 97.5])
                star = "*" if lo > 0 or hi < 0 else " "
                nm = "standard" if lat == "base" else f"MHLA {lat}"
                print(f"     window {win} {nm:>12}  {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] {star}")
                con.append({"target": t, "contrast": name, "cell_a": c1, "cell_b": c2,
                            "window": win, "family": lat, "diff": pt, "lo": lo, "hi": hi})
    pd.DataFrame(con).to_csv(a.out_dir / f"rest_density_{head}_contrasts.csv", index=False)
    print(f"\nwrote {a.out_dir / f'rest_density_{head}.csv'} and "
          f"{a.out_dir / f'rest_density_{head}_contrasts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
