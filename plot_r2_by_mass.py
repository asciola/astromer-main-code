#!/usr/bin/env python3
"""Black-hole mass R^2 as a function of true mass, windows 200 / 400 / 600 overplotted.

Reads the observed-frame fit_regressor.py predictions (whole light curves, one row
per source and model) for the ridge and MLP heads. Panels: rows = encoder (standard
attention, and one MHLA latent), columns = head; one line per pretraining window size.

Why not R^2 inside each bin: within a narrow mass bin the true masses barely vary,
so a bin's own R^2 sits near or below zero whatever the model does. Each bin is
instead scored against the FULL sample's variance,

    R^2_bin = 1 - MSE(bin) / Var(true mass, all sources)

which is on the R^2 scale and averages (weighted by bin counts) to the usual overall
R^2. Bands = +-1.96 standard errors of the bin's MSE.

    python plot_r2_by_mass.py --ridge cluster_results/agn_ridge/predictions.csv \\
        --mlp cluster_results/agn_mlp/predictions.csv --out cluster_results/r2_by_mass.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_span_window_contour import INK, INK2, MUTED, RULE, parse

WINDOWS = [200, 400, 600]
SHADES = {200: "#8fbbec", 400: "#2a78d6", 600: "#123c70"}   # one hue, light to dark
EDGES = [-np.inf, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, np.inf]


def load(path, families):
    keep = []
    for ch in pd.read_csv(path, usecols=["source_id", "m_true", "m_pred", "model"],
                          dtype={"source_id": str}, chunksize=2_000_000):
        names = {m: parse(m) for m in ch["model"].unique()}    # parse each name once
        names = {m: v for m, v in names.items() if v[0] in families and v[1] in WINDOWS}
        ch = ch[ch["model"].isin(names)].copy()
        ch["family"] = ch["model"].map(lambda m: names[m][0])
        ch["window"] = ch["model"].map(lambda m: names[m][1])
        keep.append(ch)
    p = pd.concat(keep)
    return p[p["m_true"] > 0]                      # 0 = catalog "not measured"


def by_mass(p):
    var_all = p.drop_duplicates("source_id")["m_true"].var(ddof=0)
    p = p.assign(bin=pd.cut(p["m_true"], EDGES, right=False, labels=False),
                 se=(p["m_pred"] - p["m_true"]) ** 2)
    g = p.groupby(["family", "window", "bin"])["se"].agg(["mean", "std", "count"]).reset_index()
    g["r2"] = 1 - g["mean"] / var_all
    half = 1.96 * g["std"] / np.sqrt(g["count"]) / var_all
    g["lo"], g["hi"] = g["r2"] - half, g["r2"] + half
    tot = p.groupby(["family", "window"])["se"].mean().rename("mse_all").reset_index()
    tot["r2_all"] = 1 - tot["mse_all"] / var_all
    return g, tot


def gains(p):
    """Per mass bin, paired on sources: (MSE at window 200 - MSE at window W) / Var(all),
    i.e. the bin's R^2 change from moving the same encoder family from window 200 to W."""
    var_all = p.drop_duplicates("source_id")["m_true"].var(ddof=0)
    p = p.assign(se=(p["m_pred"] - p["m_true"]) ** 2)
    wide = p.pivot_table(index="source_id", columns=["family", "window"], values="se")
    mt = p.drop_duplicates("source_id").set_index("source_id")["m_true"].reindex(wide.index)
    b = pd.cut(mt, EDGES, right=False, labels=False)
    rows = []
    for fam in wide.columns.get_level_values(0).unique():
        for W in WINDOWS[1:]:
            d = (wide[(fam, 200)] - wide[(fam, W)]) / var_all
            for k, x in d.groupby(b):
                se = 1.96 * x.std() / np.sqrt(len(x))
                rows.append({"family": fam, "window": W, "bin": k, "gain": x.mean(),
                             "lo": x.mean() - se, "hi": x.mean() + se, "n": len(x)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ridge", default="cluster_results/agn_ridge/predictions.csv")
    ap.add_argument("--mlp", default="cluster_results/agn_mlp/predictions.csv")
    ap.add_argument("--latent", default="8", help="MHLA latent for the second row (default 8)")
    ap.add_argument("--out", type=Path, default=Path("cluster_results/r2_by_mass.png"))
    a = ap.parse_args()
    fams = ["base", a.latent]

    res, gn, tots, counts = {}, {}, [], None
    for head, path in (("ridge", a.ridge), ("mlp", a.mlp)):
        p = load(path, fams)
        g, tot = by_mass(p)
        res[head] = g
        gn[head] = gains(p)
        tots.append(tot.assign(head=head))
        if counts is None:
            src = p.drop_duplicates("source_id")
            counts = pd.cut(src["m_true"], EDGES, right=False, labels=False).value_counts().sort_index()
        print(f"{head}: {p['source_id'].nunique():,} sources")
    tot = pd.concat(tots)
    print(tot.round(4).to_string(index=False))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    nb = len(EDGES) - 1
    labels = []
    for k in range(nb):
        lo, hi = EDGES[k], EDGES[k + 1]
        rng = f"< {hi:g}" if not np.isfinite(lo) else (f"{lo:g}+" if not np.isfinite(hi) else f"{lo:g}")
        labels.append(f"{rng}\n{int(counts.get(k, 0)):,}")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.32, "wspace": 0.06,
                                          "height_ratios": [1, 1.1, 1.1]})

    def frame(ax):
        ax.grid(True, axis="y", color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)

    for j, head in enumerate(["ridge", "mlp"]):
        hn = "ridge" if head == "ridge" else "MLP"
        # row 0: the bin R^2 itself, standard attention (the three windows coincide)
        ax = axes[0][j]
        g = res[head]
        ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        for W in WINDOWS:
            s_ = g[(g["family"] == "base") & (g["window"] == W)].sort_values("bin")
            ra = tot[(tot["head"] == head) & (tot["family"] == "base") & (tot["window"] == W)]["r2_all"].iloc[0]
            ax.plot(s_["bin"], s_["r2"], color=SHADES[W], lw=2, marker="o", ms=4,
                    mec="white", mew=0.8, label=f"window {W}  (overall R\u00b2 {ra:.3f})")
        ax.set_title(f"Standard attention, {hn} head: R\u00b2 in each bin", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower center")
        frame(ax)
        if j == 0:
            ax.set_ylabel("1 \u2212 MSE(bin) / Var(all)")
        # rows 1-2: change from window 200, paired on sources
        for i, fam in ((1, "base"), (2, a.latent)):
            ax = axes[i][j]
            q = gn[head]
            ax.axhline(0, color=SHADES[200], lw=1.6, label="window 200 (reference)")
            for W in WINDOWS[1:]:
                s_ = q[(q["family"] == fam) & (q["window"] == W)].sort_values("bin")
                ax.fill_between(s_["bin"], s_["lo"], s_["hi"], color=SHADES[W], alpha=0.18, lw=0)
                ax.plot(s_["bin"], s_["gain"], color=SHADES[W], lw=2, marker="o", ms=4,
                        mec="white", mew=0.8, label=f"window {W}")
            name = "Standard attention" if fam == "base" else f"MHLA latent {fam}"
            ax.set_title(f"{name}, {hn} head: change from window 200", loc="left", fontsize=10)
            ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
            frame(ax)
            if j == 0:
                ax.set_ylabel("\u0394 R\u00b2 in the bin\n(paired on sources)")
            if i == 2:
                ax.set_xticks(range(nb), labels, fontsize=7.5)
                ax.set_xlabel(r"true $\log M_{\rm BH}$ bin (sources per bin)")
    fig.suptitle("Black-hole mass R² by true mass, pretraining windows 200, 400 and 600 "
                 "(observed frame, whole light curves)", x=0.02, ha="left", y=0.995,
                 fontsize=11.5)
    fig.text(0.02, 0.968, "Each bin's error is scored against the variance of all masses, so a "
             "line's count-weighted average is its overall R². Top: the edges fall below 0 "
             "because every model pulls predictions toward the mean mass.\nBelow: the same "
             "sources at window 400 or 600 minus window 200, same encoder family; "
             "bands = ±1.96 standard errors.", fontsize=8, color=MUTED, ha="left", va="top")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")
    pd.concat([r.assign(head=h) for h, r in res.items()]).to_csv(a.out.with_suffix(".csv"), index=False)
    pd.concat([r.assign(head=h) for h, r in gn.items()]).to_csv(
        a.out.with_name(a.out.stem + "_gain.csv"), index=False)
    print(f"wrote {a.out} and {a.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
