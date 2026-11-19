#!/usr/bin/env python3
"""Black-hole mass error as a function of true mass, windows 200 / 400 / 600 overplotted.

Reads the observed-frame fit_regressor.py predictions (whole light curves, one row
per source and model). For each encoder family (--families: base = standard
attention, or an MHLA latent) and head (--heads), two panels:

  top     the metric in each true-mass bin, one line per pretraining window size;
  bottom  each bin's change from window 200, paired on the same sources, same family.

--metric mse (default): mean squared error in dex^2 (lower is better); the change is
    MSE(window W) - MSE(window 200), so below 0 = window W is better.
--metric r2: each bin scored against the FULL sample's variance,
    1 - MSE(bin) / Var(true mass, all sources), which is on the R^2 scale and averages
    (weighted by bin counts) to the overall R^2. A bin's own R^2 would sit near or
    below zero whatever the model does, because masses inside a narrow bin barely vary.

Bands = +-1.96 standard errors (of the bin MSE, or of the paired per-source difference).

    python plot_r2_by_mass.py --metric mse --families 8 --heads mlp \\
        --out cluster_results/mse_by_mass_mhla8_mlp.png
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
HEADS = {"ridge": "ridge", "mlp": "MLP"}


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
    p = p[p["m_true"] > 0]                         # 0 = catalog "not measured"
    return p.assign(se=(p["m_pred"] - p["m_true"]) ** 2)


def scale(p, metric):
    """value(mse) -> metric, and the factor that turns an MSE difference into one."""
    if metric == "mse":
        return (lambda m: m), 1.0
    var_all = p.drop_duplicates("source_id")["m_true"].var(ddof=0)
    return (lambda m: 1 - m / var_all), -1.0 / var_all


def by_mass(p, metric):
    f, _ = scale(p, metric)
    p = p.assign(bin=pd.cut(p["m_true"], EDGES, right=False, labels=False))
    g = p.groupby(["family", "window", "bin"])["se"].agg(["mean", "std", "count"]).reset_index()
    half = 1.96 * g["std"] / np.sqrt(g["count"])
    g["value"], g["lo"], g["hi"] = f(g["mean"]), f(g["mean"] + half), f(g["mean"] - half)
    tot = p.groupby(["family", "window"])["se"].mean().rename("mse_all").reset_index()
    tot["value_all"] = f(tot["mse_all"])
    return g, tot


def changes(p, metric):
    """Per bin, paired on sources: the metric at window W minus at window 200."""
    _, k = scale(p, metric)
    wide = p.pivot_table(index="source_id", columns=["family", "window"], values="se")
    mt = p.drop_duplicates("source_id").set_index("source_id")["m_true"].reindex(wide.index)
    b = pd.cut(mt, EDGES, right=False, labels=False)
    rows = []
    for fam in wide.columns.get_level_values(0).unique():
        for W in WINDOWS[1:]:
            d = (wide[(fam, W)] - wide[(fam, 200)]) * k
            for kk, x in d.groupby(b):
                se = 1.96 * x.std() / np.sqrt(len(x))
                rows.append({"family": fam, "window": W, "bin": kk, "change": x.mean(),
                             "lo": x.mean() - se, "hi": x.mean() + se, "n": len(x)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ridge", default="cluster_results/agn_ridge/predictions.csv")
    ap.add_argument("--mlp", default="cluster_results/agn_mlp/predictions.csv")
    ap.add_argument("--metric", choices=["mse", "r2"], default="mse")
    ap.add_argument("--families", nargs="+", default=["8"],
                    help="encoder families: base (standard attention) and/or MHLA latents")
    ap.add_argument("--heads", nargs="+", choices=list(HEADS), default=["mlp"])
    ap.add_argument("--out", type=Path, default=Path("cluster_results/mse_by_mass.png"))
    a = ap.parse_args()
    mse = a.metric == "mse"
    unit = "MSE (dex²)" if mse else "1 − MSE(bin) / Var(all)"

    res, ch, tots, counts = {}, {}, [], None
    for head in a.heads:
        p = load(getattr(a, head), a.families)
        g, tot = by_mass(p, a.metric)
        res[head], ch[head] = g, changes(p, a.metric)
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

    nrow, ncol = 2 * len(a.families), len(a.heads)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol + 1.2, 3.3 * nrow + 0.9),
                             sharex=True, squeeze=False,
                             gridspec_kw={"hspace": 0.38, "wspace": 0.12})

    def frame(ax):
        ax.grid(True, axis="y", color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)

    for fi, fam in enumerate(a.families):
        name = "standard attention" if fam == "base" else f"MHLA latent {fam}"
        for j, head in enumerate(a.heads):
            # the metric itself
            ax = axes[2 * fi][j]
            g = res[head]
            if not mse:
                ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
            for W in WINDOWS:
                s_ = g[(g["family"] == fam) & (g["window"] == W)].sort_values("bin")
                v = tot[(tot["head"] == head) & (tot["family"] == fam) & (tot["window"] == W)]["value_all"].iloc[0]
                ax.plot(s_["bin"], s_["value"], color=SHADES[W], lw=2, marker="o", ms=4,
                        mec="white", mew=0.8,
                        label=f"window {W}  (overall {'MSE' if mse else 'R²'} {v:.3f})")
            ax.set_title(f"{name}, {HEADS[head]} head: {'MSE' if mse else 'R²'} in each mass bin",
                         loc="left", fontsize=10)
            ax.legend(frameon=False, fontsize=8, loc="upper center" if mse else "lower center")
            frame(ax)
            if j == 0:
                ax.set_ylabel(unit)
            # change from window 200
            ax = axes[2 * fi + 1][j]
            q = ch[head]
            ax.axhline(0, color=SHADES[200], lw=1.6, label="window 200 (reference)")
            for W in WINDOWS[1:]:
                s_ = q[(q["family"] == fam) & (q["window"] == W)].sort_values("bin")
                ax.fill_between(s_["bin"], s_["lo"], s_["hi"], color=SHADES[W], alpha=0.18, lw=0)
                ax.plot(s_["bin"], s_["change"], color=SHADES[W], lw=2, marker="o", ms=4,
                        mec="white", mew=0.8, label=f"window {W}")
            better = "below 0 = better than window 200" if mse else "above 0 = better than window 200"
            ax.set_title(f"{name}, {HEADS[head]} head: change from window 200 ({better})",
                         loc="left", fontsize=10)
            ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
            frame(ax)
            if j == 0:
                ax.set_ylabel(f"Δ {'MSE (dex²)' if mse else 'R²'} in the bin\n(paired on sources)")
            if 2 * fi + 1 == nrow - 1:
                ax.set_xticks(range(nb), labels, fontsize=7.5)
                ax.set_xlabel(r"true $\log M_{\rm BH}$ bin (sources per bin)")
    what = "mean squared error" if mse else "R²"
    fig.suptitle(f"Black-hole mass {what} by true mass, pretraining windows 200, 400 and 600",
                 x=0.02, ha="left", y=1 - 0.12 / fig.get_figheight(), fontsize=11.5)
    note = ("Observed frame, whole light curves. Top: the error rises at both ends because every "
            "model pulls predictions toward the mean mass.\nBottom: the same sources at window 400 "
            "or 600 minus window 200; bands = ±1.96 standard errors.")
    h_in = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.15 / h_in)                # room for title + two-line note
    fig.text(0.02, 1 - 0.45 / h_in, note, fontsize=8, color=MUTED, ha="left", va="top")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")
    pd.concat([r.assign(head=h) for h, r in res.items()]).to_csv(a.out.with_suffix(".csv"), index=False)
    pd.concat([r.assign(head=h) for h, r in ch.items()]).to_csv(
        a.out.with_name(a.out.stem + "_change.csv"), index=False)
    print(f"wrote {a.out}, {a.out.with_suffix('.csv')} and {a.out.stem}_change.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
