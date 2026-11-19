#!/usr/bin/env python3
"""Slide-sized figures for the rest-frame experiments (November 2026 status slides).

Each figure is drawn at 6.1 x 3.6 in, the picture slot on the left of the
Longstromer deck's TITLE_AND_BODY slides, from CSVs the other scripts already wrote:

  slide_rest_duration.png   rest_duration_{ridge,mlp}.csv (plot_rest_duration*.py):
                            MHLA's gain over standard attention on log M_BH as the
                            200-point window spans 200, 300 and 400 rest-frame days.
  slide_rest_density.png    rest_density_{ridge,mlp}.csv (plot_rest_density.py):
                            every encoder's gain over standard attention at window
                            200, per cell, log M_BH only.
  slide_what_moves_r2.png   rest_density_{ridge,mlp}{,_contrasts}.csv: the effect of
                            time, one-window span, density and window-400 pretraining
                            on log M_BH for standard attention, with 95% intervals.

    python plot_rest_slides.py --duration-dir cluster_results/rest_duration \\
        --density-dir cluster_results/rest_density --out-dir cluster_results/slides
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_span_window_contour import INK, INK2, MUTED, RULE

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9c0bf"   # categorical slots 1, 2; quiet grey
SIZE = (6.1, 3.6)
CELLS = [("p200_d300", "200 pts / 300 d"), ("p400_d600", "400 pts / 600 d"),
         ("p400_d600_t2", "200 pts / 600 d (thinned)")]


def style():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": 8.5, "axes.titlecolor": INK,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"]})


def quiet(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, axis="y", color=RULE, lw=0.6)
    ax.set_axisbelow(True)


def fam_key(lat):
    return -1 if lat == "base" else int(lat)


def duration(d: Path, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, head in zip(axes, ["ridge", "mlp"]):
        r = pd.read_csv(d / f"rest_duration_{head}.csv")
        r = r[(r["target"] == "m") & ~r["latent"].isin(["base", "control"])]
        ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
        for lat, g in r.groupby("latent"):
            g = g.sort_values("days")
            hi = lat == "8"
            if hi:
                ax.fill_between(g["days"], g["gain_lo"], g["gain_hi"], color=BLUE, alpha=0.15, lw=0)
            ax.plot(g["days"], g["gain_vs_standard"], color=BLUE if hi else GREY,
                    lw=2 if hi else 1, marker="o", ms=4 if hi else 2.5, zorder=3 if hi else 2,
                    mec="white", mew=0.6)
        last = r[(r["latent"] == "8")].sort_values("days").iloc[-1]
        ax.annotate("MHLA latent 8", (last["days"], last["gain_vs_standard"]),
                    xytext=(-6, 6), textcoords="offset points", ha="right", color=BLUE,
                    fontsize=7.5)
        ax.set_title(f"{'Ridge' if head == 'ridge' else 'MLP'} head", loc="left")
        ax.set_xticks([200, 300, 400])
        quiet(ax)
    fig.supxlabel("rest-frame days spanned by a 200-point window", fontsize=8, color=INK2, y=-0.02)
    axes[0].set_ylabel(r"mass R$^2$ $-$ R$^2$(standard attention)")
    axes[0].text(300, -0.0008, "standard attention", ha="center", va="top", color=MUTED,
                 fontsize=7)
    axes[1].text(203, axes[1].get_ylim()[1] * 0.92, "grey: other MHLA latents\n"
                 "band: 68% paired bootstrap", color=MUTED, fontsize=6.5, va="top")
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def density(d: Path, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=SIZE, sharey="row", sharex=True,
                             gridspec_kw={"wspace": 0.06, "hspace": 0.32})
    for i, head in enumerate(["ridge", "mlp"]):
        r = pd.read_csv(d / f"rest_density_{head}.csv")
        r = r[r["target"] == "m"]
        fams = sorted(r["family"].unique(), key=fam_key)
        xpos = {f: k for k, f in enumerate(fams)}
        for j, (c, label) in enumerate(CELLS):
            ax = axes[i][j]
            ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
            rc = r[r["cell"] == c]
            for win, col, off in [(200, BLUE, -0.15), (400, ORANGE, 0.15)]:
                g = rc[(rc["window"] == win) & ~((rc["family"] == "base") & (rc["window"] == 200))]
                if g.empty:
                    continue
                g = g.assign(x=g["family"].map(xpos)).sort_values("x")
                mh = g[g["family"] != "base"]
                ax.plot(mh["x"] + off, mh["gain_vs_ref"], color=col, lw=0.8, alpha=0.6)
                ax.errorbar(g["x"] + off, g["gain_vs_ref"],
                            yerr=[g["gain_vs_ref"] - g["gain_lo"], g["gain_hi"] - g["gain_vs_ref"]],
                            fmt="o", color=col, ms=3.2, mec="white", mew=0.5, elinewidth=0.9,
                            capsize=0, label=f"pretrained at window {win}")
            ax.set_xticks(range(len(fams)),
                          ["std" if f == "base" else str(f) for f in fams], fontsize=6)
            quiet(ax)
            if i == 0:
                ax.set_title(label, loc="left", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{'Ridge' if head == 'ridge' else 'MLP'}\nmass R$^2$ gain", fontsize=7.5)
    fig.supxlabel("attention: std = standard, numbers = MHLA latent size", fontsize=7,
                  color=INK2, y=0.0)
    h, l = axes[0][1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.09))
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def effects(d: Path, out: Path):
    """Forest plot: four effects on mass R^2 for standard attention, both heads."""
    rows = []
    for head in ["ridge", "mlp"]:
        con = pd.read_csv(d / f"rest_density_{head}_contrasts.csv")
        con = con[(con["target"] == "m") & (con["family"] == "base")]
        res = pd.read_csv(d / f"rest_density_{head}.csv")
        res = res[(res["target"] == "m") & (res["family"] == "base") & (res["window"] == 400)
                  & (res["cell"] == "p400_d600_t2")].iloc[0]

        def c(name, win):
            x = con[(con["contrast"] == name) & (con["window"] == win)].iloc[0]
            return x["diff"], x["lo"], x["hi"]
        rows += [(head, "Twice the time, same density",
                  "200 pts / 300 d to 400 pts / 600 d, window-200 model", *c("time", 200)),
                 (head, "Twice the time in one window",
                  "200 pts / 300 d to 200 pts / 600 d, window-200 model", *c("thinned", 200)),
                 (head, "Twice the density, same time",
                  "200 to 400 pts over 600 d, window-400 model", *c("density", 400)),
                 (head, "Pretrained at window 400, not 200",
                  "same input: 200 pts over 600 d", res["gain_vs_ref"], res["gain_lo"],
                  res["gain_hi"])]
    df = pd.DataFrame(rows, columns=["head", "label", "note", "d", "lo", "hi"])
    labels = list(dict.fromkeys(df["label"]))
    fig, ax = plt.subplots(figsize=SIZE)
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    for head, col, off in [("ridge", "#8f9a99", -0.16), ("mlp", BLUE, 0.16)]:
        g = df[df["head"] == head]
        y = np.array([labels.index(s) for s in g["label"]]) + off
        ax.errorbar(g["d"], y, xerr=[g["d"] - g["lo"], g["hi"] - g["d"]], fmt="o", color=col,
                    ms=5, mec="white", mew=0.7, elinewidth=1.6, capsize=0,
                    label="ridge head" if head == "ridge" else "MLP head")
    ax.set_yticks([])
    for k, lab in enumerate(labels):
        note = df.loc[df["label"] == lab, "note"].iloc[0]       # one note per row of the plot
        ax.annotate(lab, (0, k), xycoords=("axes fraction", "data"), xytext=(-8, 1),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK)
        ax.annotate(note, (0, k), xycoords=("axes fraction", "data"), xytext=(-8, -2),
                    textcoords="offset points", ha="right", va="top", fontsize=6.5, color=MUTED)
    ax.set_ylim(len(labels) - 0.5, -0.6)
    ax.set_xlim(-0.012, 0.042)
    ax.set_xlabel(r"change in mass R$^2$ (standard attention; bars = 95% paired bootstrap)")
    ax.text(0.0005, -0.55, "no change", color=MUTED, fontsize=6.5, va="bottom")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration-dir", type=Path, default=Path("cluster_results/rest_duration"))
    ap.add_argument("--density-dir", type=Path, default=Path("cluster_results/rest_density"))
    ap.add_argument("--out-dir", type=Path, default=Path("cluster_results/slides"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    duration(a.duration_dir, a.out_dir / "slide_rest_duration.png")
    density(a.density_dir, a.out_dir / "slide_rest_density.png")
    effects(a.density_dir, a.out_dir / "slide_what_moves_r2.png")
    print(f"wrote 3 figures to {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
