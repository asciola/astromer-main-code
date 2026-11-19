#!/usr/bin/env python3
"""Rest-frame time covered by a W-point window, for W = 200 ... 1000.

For every r-band rest-frame light curve with a black-hole mass (lc_index mbh_valid),
every run of W consecutive observations is a window; its span is
t[i + W - 1] - t[i] in rest-frame days. Two panels:

  left   the distribution of those spans for each W, every light curve weighted
         equally (a curve's windows share one unit of weight), with the curve count
         and the median span;
  right  how many light curves have a W-point window whose span is within +-10% of
         a target duration D: the rule restframe_windows.py uses to fill a cell
         (the window closest to D, kept if within 10%, qualifies exactly when any
         window does). Its peak is each W's "best span".

    python plot_window_spans.py --src data/records/agn_test_rest/fold_0/test \\
        --out cluster_results/window_spans.png
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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

from plot_span_window_contour import INK, INK2, MUTED, RULE  # noqa: E402

WINDOWS = [200, 400, 600, 800, 1000]
# one hue, light to dark: longer windows darker
SHADES = ["#8fbbec", "#5b9be2", "#2a78d6", "#1c5aa6", "#123c70"]
TOL = 0.10


def load_times(src, bands, wmin):
    idx = pd.read_csv(os.path.join(src, "lc_index.csv"), dtype={"lcid": str, "source_id": str})
    keep = idx[idx["band_name"].isin(bands) & idx["mbh_valid"].astype(bool)
               & (idx["n_points"] >= wmin)]
    want = set(keep["lcid"])
    times = []
    for p in sorted(glob.glob(os.path.join(src, "*.record"))):
        for raw in tf.data.TFRecordDataset(p).as_numpy_iterator():
            ex = tf.train.SequenceExample.FromString(raw)
            if ex.context.feature["ID"].bytes_list.value[0].decode() in want:
                times.append(np.asarray(
                    ex.feature_lists.feature_list["mjd"].feature[0].float_list.value, float))
    return times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/records/agn_test_rest/fold_0/test")
    ap.add_argument("--bands", nargs="+", default=["r"])
    ap.add_argument("--out", type=Path, default=Path("cluster_results/window_spans.png"))
    a = ap.parse_args()

    times = load_times(a.src, a.bands, min(WINDOWS))
    print(f"{len(times):,} {'/'.join(a.bands)}-band light curves with a mass and "
          f">= {min(WINDOWS)} points")

    edges = np.arange(0, 2001, 20.0)
    D = np.arange(100, 2001, 25.0)
    dens, reach, stats = {}, {}, []
    for W in WINDOWS:
        h = np.zeros(len(edges) - 1)
        r = np.zeros(len(D), int)
        meds, n = [], 0
        for t in times:
            if len(t) < W:
                continue
            n += 1
            s = np.sort(t[W - 1:] - t[:len(t) - W + 1])
            h += np.histogram(s, edges)[0] / len(s)          # one unit of weight per curve
            meds.append(np.median(s))
            # any span in [0.9 D, 1.1 D]
            r += np.searchsorted(s, (1 + TOL) * D, "right") > np.searchsorted(s, (1 - TOL) * D, "left")
        dens[W], reach[W] = h, r
        q = np.percentile(meds, [5, 50, 95]) if meds else [np.nan] * 3
        best = D[np.argmax(r)]
        stats.append({"window": W, "curves": n, "median_span_p5": q[0], "median_span": q[1],
                      "median_span_p95": q[2], "best_span": best, "curves_at_best": int(r.max())})
    st = pd.DataFrame(stats)
    print(st.round(0).to_string(index=False))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(len(WINDOWS), 2, width_ratios=[1, 1.05], wspace=0.22, hspace=0.12)
    ctr = (edges[:-1] + edges[1:]) / 2
    for k, (W, col) in enumerate(zip(WINDOWS, SHADES)):
        ax = fig.add_subplot(gs[k, 0])
        row = st.iloc[k]
        f = dens[W] / max(row["curves"], 1)                    # share of curves per bin
        ax.fill_between(ctr, f, color=col, alpha=0.35, lw=0)
        ax.plot(ctr, f, color=col, lw=1.6)
        ax.axvline(row["median_span"], color=col, lw=1, ls=(0, (3, 2)))
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, None)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        ax.text(-0.02, 0.35, f"{W} pts", transform=ax.transAxes, ha="right", va="center",
                fontsize=9.5, color=INK)
        ax.text(-0.02, 0.0, f"{int(row['curves']):,} curves", transform=ax.transAxes,
                ha="right", va="center", fontsize=7.5, color=MUTED)
        ax.text(row["median_span"] + 15, ax.get_ylim()[1] * 0.85,
                f"median {row['median_span']:.0f} d", fontsize=7.5, color=INK2, va="top")
        if k < len(WINDOWS) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("rest-frame days spanned by one W-point window")
        if k == 0:
            ax.set_title("Span of W-point windows (each light curve weighted equally)",
                         loc="left", fontsize=10, pad=8)

    ax = fig.add_subplot(gs[:, 1])
    for k, (W, col) in enumerate(zip(WINDOWS, SHADES)):
        r = reach[W]
        ax.plot(D, np.where(r > 0, r, np.nan), color=col, lw=2)
        b = st.iloc[k]
        ax.plot(b["best_span"], b["curves_at_best"], "o", color=col, ms=6, mec="white", mew=1.2)
        ax.annotate(f"{W} pts: {int(b['curves_at_best']):,} at {b['best_span']:.0f} d",
                    (b["best_span"], b["curves_at_best"]), xytext=(8, 4),
                    textcoords="offset points", fontsize=8, color=INK2)
    for d0 in (300, 600, 900):                               # cells run, and the 600-pt option
        ax.axvline(d0, color="#c9cfce", lw=0.9, ls=(0, (2, 3)), zorder=0)
        ax.text(d0 + 12, 0.02, f"{d0} d", fontsize=7.5, color=MUTED,
                transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xlim(100, 2000)
    ax.set_xlabel("target rest-frame duration D (days)")
    ax.set_ylabel(f"light curves with a window within ±{TOL:.0%} of D")
    ax.set_title("Light curves that could fill a cell; dot = best span", loc="left",
                 fontsize=10, pad=20)
    ax.text(0, 1.015, "dotted: the 300 and 600 d cells run so far, and 900 d for a "
            "window-600 cell", transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.grid(True, which="major", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")
    st.to_csv(a.out.with_suffix(".csv"), index=False)
    print(f"wrote {a.out} and {a.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
