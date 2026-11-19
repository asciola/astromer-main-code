#!/usr/bin/env python3
"""Histogram of black-hole masses (LOGMBH), the full AGN sample against the subsets
the rest-frame experiments use, to see whether long-window samples lean in mass.

  all sources          every source with a valid mass (lc_index mbh_valid, one row
                       per source, not per band)
  duration experiment  sources in restframe_windows.py --durations (D200 cell)
  density experiment   sources in restframe_windows.py --cells (p200_d300 cell)
  window-600 option    r-band curves that would also fill a 600-point / 900-day cell
                       at the same 0.67 observations per day (computed here with
                       restframe_windows.py's rule: the window closest to D, kept if
                       within 10%)

Subsets differ ~15x in size, so the plot shows the share of each sample per bin;
counts and medians are in the legend.

    python plot_mass_hist.py --out cluster_results/mass_hist.png
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
from restframe_windows import best_window  # noqa: E402

FULL = "#c9cfce"                               # the whole sample: quiet grey fill
SUBSETS = ["#2a78d6", "#eb6834", "#1baf7a"]    # categorical slots 1-3 (validated)
LADDER = [(200, 300), (400, 600), (600, 900)]  # same density as the cells run so far


def masses(index_csv):
    idx = pd.read_csv(index_csv, dtype={"lcid": str, "source_id": str})
    ok = idx["mbh_valid"].astype(bool) & (idx["LOGMBH"] > 0)   # 0 = catalog "not measured"
    return idx[ok].drop_duplicates("source_id").set_index("source_id")["LOGMBH"]


def window600_sources(src, idx):
    cand = idx[(idx["band_name"] == "r") & idx["mbh_valid"].astype(bool)
               & (idx["n_points"] >= LADDER[-1][0])]
    want = dict(zip(cand["lcid"], cand["source_id"]))
    keep = set()
    for p in sorted(glob.glob(os.path.join(src, "*.record"))):
        for raw in tf.data.TFRecordDataset(p).as_numpy_iterator():
            ex = tf.train.SequenceExample.FromString(raw)
            lc = ex.context.feature["ID"].bytes_list.value[0].decode()
            if lc not in want:
                continue
            t = np.asarray(ex.feature_lists.feature_list["mjd"].feature[0].float_list.value, float)
            fits = True
            for n, D in LADDER:
                i, span = best_window(t, n, D)
                if i is None or abs(span - D) > 0.1 * D:
                    fits = False
                    break
            if fits:
                keep.add(want[lc])
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/records/agn_test_rest/fold_0/test")
    ap.add_argument("--duration-index",
                    default="data/records/agn_rest_windows/D200/fold_0/test/lc_index.csv")
    ap.add_argument("--density-index",
                    default="data/records/agn_rest_cells/p200_d300/fold_0/test/lc_index.csv")
    ap.add_argument("--out", type=Path, default=Path("cluster_results/mass_hist.png"))
    a = ap.parse_args()

    idx = pd.read_csv(os.path.join(a.src, "lc_index.csv"), dtype={"lcid": str, "source_id": str})
    full = masses(os.path.join(a.src, "lc_index.csv"))
    w600 = window600_sources(a.src, idx)
    samples = [("all sources with a mass", full),
               ("duration experiment", masses(a.duration_index)),
               ("density experiment", masses(a.density_index)),
               ("window-600 option", full[full.index.isin(w600)])]

    rows = []
    for name, m in samples:
        q = np.percentile(m, [16, 50, 84])
        rows.append({"sample": name, "sources": len(m), "p16": q[0], "median": q[1], "p84": q[2]})
    st = pd.DataFrame(rows)
    print(st.round(2).to_string(index=False))

    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    edges = np.arange(6.5, 11.01, 0.1)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, ((name, m), r) in enumerate(zip(samples, rows)):
        h = np.histogram(m, edges)[0] / len(m) * 100
        label = f"{name}: {r['sources']:,} sources, median {r['median']:.2f}"
        if k == 0:
            ax.fill_between(edges[:-1], h, step="post", color=FULL, lw=0, label=label)
        else:
            ax.step(edges[:-1], h, where="post", color=SUBSETS[k - 1], lw=1.8, label=label)
    ax.set_xlim(6.8, 10.8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)               # headroom for the legend
    ax.set_xlabel(r"black-hole mass, $\log M_{\rm BH}$ (LOGMBH)")
    ax.set_ylabel("share of the sample per 0.1-dex bin (%)")
    ax.set_title("Black-hole masses: the full sample and the subsets the rest-frame "
                 "experiments use", loc="left", fontsize=11, pad=10)
    ax.grid(True, axis="y", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")
    st.to_csv(a.out.with_suffix(".csv"), index=False)
    print(f"wrote {a.out} and {a.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
