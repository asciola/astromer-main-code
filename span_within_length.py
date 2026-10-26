#!/usr/bin/env python3
"""Does a model gain more on light curves that cover a longer TIME range, once the
number of points is held fixed?

Almost every ZTF curve spans the full ~6 yr baseline, and the ones that do not also
tend to have fewer points (Spearman ~0.57 between r-band span and r-band points), so
binning by span alone (length_mass_gain.py --length span) mixes the two effects. This
script separates them:

  1. Keep sources with fewer than the last --point-bins edge of r-band points (default
     500: above that, short spans barely exist, so there is nothing to compare).
  2. Cut them into point bins (--point-bins) and, independently, into span groups
     (--span-bins, days from first to last r-band observation).
  3. Reweight every span group to the SAME point-bin mix (that of all kept sources),
     so the groups differ in span but not in length.
  4. For every model, per span group: MSE(base) / MSE(model) pooled with those
     weights (> 1 = the model beats the base), and its difference from the longest-
     span group. The intervals are a bootstrap over sources, paired between model and
     base; every group uses the same resampling draws.

Reads the out-of-fold predictions.csv from an EMBEDDINGS-ONLY fit_regressor.py run
(no --with-stats) and lc_index.csv. Writes one CSV:

  target, model, latent, window, span, n   span group and its size
  mse_ratio, lo, hi                        ratio and its --ci interval
  minus_full, minus_full_lo, minus_full_hi
                                           ratio minus the longest-span group's ratio,
                                           with a 95% interval; < 0 = the model's gain
                                           is smaller on shorter spans

    python span_within_length.py --pred cluster_results/agn_mlp/predictions.csv \\
        --index $IDX --out cluster_results/span_within_length_mlp.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL = "CONTROL: summary stats only"
PAT_LATENT = re.compile(r"_K_01_(\d+)_(\d+)$")
PAT_BASE = re.compile(r"_base_\d+_(\d+)_K_baseline$")
PAT_LINEAR = re.compile(r"_linear_\d+_(\d+)_K(?:_.*)?$")


def parse(name):
    m = PAT_LATENT.search(name)
    if m:
        return m.group(1), int(m.group(2))
    m = PAT_BASE.search(name)
    if m:
        return "base", int(m.group(1))
    m = PAT_LINEAR.search(name)
    if m:
        return "linear", int(m.group(1))
    return None, None


def span_labels(edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if a == 0:
            out.append(f"<{int(b)} d")
        elif not np.isfinite(b):
            out.append(f"{int(a)}+ d")
        else:
            out.append(f"{int(a)}-{int(b)} d")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True,
                    help="predictions.csv from an embeddings-only run (agn_ridge or agn_mlp)")
    ap.add_argument("--index", type=Path, required=True, help="lc_index.csv")
    ap.add_argument("--targets", nargs="+", default=["m", "z", "e"], choices=["m", "z", "e"])
    ap.add_argument("--base", default="base_256_200_K_baseline",
                    help="substring naming the reference model (default: base, window 200)")
    ap.add_argument("--band", default="r")
    ap.add_argument("--point-bins", type=float, nargs="+", default=[0, 200, 300, 400, 500],
                    help="point-count bins to match on; the last edge is the cut")
    ap.add_argument("--span-bins", type=float, nargs="+", default=[0, 2000, 2200, np.inf],
                    help="span groups in days; the last group is the reference")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--ci", type=float, default=68.0, help="interval width in percent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("span_within_length.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    lo_q, hi_q = 50 - a.ci / 2, 50 + a.ci / 2
    slabs = span_labels(a.span_bins)
    nbins = len(a.point_bins) - 1

    # ---- points and span per source ---------------------------------------------
    idx = pd.read_csv(a.index, dtype={"source_id": str},
                      usecols=["source_id", "band_name", "n_points", "mjd_min", "mjd_max"])
    r = idx[idx["band_name"] == a.band].drop_duplicates("source_id").set_index("source_id")
    if r.empty:
        print(f"no band {a.band!r} in {a.index}", file=sys.stderr)
        return 1
    r["span"] = r["mjd_max"] - r["mjd_min"]
    r = r[r["n_points"] < a.point_bins[-1]]

    # ---- predictions ------------------------------------------------------------
    pred = pd.read_csv(a.pred, dtype={"source_id": str})
    pred = pred[pred["model"] != CONTROL]
    names = sorted(pred["model"].unique())
    base_hits = [n for n in names if a.base in n]
    if len(base_hits) != 1:
        print(f"--base {a.base!r} matched {base_hits or 'nothing'}; models are:\n  " +
              "\n  ".join(names), file=sys.stderr)
        return 1
    base = base_hits[0]

    rows = []
    for t in a.targets:
        wide = pred.pivot_table(index="source_id", columns="model", values=f"{t}_pred")
        y = pred.drop_duplicates("source_id").set_index("source_id")[f"{t}_true"] \
            .reindex(wide.index)
        ok = y.notna() & wide.index.isin(r.index) & wide[base].notna()
        wide, y = wide[ok], y[ok].to_numpy(float)
        nb = pd.cut(r["n_points"].reindex(wide.index), a.point_bins, right=False,
                    labels=False).to_numpy()
        sb = pd.cut(r["span"].reindex(wide.index), a.span_bins, right=False,
                    labels=False).to_numpy()

        # weight each source so that every span group has the overall point-bin mix
        target_mix = np.bincount(nb, minlength=nbins) / len(nb)
        w = np.zeros(len(y))
        for g in range(len(slabs)):
            sg = sb == g
            cnt = np.bincount(nb[sg], minlength=nbins)
            with np.errstate(divide="ignore", invalid="ignore"):
                w[sg] = np.where(cnt[nb[sg]] > 0, target_mix[nb[sg]] / cnt[nb[sg]], 0)
            missing = [f"{int(a.point_bins[k])}+" for k in range(nbins) if cnt[k] == 0]
            if missing:
                print(f"note: {t} span {slabs[g]} has no sources in point bins {missing}; "
                      "its mix is not fully matched")

        eb = (wide[base].to_numpy(float) - y) ** 2
        BW = np.stack([np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
                       for _ in range(a.bootstrap)]).astype(np.float32)
        for mdl in wide.columns:
            lat, ws = parse(mdl)
            if lat is None or mdl == base:
                continue
            em = (wide[mdl].to_numpy(float) - y) ** 2
            good = np.isfinite(em)
            em = np.where(good, em, 0)
            ratios, boots = [], []
            for g in range(len(slabs)):
                wg = w * (sb == g) * good
                ratios.append((wg @ eb) / (wg @ em))
                boots.append((BW @ (wg * eb)) / (BW @ (wg * em)))
            for g in range(len(slabs)):
                diff = boots[g] - boots[-1]
                rows.append({"target": t, "model": mdl, "latent": lat, "window": ws,
                             "span": slabs[g], "n": int(((sb == g) & good).sum()),
                             "mse_ratio": ratios[g],
                             "lo": np.percentile(boots[g], lo_q),
                             "hi": np.percentile(boots[g], hi_q),
                             "minus_full": ratios[g] - ratios[-1],
                             "minus_full_lo": np.percentile(diff, 2.5),
                             "minus_full_hi": np.percentile(diff, 97.5)})
        print(f"{t}: {len(y):,} sources with < {int(a.point_bins[-1])} {a.band}-band points")

    res = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)

    # ---- console summary --------------------------------------------------------
    print(f"\nMSE({base[-24:]}) / MSE(model), point mix matched; "
          f"(minus longest span, 95% CI):")
    for t, g in res.groupby("target", sort=False):
        cell = g.apply(lambda q: f"{q.mse_ratio:.3f}" + (
            "" if q.span == slabs[-1] else
            f" ({q.minus_full:+.3f} [{q.minus_full_lo:+.3f},{q.minus_full_hi:+.3f}])"), axis=1)
        tab = g.assign(cell=cell, tag=g["latent"].astype(str) + "_" + g["window"].astype(str)) \
            .pivot(index="tag", columns="span", values="cell").reindex(columns=slabs)
        print(f"\n{t}  (n per group: " + ", ".join(
            f"{s}: {n:,}" for s, n in g.groupby("span")["n"].max().reindex(slabs).items()) + ")")
        print(tab.to_string())
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
