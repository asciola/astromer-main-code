#!/usr/bin/env python3
"""Cut one window of --points consecutive observations per light curve whose
REST-FRAME duration is close to each of several targets, keeping only the light
curves that can do so for EVERY target (a paired design: the same sources at every
duration, so a trend with duration cannot come from different sources).

Reads a rest-frame record set from restframe_records.py, where the 'mjd' feature is
rest-frame days since the first observation. For a light curve and a target D days,
every run of --points consecutive observations is a candidate; its duration is
last time - first time. The candidate closest to D is chosen, and the light curve
qualifies at D if that duration is within --tol (fraction) of D. Only light curves
that qualify at every --durations value are kept. Different durations of the same
curve may share observations.

Why paired, and why these durations: with 200 points a window spans a median ~356
rest days (5-95%: ~190-690), and a single curve's 200-point windows only span about
a factor of two, so few curves reach widely separated durations. ~6.7k r-band curves
reach 200, 300 and 400 days; 100 days is out of reach for ~97% of them. Rest-frame
duration also falls with redshift, which is why the same sources must be used at
every duration.

Writes, for each target D, a record set <dst>/D<D>/fold_0/test/ holding exactly one
window per kept light curve (same lcid as the source curve, so labels join as
before; times re-zeroed to the window start), plus config.toml at each level and an
lc_index.csv: the source rows, restricted to the kept curves, with the photometric
statistics (mag_mean, mag_std, err_med, mag_skew, mag_kurt, n_points) RECOMPUTED on
the window, so the summary-statistics control describes the same observations as the
embedding, and three added columns:
  target_days    D
  window_start   index of the window's first observation in the full curve
  window_days    the window's rest-frame duration
Extract embeddings from each D set with window-200 models (one window per curve) and
fit each set with fit_regressor.py --targets m,e; the KFold assignment depends only
on the sources, which are identical across D, so the folds are paired too.

CELLS: WINDOW SIZE AND DENSITY AS WELL AS DURATION
  --cells replaces --durations/--points with a list of cells POINTS@DAYS, or
  POINTS@DAYS/THIN: the contiguous POINTS-point window whose duration is closest to
  DAYS (as above), then, with /THIN, every THIN-th observation of that same window
  (POINTS/THIN points over almost the same span: half the density for /2). A light
  curve is kept only if it qualifies for every contiguous POINTS@DAYS the cells
  name, so all cells hold the same sources again, and a thinned cell is a strict
  subset of the observations of its contiguous one. Each cell is written to
  <dst>/p<POINTS>_d<DAYS>[_t<THIN>]/fold_0/test/, and its lc_index.csv adds
  points_cut (POINTS) and thin (THIN, 1 if none); n_points is the number actually
  kept. Within one curve the duration of a P-point window varies only about
  twofold, so cells that need both a dense and a sparse contiguous stretch keep few
  curves; thinning reaches low densities without that cost. For example

      --cells 200@300 400@600 400@600/2

  gives 200 points over 300 days, 400 points over 600 days (the same density, twice
  the time) and those 400 thinned to 200 (the same time, half the density): ~50k
  r-band curves with a mass, against ~120 for a contiguous 200@600 as well.

Needs TensorFlow (the root uv env: source activate_env.sh).

    python restframe_windows.py --src data/records/agn_test_rest/fold_0/test \\
        --dst data/records/agn_rest_windows --durations 200 300 400
    python restframe_windows.py --src data/records/agn_test_rest/fold_0/test \\
        --dst data/records/agn_rest_cells --cells 200@300 400@600 400@600/2
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from scipy import stats  # noqa: E402


def select(subset_dir, cand_lcids, needs, tol):
    """Stream the records; keep only light curves that qualify for every need.

    needs is a list of (points, days). Returns lcid -> (SequenceExample, times, mags,
    errs) and lcid -> {(points, days): (start, days covered)}.
    """
    curves, choice = {}, {}
    for p in sorted(glob.glob(os.path.join(subset_dir, "*.record"))):
        for raw in tf.data.TFRecordDataset(p).as_numpy_iterator():
            ex = tf.train.SequenceExample.FromString(raw)
            lcid = ex.context.feature["ID"].bytes_list.value[0].decode()
            if lcid not in cand_lcids:
                continue
            fl = ex.feature_lists.feature_list
            t = np.asarray(fl["mjd"].feature[0].float_list.value, dtype=np.float64)
            row = {}
            for n, D in needs:
                i, span = best_window(t, n, D)
                if i is None or abs(span - D) > tol * D:
                    break
                row[(n, D)] = (i, span)
            else:
                choice[lcid] = row
                curves[lcid] = (ex, t, *(np.asarray(fl[k].feature[0].float_list.value,
                                                    dtype=np.float64) for k in ("mag", "err")))
    return curves, choice


def best_window(t, n, D):
    """Start index and duration of the n-point window whose duration is closest to D."""
    if len(t) < n:
        return None, np.nan
    spans = t[n - 1:] - t[:len(t) - n + 1]
    i = int(np.argmin(np.abs(spans - D)))
    return i, float(spans[i])


def window_example(ex, t, mag, err, sel):
    """Copy of ex holding the observations at index array sel, times re-zeroed to the
    first of them."""
    w = tf.train.SequenceExample()
    w.CopyFrom(ex)
    fl = w.feature_lists.feature_list
    for k, v in (("mjd", t[sel] - t[sel[0]]), ("mag", mag[sel]), ("err", err[sel])):
        f = fl[k].feature[0].float_list
        del f.value[:]
        f.value.extend(v.astype(np.float32).tolist())
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="rest-frame subset dir with *.record and lc_index.csv "
                         "(e.g. data/records/agn_test_rest/fold_0/test)")
    ap.add_argument("--dst", required=True,
                    help="output root; gets D<days>/fold_0/test/ (or one dir per --cells)")
    ap.add_argument("--durations", type=float, nargs="+", default=[200, 300, 400],
                    help="target rest-frame window durations in days")
    ap.add_argument("--points", type=int, default=200, help="observations per window")
    ap.add_argument("--cells", nargs="+", metavar="POINTS@DAYS[/THIN]",
                    help="instead of --durations/--points: one window set per cell, "
                         "e.g. 200@300 400@600 400@600/2 (see CELLS above)")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="largest allowed |duration - D| / D (default 0.10)")
    ap.add_argument("--bands", nargs="+", default=["r"],
                    help="bands to use (default r); each band of a source must qualify "
                         "on its own")
    ap.add_argument("--max-shard", type=int, default=5000, help="windows per output shard")
    a = ap.parse_args()

    # (points, days, thin, output dir name) per window set
    if a.cells:
        cells = []
        for c in a.cells:
            try:
                pd_, _, thin = c.partition("/")
                n, D = pd_.split("@")
                n, D, thin = int(n), float(D), int(thin or 1)
            except ValueError:
                print(f"bad cell {c!r}: expected POINTS@DAYS or POINTS@DAYS/THIN",
                      file=sys.stderr)
                return 1
            if thin < 1 or n % thin:
                print(f"bad cell {c!r}: THIN must divide POINTS", file=sys.stderr)
                return 1
            cells.append((n, D, thin, f"p{n}_d{D:g}" + (f"_t{thin}" if thin > 1 else "")))
    else:
        cells = [(a.points, D, 1, f"D{int(D)}") for D in a.durations]
    needs = sorted({(n, D) for n, D, _, _ in cells})
    nmax = max(n for n, _ in needs)

    idx = pd.read_csv(os.path.join(a.src, "lc_index.csv"), dtype={"lcid": str, "source_id": str})
    if "frame" not in idx or (idx["frame"] != "rest").any():
        print(f"{a.src}/lc_index.csv is not from restframe_records.py (no frame=rest); "
              "run that first", file=sys.stderr)
        return 1
    cand = idx[idx["band_name"].isin(a.bands) & (idx["n_points"] >= nmax)]
    print(f"{len(cand):,} {'/'.join(a.bands)}-band light curves with >= {nmax} points; "
          "reading them", flush=True)
    curves, choice = select(a.src, set(cand["lcid"]), needs, a.tol)
    kept = sorted(choice)
    print(f"{len(kept):,} light curves have a window within {a.tol:.0%} of every "
          + ", ".join(f"{n} points @ {D:g} d" for n, D in needs))
    if not kept:
        return 1

    cfg = os.path.join(a.src, "config.toml")
    for n, D, thin, name in cells:
        root = os.path.join(a.dst, name)
        sub = os.path.join(root, "fold_0", "test")
        os.makedirs(sub, exist_ok=True)
        # the record schema, at every level, so the root, the fold or the subset works
        for where in (root, os.path.dirname(sub), sub):
            shutil.copyfile(cfg, os.path.join(where, "config.toml"))

        rows = []
        for s0 in range(0, len(kept), a.max_shard):
            path = os.path.join(sub, f"{s0 // a.max_shard:04d}.record")
            with tf.io.TFRecordWriter(path + ".tmp") as w:
                for lcid in kept[s0:s0 + a.max_shard]:
                    ex, t, mag, err = curves[lcid]
                    i, _ = choice[lcid][(n, D)]
                    sel = np.arange(i, i + n, thin)
                    w.write(window_example(ex, t, mag, err, sel).SerializeToString())
                    m, e = mag[sel], err[sel]
                    row = {"lcid": lcid, "target_days": D, "window_start": i,
                           "window_days": t[sel[-1]] - t[sel[0]], "mag_mean": m.mean(),
                           "mag_std": m.std(), "err_med": np.median(e),
                           "mag_skew": stats.skew(m), "mag_kurt": stats.kurtosis(m),
                           "n_points": len(sel)}
                    if a.cells:
                        row.update(points_cut=n, thin=thin)
                    rows.append(row)
            os.replace(path + ".tmp", path)
        win = pd.DataFrame(rows)
        base = idx.drop(columns=[c for c in win.columns if c != "lcid" and c in idx])
        out = base.merge(win, on="lcid", how="inner")
        tmp = os.path.join(sub, "lc_index.csv.tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(sub, "lc_index.csv"))
        q = out["window_days"].quantile([0.05, 0.5, 0.95]).round(0).tolist()
        print(f"{name}: wrote {sub} ({len(out):,} windows of {n // thin} points; "
              f"duration 5/50/95%: {q} d)")

    s = idx[idx["lcid"].isin(kept)].drop_duplicates("source_id")
    allb = idx[idx["band_name"].isin(a.bands)].drop_duplicates("source_id")
    for name, col in (("redshift", "Z_FIT"), ("LOGMBH", "LOGMBH")):
        print(f"  median {name}: kept {s[col].median():.2f}, all {allb[col].median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
