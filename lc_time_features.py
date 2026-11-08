#!/usr/bin/env python3
"""How much TIME each light curve, and each encoder window, covers.

A light curve's number of points says nothing direct about the time it spans: ZTF
cadence varies by field, and the encoder never sees a whole curve, only windows of
window_size consecutive points. This script reads the TFRecord shards that
agn_to_tfrecord.py wrote and, for every lcid, records

  n_points        number of observations
  span_days       mjd_max - mjd_min of the whole curve
  dt_median       median gap between consecutive observations (days)
  wspan_<W>       days covered by one window of W points (last mjd - first mjd),
                  averaged over the windows of that curve
  nwin_<W>        how many windows the curve gives at that window size

The windows are cut exactly as extract_embeddings.py cuts them (get_windows in
src/data/preprocessing.py with sampling=False): starts at 0, W, 2W, ... up to
floor(n/W)*W, each start clamped to n-W so that every window has W points (the last
one overlaps the one before it; when n is a multiple of W it repeats it), and a
curve shorter than W is one window. The embedding of a curve is the plain mean over
those windows, so wspan_<W> is the plain mean too.

Needs TensorFlow only to read the records (the root uv env: source activate_env.sh). Writes one row
per lcid; join to lc_index.csv on lcid. length_mass_gain.py --length wspan reads it.

    python lc_time_features.py --data data/records/agn_test/fold_0/test \\
        --out lc_time_features.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


def window_spans(t, W):
    """Days covered by each window of W points, cut as get_windows does."""
    n = len(t)
    if n == 0:
        return np.zeros(0)
    k = np.arange(n // W + 1) * W
    start = np.maximum(0, np.minimum(k, n - W))
    end = start + min(n, W) - 1
    return t[end] - t[start]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="directory of .record shards (e.g. .../fold_0/test)")
    ap.add_argument("--windows", type=int, nargs="+", default=[200, 400, 600, 800, 1000])
    ap.add_argument("--out", default="lc_time_features.csv")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.data, "*.record")))
    if not shards:
        print(f"no .record files in {a.data}", file=sys.stderr)
        return 1
    rows = []
    t0 = time.time()
    for i, path in enumerate(shards):
        for raw in tf.data.TFRecordDataset(path).as_numpy_iterator():
            ex = tf.train.SequenceExample.FromString(raw)
            lcid = ex.context.feature["ID"].bytes_list.value[0].decode()
            t = np.sort(np.asarray(
                ex.feature_lists.feature_list["mjd"].feature[0].float_list.value, dtype=np.float64))
            r = {"lcid": lcid, "n_points": len(t),
                 "span_days": float(t[-1] - t[0]) if len(t) else np.nan,
                 "dt_median": float(np.median(np.diff(t))) if len(t) > 1 else np.nan}
            for W in a.windows:
                s = window_spans(t, W)
                r[f"wspan_{W}"] = float(s.mean()) if len(s) else np.nan
                r[f"nwin_{W}"] = len(s)
            rows.append(r)
        print(f"[{i + 1}/{len(shards)}] {os.path.basename(path)}  {len(rows):,} curves  "
              f"{time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df):,} light curves)")
    print(df.drop(columns="lcid").describe(percentiles=[.05, .25, .5, .75, .95]).T.round(1)
          .to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
