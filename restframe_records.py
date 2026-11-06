#!/usr/bin/env python3
"""Write a rest-frame copy of an AGN TFRecord set: every light curve's time axis is
divided by (1 + z), undoing cosmological time dilation.

    t_rest = (mjd - mjd_first) / (1 + Z_FIT)

mjd_first is the light curve's first observation, so the 'mjd' feature in the new
records holds rest-frame DAYS SINCE THE FIRST OBSERVATION, not an MJD. Nothing reads
absolute times: the loader zero-means each window's time column (standardize in
src/data/preprocessing.py), so only the spacing between observations reaches the
encoder, and subtracting the first time also keeps full float32 precision.
Magnitudes and errors are copied unchanged, and so is everything else in each
SequenceExample (ID, Label), so lcids and the number of points per curve, and hence
the windows, are identical to the observed-frame set.

Only the time axis is corrected. The bands are not: at redshift z an observed band
samples rest-frame wavelengths 1/(1 + z) times shorter.

REDSHIFT AS A TARGET
  Almost every ZTF curve spans the same ~6 yr, so after dividing by (1 + z) a
  curve's time span is close to a direct readout of z (Spearman ~ -0.99 between
  rest-frame span and z). A regressor on rest-frame embeddings can therefore read z
  off the time axis: run fit_regressor.py with --targets m,e on these records, not z.
  Mass correlates with z (Spearman ~0.4), so some of that leaks into m and e too.

Light curves whose source has no usable redshift (z_valid False, or Z_FIT missing
or <= 0) are dropped and counted.

Layout: --src is a records root such as data/records/agn_test, holding config.toml
and fold_*/<subset>/{*.record, lc_index.csv, config.toml}. --dst gets the same tree.
Each lc_index.csv is copied with the dropped rows removed and three columns added:
  rest_factor   1 + Z_FIT
  span_rest     (mjd_max - mjd_min) / rest_factor   rest-frame days
  frame         "rest"
mjd_min and mjd_max stay in the observed frame.

Needs TensorFlow (the astromer conda env). Each shard is written to a temporary file
and renamed when complete, so an interrupted run never leaves a truncated shard;
rerunning skips shards that already exist.

    python restframe_records.py --src data/records/agn_test --dst data/records/agn_test_rest
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


def rest_factors(index_path):
    """lcid -> 1 + z for light curves with a usable redshift, and the index itself."""
    idx = pd.read_csv(index_path, dtype={"lcid": str, "source_id": str})
    z = pd.to_numeric(idx["Z_FIT"], errors="coerce")
    ok = z.notna() & (z > 0)
    if "z_valid" in idx:
        ok &= idx["z_valid"].astype(bool)
    idx = idx.assign(rest_factor=np.where(ok, 1.0 + z, np.nan))
    return idx, ok.to_numpy()


def convert_shard(src, dst, factor):
    """Rewrite one shard; returns (written, dropped)."""
    written = dropped = 0
    tmp = dst + ".tmp"
    with tf.io.TFRecordWriter(tmp) as w:
        for raw in tf.data.TFRecordDataset(src).as_numpy_iterator():
            ex = tf.train.SequenceExample.FromString(raw)
            lcid = ex.context.feature["ID"].bytes_list.value[0].decode()
            f = factor.get(lcid)
            if f is None:
                dropped += 1
                continue
            fl = ex.feature_lists.feature_list["mjd"].feature[0].float_list
            t = np.asarray(fl.value, dtype=np.float64)
            t_rest = (t - t.min()) / f
            del fl.value[:]
            fl.value.extend(t_rest.astype(np.float32).tolist())
            w.write(ex.SerializeToString())
            written += 1
    os.replace(tmp, dst)
    return written, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="observed-frame records root")
    ap.add_argument("--dst", required=True, help="rest-frame records root to write")
    a = ap.parse_args()
    if os.path.abspath(a.src) == os.path.abspath(a.dst):
        print("--dst must differ from --src", file=sys.stderr)
        return 1

    subsets = sorted(os.path.dirname(p) for p in
                     glob.glob(os.path.join(a.src, "**", "lc_index.csv"), recursive=True))
    if not subsets:
        print(f"no lc_index.csv under {a.src}", file=sys.stderr)
        return 1
    # config.toml at every level, so either the root or a fold works as --data
    for cfg in glob.glob(os.path.join(a.src, "**", "config.toml"), recursive=True):
        out = os.path.join(a.dst, os.path.relpath(cfg, a.src))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(cfg, out)

    t0 = time.time()
    for sub in subsets:
        rel = os.path.relpath(sub, a.src)
        out_dir = os.path.join(a.dst, rel)
        os.makedirs(out_dir, exist_ok=True)
        idx, ok = rest_factors(os.path.join(sub, "lc_index.csv"))
        factor = dict(zip(idx.loc[ok, "lcid"], idx.loc[ok, "rest_factor"]))
        print(f"{rel}: {ok.sum():,} of {len(idx):,} light curves have a usable redshift",
              flush=True)

        shards = sorted(glob.glob(os.path.join(sub, "*.record")))
        written = dropped = 0
        for i, src in enumerate(shards):
            dst = os.path.join(out_dir, os.path.basename(src))
            if os.path.exists(dst):
                continue
            w, d = convert_shard(src, dst, factor)
            written, dropped = written + w, dropped + d
            if (i + 1) % 10 == 0 or i + 1 == len(shards):
                print(f"  [{i + 1}/{len(shards)}] {written:,} written, {dropped:,} dropped  "
                      f"{time.time() - t0:.0f}s", flush=True)

        keep = idx[ok].copy()
        keep["span_rest"] = (keep["mjd_max"] - keep["mjd_min"]) / keep["rest_factor"]
        keep["frame"] = "rest"
        tmp = os.path.join(out_dir, "lc_index.csv.tmp")
        keep.to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(out_dir, "lc_index.csv"))
        print(f"  wrote {os.path.join(out_dir, 'lc_index.csv')} ({len(keep):,} rows)")
    print(f"done in {time.time() - t0:.0f}s -> {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
