#!/usr/bin/env python3
"""Synthetic AGN light curves with a known mass -> timescale law, for testing whether
a longer window lets the frozen encoder recover heavier black-hole masses.

HYPOTHESIS UNDER TEST
  Heavier black holes vary on longer timescales. A window of W observations spans a
  limited stretch of time, and once the damped-random-walk timescale tau is a large
  fraction of that span the window only sees the rising (random-walk) part of the
  structure function, so masses above some threshold become indistinguishable. A
  longer window should push that threshold up. Here mass enters the light curve ONLY
  through tau (by default), so any mass signal the encoder finds is timescale signal,
  and the mass at which regression error blows up can be compared across windows.

GENERATIVE MODEL (one band per source)
  log M          uniform on [--mass-min, --mass-max]    (--mass-dist uniform)
                 or drawn from the real catalogue          (--mass-dist catalog)
  z              --z-mode fixed:       every source at --z
                 --z-mode independent: catalogue redshift, independent of mass
                 --z-mode joint:       catalogue (LOGMBH, Z_FIT) pairs, so the real
                                       mass-redshift correlation is kept
  tau_rest       --tau0 * (M / 1e8 Msun) ** --tau-slope * 10**N(0, --tau-scatter)
                 defaults 107 d and 0.38 (Burke et al. 2021, Science 373, 789)
  tau_obs        tau_rest * (1 + z)            (time dilation; times are observed)
  SF_inf         --sf-median * 10**N(0, --sf-scatter) * (M / 1e8) ** --sf-mass-slope
                 slope 0 by default, so amplitude carries no mass information.
                 MacLeod et al. (2010) find ~0.18 at fixed luminosity; set it to
                 test a second mass channel.
  x(t)           exact Ornstein-Uhlenbeck (DRW) recursion on the cadence's times,
                 stationary start, variance SF_inf^2 / 2
  mag            m0 + x(t) + N(0, err)

  The defaults are near what plot_sf_by_mass.py measures on the real r band
  (cluster_results/sf_by_mass.csv: SF_inf ~0.2-0.27 mag, rest tau ~150-230 d at
  log M 7.8-8.6), and the real curves only resolve tau below log M ~8.8 there.

CADENCE AND NOISE
  --cadence-from DIR  (recommended) take (mjd, err, median mag) from real light curves
                      in a records directory holding *.record and config.toml, e.g.
                      data/records/agn_test/fold_0/test. Each synthetic curve gets a
                      random real curve's timestamps, per-point errors and median mag,
                      so seasons, gaps, lengths and the noise model are ZTF's, and none
                      of it is related to the synthetic mass.
  --cadence sim       a ZTF-like generator (~6 yr, yearly seasons, ~2-3 d spacing,
                      150-500 points) for when no real records are at hand.

OUTPUT (under --out; each part is skipped when its count is 0)
  train/, validation/          pretraining set (--n-train, --n-val), loaded by
                               pretrain.py --data <out>
  fold_0/test/                 labelled evaluation set (--n-eval), for
                               extract_embeddings.py --data <out>/fold_0 --subset test
  each has *.record shards, config.toml and lc_index.csv. lc_index.csv uses the
  agn_to_tfrecord.py columns that fit_regressor.py reads (LOGMBH, Z_FIT, mbh_valid,
  the photometric stats), plus the truth: tau_rest, tau_obs, sf_inf, m0, cadence_lcid.
  LOGEDD is NaN (no luminosity is simulated): use fit_regressor.py --targets m (add z
  only when --z-mode is not fixed).
  diagnostics.csv              per mass bin of the eval set (else the training set):
                               median tau_obs and, for each --diag-windows W, the
                               median observed-frame span of a W-point window and the
                               fraction of curves whose span is >= 3 tau_obs.

  The parts use different random streams, so no curve appears in two of them. Each
  shard is written to a temporary file and renamed when complete.

    # pretraining + evaluation sets on real cadences
    python synth_drw_records.py --cadence-from data/records/agn_test/fold_0/test \\
        --out data/records/synth_drw --n-train 200000 --n-val 20000 --n-eval 40000

    python -m presentation.pipelines.pipeline_0.pretrain --data data/records/synth_drw ...
    python -m extract_embeddings --pt-model <...>/pretraining \\
        --data data/records/synth_drw/fold_0 --subset test --out emb_synth.npz
    python fit_regressor.py --emb emb_synth.npz \\
        --index data/records/synth_drw/fold_0/test/lc_index.csv --targets m ...

Needs TensorFlow (the root uv env: source activate_env.sh).
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import toml  # noqa: E402

BAND, BAND_NAME = 2, "r"
LOG_M_PIVOT = 8.0


# --- records ----------------------------------------------------------------

def _f(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def sequence_example(lcid, mjd, mag, err):
    context = tf.train.Features(feature={
        "ID": tf.train.Feature(bytes_list=tf.train.BytesList(value=[lcid.encode()])),
        "Label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
    })
    lists = tf.train.FeatureLists(feature_list={
        k: tf.train.FeatureList(feature=[_f(v.astype(np.float32).tolist())])
        for k, v in (("mjd", mjd), ("mag", mag), ("err", err))
    })
    return tf.train.SequenceExample(context=context, feature_lists=lists)


def write_config(directory):
    cfg = {
        "id_column": {"value": "id", "dtype": "string"},
        "target": {"value": directory, "dtype": "string"},
        "context_features": {"value": ["ID", "Label"], "dtypes": ["string", "integer"]},
        "sequential_features": {"value": ["mjd", "mag", "err"],
                                "dtypes": ["float", "float", "float"]},
    }
    with open(os.path.join(directory, "config.toml"), "w") as fh:
        toml.dump(cfg, fh)


# --- cadences ---------------------------------------------------------------

def load_real_cadences(directory, pool, min_points, seed):
    """Up to `pool` (lcid, mjd, err, median mag) tuples from real records."""
    shards = sorted(glob.glob(os.path.join(directory, "*.record")))
    if not shards:
        raise SystemExit(f"no *.record files in {directory}")
    np.random.default_rng(seed).shuffle(shards)
    spec_ctx = {"ID": tf.io.FixedLenFeature([], tf.string)}
    spec_seq = {k: tf.io.VarLenFeature(tf.float32) for k in ("mjd", "mag", "err")}
    out = []
    for path in shards:
        for raw in tf.data.TFRecordDataset(path):
            ctx, seq = tf.io.parse_single_sequence_example(raw, spec_ctx, spec_seq)
            mjd, mag, err = (tf.sparse.to_dense(seq[k]).numpy().ravel().astype(np.float64)
                             for k in ("mjd", "mag", "err"))
            ok = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
            if ok.sum() < min_points:
                continue
            o = np.argsort(mjd[ok], kind="stable")
            out.append((ctx["ID"].numpy().decode(), mjd[ok][o], err[ok][o],
                        float(np.median(mag[ok]))))
            if len(out) >= pool:
                return out
    return out


def sim_cadence(rng):
    """ZTF-like: ~6 yr, one ~6-8 month season per year, visits every ~2-3 nights."""
    t0, years = 58200.0, 6
    nights = []
    phase = rng.uniform(0, 365.25)
    for y in range(years):
        start = t0 + 365.25 * y + phase + rng.normal(0, 10)
        length = rng.uniform(180, 260)
        n = np.arange(int(length))
        p = rng.uniform(0.25, 0.55)                       # visit probability per night
        keep = rng.random(n.size) < p
        weather = rng.random(n.size) < 0.85               # nights not lost to weather
        nights.append(start + n[keep & weather] + rng.uniform(-0.1, 0.1, (keep & weather).sum()))
    mjd = np.sort(np.concatenate(nights))
    m0 = rng.normal(19.6, 0.6)
    err_med = 0.02 + 0.06 * np.exp((m0 - 20.0) / 0.8)    # fainter -> noisier
    err = err_med * 10 ** rng.normal(0, 0.08, mjd.size)
    return "sim", mjd, np.clip(err, 0.005, None), m0


# --- physics ----------------------------------------------------------------

def draw_params(n, rng, args, catalog):
    if args.mass_dist == "uniform":
        logm = rng.uniform(args.mass_min, args.mass_max, n)
    else:
        sel = catalog[(catalog.LOGMBH >= args.mass_min) & (catalog.LOGMBH <= args.mass_max)]
        rows = sel.iloc[rng.integers(0, len(sel), n)]
        logm = rows.LOGMBH.to_numpy()

    if args.z_mode == "fixed":
        z = np.full(n, args.z)
    elif args.z_mode == "independent":
        z = catalog.Z_FIT.to_numpy()[rng.integers(0, len(catalog), n)]
    else:   # joint: the catalogue redshift of a source with (nearly) the same mass
        cm = catalog.sort_values("LOGMBH")
        pos = np.clip(np.searchsorted(cm.LOGMBH.to_numpy(), logm), 0, len(cm) - 1)
        jitter = rng.integers(-50, 51, n)                 # neighbours in mass order
        z = cm.Z_FIT.to_numpy()[np.clip(pos + jitter, 0, len(cm) - 1)]

    tau_rest = args.tau0 * 10 ** (args.tau_slope * (logm - LOG_M_PIVOT)
                                  + rng.normal(0, args.tau_scatter, n))
    sf_inf = args.sf_median * 10 ** (rng.normal(0, args.sf_scatter, n)
                                     + args.sf_mass_slope * (logm - LOG_M_PIVOT))
    return logm, z, tau_rest, tau_rest * (1 + z), sf_inf


def simulate_drw(times, tau, sf_inf, rng):
    """Exact DRW on each curve's times. times: list of 1-D arrays; returns list."""
    n = len(times)
    lengths = np.array([t.size for t in times])
    L = lengths.max()
    T = np.zeros((n, L))
    for i, t in enumerate(times):
        T[i, :t.size] = t
        T[i, t.size:] = t[-1]                             # dt = 0 past the end
    sigma = sf_inf / np.sqrt(2.0)
    x = np.empty((n, L))
    x[:, 0] = sigma * rng.standard_normal(n)
    for j in range(1, L):
        a = np.exp(-(T[:, j] - T[:, j - 1]) / tau)
        x[:, j] = a * x[:, j - 1] + sigma * np.sqrt(1 - a * a) * rng.standard_normal(n)
    return [x[i, :lengths[i]] for i in range(n)]


# --- writing ----------------------------------------------------------------

def make_part(name, n, subdir, args, cadences, catalog, seed):
    rng = np.random.default_rng(seed)
    out_dir = os.path.join(args.out, subdir)
    os.makedirs(out_dir, exist_ok=True)
    write_config(out_dir)
    rows, t_start, shard, done = [], time.time(), 0, 0
    while done < n:
        k = min(args.shard_size, n - done)
        path = os.path.join(out_dir, f"{shard:04d}.record")
        logm, z, tau_rest, tau_obs, sf_inf = draw_params(k, rng, args, catalog)
        if cadences is not None:
            picks = [cadences[i] for i in rng.integers(0, len(cadences), k)]
        else:
            picks = [sim_cadence(rng) for _ in range(k)]
        xs = []
        for b in range(0, k, args.sim_batch):
            sl = slice(b, b + args.sim_batch)
            xs += simulate_drw([p[1] for p in picks[sl]], tau_obs[sl], sf_inf[sl], rng)
        tmp = path + ".tmp"
        with tf.io.TFRecordWriter(tmp) as writer:
            for i, ((clcid, mjd, err, m0), x) in enumerate(zip(picks, xs)):
                err = err * args.noise_scale
                mag = m0 + x + err * rng.standard_normal(err.size)
                sid = f"syn{seed}{done + i:08d}"
                lcid = f"{sid}_b{BAND}"
                writer.write(sequence_example(lcid, mjd, mag, err).SerializeToString())
                rows.append(dict(
                    lcid=lcid, source_id=sid, band=BAND, band_name=BAND_NAME,
                    n_points=mjd.size, mjd_min=mjd[0], mjd_max=mjd[-1],
                    mag_mean=mag.mean(), mag_std=mag.std(), err_med=np.median(err),
                    mag_skew=skew(mag), mag_kurt=kurtosis(mag),
                    Z_FIT=z[i], LOGMBH=logm[i], LOGMBH_ERR=0.0,
                    LOGLBOL=np.nan, LOGEDD=np.nan,
                    mbh_valid=True, lbol_valid=False, z_valid=True, label_valid=True,
                    lbol_lowz_caveat=False, in_pretrain=False, in_pretrain_band=False,
                    tau_rest=tau_rest[i], tau_obs=tau_obs[i], sf_inf=sf_inf[i],
                    m0=m0, cadence_lcid=clcid))
        os.replace(tmp, path)
        done += k
        shard += 1
        print(f"[{name}] {done}/{n} curves, {shard} shards, "
              f"{time.time() - t_start:.0f}s", flush=True)
    index = pd.DataFrame(rows)
    index.to_csv(os.path.join(out_dir, "lc_index.csv"), index=False)
    return index, cadences


def window_span(mjd, w):
    """Median observed-frame span of the consecutive w-point windows of one curve."""
    if mjd.size <= w:
        return mjd[-1] - mjd[0]
    starts = np.arange(0, mjd.size - w + 1, w)
    return float(np.median(mjd[starts + w - 1] - mjd[starts]))


def diagnostics(index, cadences, args):
    lookup = {c[0]: c[1] for c in cadences} if cadences is not None else None
    edges = np.arange(args.mass_min, args.mass_max + 0.25, 0.5)
    idx = index.copy()
    idx["mbin"] = pd.cut(idx.LOGMBH, edges)
    out = []
    for b, g in idx.groupby("mbin", observed=True):
        row = {"mass_bin": str(b), "n": len(g), "tau_obs_med": g.tau_obs.median()}
        if lookup is not None:
            spans = {w: np.array([window_span(lookup[c], w) for c in g.cadence_lcid])
                     for w in args.diag_windows}
            for w, s in spans.items():
                row[f"span_{w}"] = np.median(s)
                row[f"frac_span_ge_3tau_{w}"] = np.mean(s >= 3 * g.tau_obs.to_numpy())
        out.append(row)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=200_000)
    p.add_argument("--n-val", type=int, default=20_000)
    p.add_argument("--n-eval", type=int, default=40_000)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cadence-from", help="records dir with *.record + config.toml")
    g.add_argument("--cadence", choices=["sim"])
    p.add_argument("--cadence-pool", type=int, default=50_000,
                   help="real curves to read cadences from")
    p.add_argument("--min-points", type=int, default=50)
    p.add_argument("--catalog", default="qzo_labels.csv",
                   help="for --mass-dist catalog and --z-mode independent/joint")
    p.add_argument("--mass-dist", choices=["uniform", "catalog"], default="uniform")
    p.add_argument("--mass-min", type=float, default=7.0)
    p.add_argument("--mass-max", type=float, default=10.5)
    p.add_argument("--z-mode", choices=["fixed", "independent", "joint"], default="fixed")
    p.add_argument("--z", type=float, default=1.5)
    p.add_argument("--tau0", type=float, default=107.0, help="rest days at 1e8 Msun")
    p.add_argument("--tau-slope", type=float, default=0.38)
    p.add_argument("--tau-scatter", type=float, default=0.1, help="dex at fixed mass")
    p.add_argument("--sf-median", type=float, default=0.25, help="mag")
    p.add_argument("--sf-scatter", type=float, default=0.15, help="dex")
    p.add_argument("--sf-mass-slope", type=float, default=0.0)
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="multiply the per-point errors (0 = noiseless)")
    p.add_argument("--diag-windows", type=int, nargs="+", default=[200, 400, 800])
    p.add_argument("--shard-size", type=int, default=20_000)
    p.add_argument("--sim-batch", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    catalog = None
    if args.mass_dist == "catalog" or args.z_mode != "fixed":
        catalog = pd.read_csv(args.catalog, dtype={"id": str})
        catalog = catalog[(catalog.LOGMBH > 6) & (catalog.LOGMBH < 11)
                          & (catalog.LOGMBH_ERR > 0) & (catalog.Z_FIT > 0)]
    if args.noise_scale == 0:
        args.noise_scale = 1e-6      # the loader's error weighting needs err > 0

    cadences = None
    if args.cadence_from:
        t = time.time()
        cadences = load_real_cadences(args.cadence_from, args.cadence_pool,
                                      args.min_points, args.seed)
        lens = np.array([c[1].size for c in cadences])
        print(f"[cadence] {len(cadences)} real curves in {time.time() - t:.0f}s, "
              f"points median {np.median(lens):.0f} (5-95%: "
              f"{np.percentile(lens, 5):.0f}-{np.percentile(lens, 95):.0f})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "generator_args.toml"), "w") as fh:
        toml.dump(vars(args), fh)

    last = None
    for name, n, sub, off in (("train", args.n_train, "train", 1),
                              ("validation", args.n_val, "validation", 2),
                              ("eval", args.n_eval, os.path.join("fold_0", "test"), 3)):
        if n > 0:
            last, _ = make_part(name, n, sub, args, cadences, catalog,
                                seed=args.seed * 10 + off)
    if last is not None:
        diag = diagnostics(last, cadences, args)
        diag.to_csv(os.path.join(args.out, "diagnostics.csv"), index=False)
        with pd.option_context("display.width", 200, "display.precision", 2):
            print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
