"""
Convert the multi-band AGN/quasar light curves into TFRecord shards matching this
project's SequenceExample schema, for use as a held-out *evaluation* set
(reconstruction metrics and embedding regressions).

INPUT
-----
  light curves  ->  <sourceid>.csv, each with columns
                        mjd, mag, magerr, filter      (filter in {1,2,3} = g,r,i)
                    either an extracted directory (--lc-dir) or a .tgz (--tgz)
  labels        ->  qzo_labels.csv from crossmatch_labels.py (or comb.csv):
                        id, RA, DEC, Z_FIT, Z_SYS_ERR, LOGLBOL, LOGLBOL_ERR,
                        LOGMBH, LOGMBH_ERR
  overlap       ->  pretrain_overlap.csv from survivor_overlap.py (optional):
                        id, in_pretrain, in_pretrain_g, in_pretrain_r, ...

OUTPUT
------
  <output>/config.toml
  <output>/fold_0/<subset>/config.toml
  <output>/fold_0/<subset>/NNNN.record
  <output>/fold_0/<subset>/lc_index.csv     (lcid -> source params and flags)
  <output>/fold_0/<subset>/read_errors.txt  (only if some files could not be read)

Each (source, band) pair becomes one light curve, mirroring how ztf_to_tfrecord.py
treated ZTF bands. lcid is "<sourceid>_b<band>".

WHICH SOURCES ARE WRITTEN
-------------------------
  --labeled-only   only sources with a row in --meta (171k of 1.74M for qzo).
  --overlap FILE   drop every source that was in the pretraining set, in ANY band.
                   Dropping only the band that was used would still leak: the other
                   band traces the same quasar's variability. --keep-overlap keeps
                   them instead and records in_pretrain in lc_index.csv.
  The file list is filtered by ID before any light curve is opened, so these also
  make the run proportionally faster.

LABEL FLAGS (in lc_index.csv)
-----------------------------
  mbh_valid     6 < LOGMBH < 11 and LOGMBH_ERR > 0   (LOGMBH = 0 / ERR = -1 is the
                catalog's "not measured" sentinel)
  lbol_valid    43 < LOGLBOL < 48.5                    (0 = not measured)
  z_valid       Z_FIT > 0
  label_valid   = mbh_valid (unchanged meaning, for existing downstream code)
  LOGEDD        LOGLBOL - 38.1 - LOGMBH, NaN unless mbh_valid and lbol_valid
  lbol_lowz_caveat  Z_FIT < 0.7: Wu & Shen note (May 2025) that LOGLBOL for these was
                computed from L5100 rather than L3000; treat LOGLBOL / LOGEDD there
                with care, or recompute from LOGL3000.

IMPORTANT: the kurtosis/skewness/std "white noise" filter used when building the
ZTF *training* set is deliberately NOT applied here. That filter selects strongly
variable sources out of a large unlabeled pool; AGN variability is stochastic and
broadly low-kurtosis, so applying it would discard most of this set and bias what
remains. Only a minimum-length cut is applied.

Usage:
    python agn_to_tfrecord.py \\
        --lc-dir  /path/to/qzo \\
        --meta    qzo_labels.csv \\
        --overlap pretrain_overlap.csv \\
        --labeled-only \\
        --output  ./data/records/agn_test \\
        --subset  test
"""
import argparse
import glob
import logging
import os
import shutil
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis, skew as _skew
import tensorflow as tf
import toml

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

BAND_NAMES = {1: 'g', 2: 'r', 3: 'i'}

# Physically plausible ranges. Values outside are missing-data sentinels (notably
# exactly 0.0, with the matching _ERR = -1.0) or failed fits.
MBH_VALID_RANGE = (6.0, 11.0)      # log10(M_BH / Msun)
LBOL_VALID_RANGE = (43.0, 48.5)    # log10(L_bol / erg s^-1)
LOWZ_LBOL_CAVEAT = 0.7


# --- TFRecord encoding (mirrors src/data/record.py conventions) -------------

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def make_sequence_example(lcid, label, mjd, mag, err):
    context = tf.train.Features(feature={
        'ID':    _bytes_feature([str(lcid).encode()]),
        'Label': _int64_feature([int(label)]),
    })
    feature_lists = tf.train.FeatureLists(feature_list={
        'mjd': tf.train.FeatureList(
            feature=[_float_feature(mjd.astype(np.float32).tolist())]),
        'mag': tf.train.FeatureList(
            feature=[_float_feature(mag.astype(np.float32).tolist())]),
        'err': tf.train.FeatureList(
            feature=[_float_feature(err.astype(np.float32).tolist())]),
    })
    return tf.train.SequenceExample(context=context, feature_lists=feature_lists)


def write_config(output_dir):
    cfg = {
        'id_column': {'value': 'id', 'dtype': 'string'},
        'target':    {'value': output_dir, 'dtype': 'string'},
        'context_features': {
            'value':  ['ID', 'Label'],
            'dtypes': ['string', 'integer'],
        },
        'sequential_features': {
            'value':  ['mjd', 'mag', 'err'],
            'dtypes': ['float', 'float', 'float'],
        },
    }
    with open(os.path.join(output_dir, 'config.toml'), 'w') as f:
        toml.dump(cfg, f)


# --- Input handling ---------------------------------------------------------

def resolve_input(tgz, lc_dir, workdir):
    """Return a directory containing the per-source csv files."""
    if lc_dir:
        return lc_dir
    logging.info(f'Extracting {tgz}')
    with tarfile.open(tgz, 'r:gz') as tar:
        tar.extractall(workdir)
    # the archive contains a single top-level directory
    entries = [os.path.join(workdir, e) for e in os.listdir(workdir)]
    dirs = [e for e in entries if os.path.isdir(e)]
    return dirs[0] if len(dirs) == 1 else workdir


def _hms(seconds):
    seconds = int(round(seconds))
    return f'{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def list_sources(src_dir):
    """{source_id: path} for every <digits>.csv in src_dir (os.scandir: fast at 1M+)."""
    out = {}
    with os.scandir(src_dir) as it:
        for e in it:
            if e.name.endswith('.csv') and e.is_file():
                sid = e.name[:-4]
                if sid.isdigit():
                    out[sid] = e.path
    return out


def load_metadata(meta_path):
    """
    Read the parameter table. `id` MUST be read as a string: these are 18-digit
    integers, which exceed float64's ~15-16 significant digits, so pandas'
    default inference silently corrupts them (and does so as soon as the file has
    a single blank row, as comb.csv does).
    """
    meta = pd.read_csv(meta_path, dtype=str)
    meta = meta.rename(columns={meta.columns[0]: 'id'})
    meta['id'] = meta['id'].str.strip()
    meta = meta[meta['id'].str.fullmatch(r'\d+', na=False)]
    for c in meta.columns:
        if c != 'id':
            meta[c] = pd.to_numeric(meta[c], errors='coerce')
    meta = meta.drop_duplicates('id')
    return meta.set_index('id')


def load_overlap(path):
    ov = pd.read_csv(path, dtype={'id': str}).set_index('id')
    for c in [c for c in ov.columns if c.startswith('in_pretrain')]:
        ov[c] = ov[c].astype(str).str.lower().isin(['true', '1'])
    return ov


def label_fields(m):
    """Label columns + validity flags for one metadata row (a pd.Series)."""
    row = {col: m[col] for col in m.index}
    mbh, mbh_err = m.get('LOGMBH', np.nan), m.get('LOGMBH_ERR', np.nan)
    lbol, z = m.get('LOGLBOL', np.nan), m.get('Z_FIT', np.nan)
    mbh_ok = bool(np.isfinite(mbh) and MBH_VALID_RANGE[0] < mbh < MBH_VALID_RANGE[1]
                  and not (np.isfinite(mbh_err) and mbh_err <= 0))
    lbol_ok = bool(np.isfinite(lbol) and LBOL_VALID_RANGE[0] < lbol < LBOL_VALID_RANGE[1])
    z_ok = bool(np.isfinite(z) and z > 0)
    # Derived target: log Eddington ratio.
    #   L_Edd = 1.26e38 * (M/Msun) erg/s  ->  log(L/L_Edd) = LOGLBOL - 38.1 - LOGMBH
    row['LOGEDD'] = (lbol - 38.1 - mbh) if (mbh_ok and lbol_ok) else np.nan
    row['mbh_valid'] = mbh_ok
    row['lbol_valid'] = lbol_ok
    row['z_valid'] = z_ok
    row['label_valid'] = mbh_ok
    row['lbol_lowz_caveat'] = bool(z_ok and z < LOWZ_LBOL_CAVEAT)
    return row


# --- Driver -----------------------------------------------------------------

def convert(tgz, lc_dir, meta_path, output_dir, subset, fold=0,
            shard_size=5000, min_points=5, bands=(1, 2, 3),
            labeled_only=False, overlap_path=None, keep_overlap=False, limit=None,
            progress_every=5000):
    out_root = os.path.join(output_dir, f'fold_{fold}', subset)
    os.makedirs(out_root, exist_ok=True)
    for stale in glob.glob(os.path.join(out_root, '*.record')):
        os.remove(stale)                       # a rerun must not mix with old shards
    write_config(output_dir)
    shutil.copyfile(os.path.join(output_dir, 'config.toml'),
                    os.path.join(out_root, 'config.toml'))
    index_path = os.path.join(out_root, 'lc_index.csv')
    err_path = os.path.join(out_root, 'read_errors.txt')
    for p in (index_path, err_path):
        if os.path.exists(p):
            os.remove(p)

    meta = load_metadata(meta_path) if meta_path else None
    if meta is not None:
        logging.info(f'Loaded metadata for {len(meta):,} sources')
    if labeled_only and meta is None:
        raise SystemExit('--labeled-only needs --meta')
    ov = load_overlap(overlap_path) if overlap_path else None

    with tempfile.TemporaryDirectory() as workdir:
        src_dir = resolve_input(tgz, lc_dir, workdir)
        logging.info(f'Listing light-curve files in {src_dir} ...')
        t_list = time.time()
        sources = list_sources(src_dir)
        if not sources:
            raise FileNotFoundError(f'No per-source csv files under {src_dir}')
        logging.info(f'Found {len(sources):,} source light-curve files '
                     f'(listing took {_hms(time.time() - t_list)})')

        # ---- select sources by ID before opening anything -------------------
        sids = sorted(sources)
        n_unlabeled = n_overlap = 0
        if labeled_only:
            keep = [s for s in sids if s in meta.index]
            n_unlabeled = len(sids) - len(keep)
            sids = keep
        if ov is not None:
            seen = set(ov.index[ov['in_pretrain']])
            not_checked = sum(1 for s in sids if s not in ov.index)
            if not_checked:
                logging.warning(f'{not_checked:,} sources are not in {overlap_path}; '
                                'they are treated as unseen')
            if not keep_overlap:
                keep = [s for s in sids if s not in seen]
                n_overlap = len(sids) - len(keep)
                sids = keep
        if limit:
            sids = sids[:limit]
        logging.info(f'Converting {len(sids):,} sources '
                     f'(skipped {n_unlabeled:,} unlabeled, {n_overlap:,} seen in pretraining)')

        n_written = n_short = n_absent = n_nan = n_err = 0
        t_start = time.time()
        missing_meta = 0
        shard_index = 0
        written_in_shard = 0
        writer = None
        pending_rows = []
        header_written = False
        band_counts = {}

        def flush_index():
            nonlocal pending_rows, header_written
            if pending_rows:
                pd.DataFrame(pending_rows).to_csv(index_path, mode='a', index=False,
                                                  header=not header_written)
                header_written = True
                pending_rows = []

        def open_new_shard():
            nonlocal writer, shard_index, written_in_shard
            if writer is not None:
                writer.close()
            flush_index()
            path = os.path.join(out_root, f'{shard_index:04d}.record')
            writer = tf.io.TFRecordWriter(path)
            written_in_shard = 0
            shard_index += 1

        open_new_shard()
        try:
            for k, sid in enumerate(sids):
                path = sources[sid]
                try:
                    d = pd.read_csv(path)
                    d = d[['mjd', 'mag', 'magerr', 'filter']]
                except Exception as e:                           # noqa: BLE001
                    n_err += 1
                    with open(err_path, 'a') as fh:
                        fh.write(f'{path}\t{type(e).__name__}: {e}\n')
                    continue

                before = len(d)
                d = d.apply(pd.to_numeric, errors='coerce')
                d = d[np.isfinite(d).all(axis=1)]
                n_nan += before - len(d)

                has_meta = meta is not None and sid in meta.index
                if meta is not None and not has_meta:
                    missing_meta += 1
                labels = label_fields(meta.loc[sid]) if has_meta else {}

                for band in bands:
                    g = d[d['filter'] == band]
                    if len(g) == 0:
                        n_absent += 1
                        continue
                    if len(g) < min_points:
                        n_short += 1
                        continue
                    g = g.sort_values('mjd', kind='stable')

                    lcid = f'{sid}_b{band}'
                    ex = make_sequence_example(
                        lcid, band,
                        g['mjd'].to_numpy(),
                        g['mag'].to_numpy(),
                        g['magerr'].to_numpy())
                    writer.write(ex.SerializeToString())
                    n_written += 1
                    written_in_shard += 1
                    band_counts.setdefault(band, []).append(len(g))

                    # Simple photometric summary statistics. These are a
                    # necessary baseline: the pipeline normalises each window to
                    # zero mean, so the encoder never sees mean magnitude, yet
                    # mean magnitude alone is strongly predictive of mass. Keep
                    # them here so the regressor can use them as a control and
                    # as optional extra features (--with-stats).
                    _m = g['mag'].to_numpy()
                    row = {'lcid': lcid, 'source_id': sid, 'band': band,
                           'band_name': BAND_NAMES.get(band, str(band)),
                           'n_points': len(g),
                           'mjd_min': float(g['mjd'].min()),
                           'mjd_max': float(g['mjd'].max()),
                           'mag_mean': float(_m.mean()),
                           'mag_std': float(_m.std()),
                           'err_med': float(g['magerr'].median()),
                           'mag_skew': float(_skew(_m, bias=False)),
                           'mag_kurt': float(_kurtosis(_m, bias=False))}
                    row.update(labels)
                    if ov is not None:
                        in_ov = sid in ov.index
                        row['in_pretrain'] = bool(ov.at[sid, 'in_pretrain']) if in_ov else False
                        bcol = f'in_pretrain_{BAND_NAMES.get(band, band)}'
                        row['in_pretrain_band'] = (bool(ov.at[sid, bcol])
                                                   if in_ov and bcol in ov.columns else False)
                    pending_rows.append(row)

                    if written_in_shard >= shard_size:
                        open_new_shard()
                if (k + 1) % progress_every == 0 or k + 1 == len(sids):
                    el = time.time() - t_start
                    rate = (k + 1) / el if el > 0 else float('inf')
                    eta = (len(sids) - k - 1) / rate if rate > 0 else 0
                    logging.info(f'  {k + 1:,}/{len(sids):,} sources '
                                 f'({100 * (k + 1) / len(sids):.1f}%), '
                                 f'{n_written:,} light curves, {rate:.0f} src/s, '
                                 f'elapsed {_hms(el)}, ETA {_hms(eta)}')
        finally:
            if writer is not None:
                writer.close()
            flush_index()

    # an empty trailing shard is left when the last write filled a shard exactly
    last = os.path.join(out_root, f'{shard_index - 1:04d}.record')
    if written_in_shard == 0 and os.path.exists(last):
        os.remove(last)
        shard_index -= 1

    idx = pd.read_csv(index_path, dtype={'source_id': str}) if header_written else pd.DataFrame()
    logging.info(
        f'Done. Wrote {n_written:,} light curves '
        f'({idx.source_id.nunique() if len(idx) else 0:,} sources) '
        f'across {shard_index} shard(s).')
    logging.info(f'  (source, band) curves dropped for <{min_points} points: {n_short:,}'
                 f'   (band absent: {n_absent:,})')
    logging.info(f'  rows dropped for NaN / non-numeric:  {n_nan:,}')
    if n_err:
        logging.warning(f'  files that could not be read:        {n_err:,} (see {err_path})')
    if missing_meta:
        logging.warning(f'  sources with no metadata row:        {missing_meta:,}')
    if len(idx) and 'mbh_valid' in idx.columns:
        src = idx.drop_duplicates('source_id')
        lab = src[src['z_valid'].notna()]
        logging.info(f'  labeled sources: {len(lab):,}   '
                     f'valid M_BH: {int(lab.mbh_valid.sum()):,}   '
                     f'valid L_bol: {int(lab.lbol_valid.sum()):,}   '
                     f'valid Eddington: {int(lab.LOGEDD.notna().sum()):,}   '
                     f'z<{LOWZ_LBOL_CAVEAT} L_bol caveat: {int(lab.lbol_lowz_caveat.sum()):,}')
    if len(idx) and 'in_pretrain' in idx.columns and keep_overlap:
        logging.info(f'  kept sources seen in pretraining: '
                     f'{idx.loc[idx.in_pretrain, "source_id"].nunique():,}')
    logging.info('  points per band:')
    for band in sorted(band_counts):
        n = np.array(band_counts[band])
        logging.info(f'    band {band} ({BAND_NAMES.get(band, band)}): '
                     f'{len(n):,} curves, median {int(np.median(n))} points, '
                     f'max {int(n.max())}')
    logging.info(f'  index written to {index_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--tgz', help='.tgz archive of per-source csv files')
    src.add_argument('--lc-dir', help='already-extracted directory of csv files')
    p.add_argument('--meta', help='label table (qzo_labels.csv); carried into lc_index.csv')
    p.add_argument('--labeled-only', action='store_true',
                   help='write only sources that have a row in --meta')
    p.add_argument('--overlap', help='pretrain_overlap.csv: drop sources seen in pretraining')
    p.add_argument('--keep-overlap', action='store_true',
                   help='with --overlap: keep those sources, flag them in lc_index.csv')
    p.add_argument('--output', required=True, help='Output records root')
    p.add_argument('--subset', default='test',
                   choices=['train', 'validation', 'val', 'test'])
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--shard-size', type=int, default=5000)
    p.add_argument('--min-points', type=int, default=5,
                   help='Drop (source,band) curves shorter than this. The '
                        'training pipeline drops <5 anyway via filter_fn.')
    p.add_argument('--bands', type=int, nargs='+', default=[1, 2, 3],
                   help='Which filter codes to export (1=g 2=r 3=i)')
    p.add_argument('--progress-every', type=int, default=5000,
                   help='log progress every N sources (default 5000)')
    p.add_argument('--limit', type=int, default=None,
                   help='convert only the first N selected sources (for a quick test)')
    args = p.parse_args()

    convert(tgz=args.tgz, lc_dir=args.lc_dir, meta_path=args.meta,
            output_dir=args.output, subset=args.subset, fold=args.fold,
            shard_size=args.shard_size, min_points=args.min_points,
            bands=tuple(args.bands), labeled_only=args.labeled_only,
            overlap_path=args.overlap, keep_overlap=args.keep_overlap,
            limit=args.limit, progress_every=args.progress_every)
