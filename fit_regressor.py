"""
Fit a multi-output regressor on frozen ASTROMER embeddings to predict AGN
parameters: redshift (z), black-hole mass (m), and Eddington ratio (e).

This is step 2b. It consumes the .npz files written by extract_embeddings.py
and the lc_index.csv written by agn_to_tfrecord.py, and needs no TensorFlow, so
you can iterate on the head quickly once embeddings are cached.

TARGETS
    z  Z_FIT     redshift
    m  LOGMBH    log10 black-hole mass
    e  LOGEDD    log10 Eddington ratio = LOGLBOL - 38.1 - LOGMBH
Rows without a usable mass (mbh_valid == False, or label_valid == False in an
older lc_index.csv: LOGMBH sentinel 0.0 / err -1.0) are dropped from the m and e
fits, and from z as well unless --keep-invalid-for-z is passed. Rows whose target
is NaN (e.g. LOGEDD when LOGLBOL is missing) are always dropped.

LOW-z LUMINOSITY CAVEAT
Wu & Shen computed LOGLBOL from L5100 instead of L3000 for z < 0.7 (their May 2025
note), flagged in lc_index.csv as lbol_lowz_caveat. By default those sources are
kept, and the summary also reports r2_e on the sources WITHOUT the caveat
(r2_e_no_lowz). --drop-lowz-lbol removes them from every fit.

AGGREGATION
Embeddings are per (source, band); labels are per source. Bands are combined
before fitting so that each source contributes exactly one row, which also means
a plain KFold cannot leak one band of a source into the fold that scores another.
    --agg mean    average the available band embeddings   (default)
    --agg concat  concatenate [g, r, i], zero-filling any missing band and
                  adding a presence indicator per band
    --agg band --band r   use a single band

HEADS
    --model ridge        RidgeCV over a log-spaced alpha grid (default). A linear
                         probe: how much is LINEARLY readable from the embedding.
    --model mlp          PyTorch MLP on GPU if one is visible (CPU otherwise).
                         Early stopping on a validation split carved out of each
                         training fold, so the test fold is never touched.
    --model mlp-sklearn  the previous scikit-learn MLPRegressor (CPU; slow at 169k)
Run ridge and mlp into separate --out-dir's; plot_true_vs_pred.py --compare
overlays the two. The ridge-vs-MLP gap is itself a result: information that is
in the embedding but not linearly accessible.

TUNING THE MLP (do this once, then fix the settings for every model)
    python fit_regressor.py --tune --emb emb_<one model>.npz --index lc_index.csv
trains a small grid of MLP settings on ONE embedding, on a single 80/20 split of
sources, and writes <out-dir>/mlp_tuning.csv and <out-dir>/mlp_config.json. Pass
that file to every real run with --mlp-config. Tuning each model on its own scores
would reward whichever model got tuned hardest.

All scores are out-of-fold from --folds cross-validation, never in-sample.

Usage:
    python fit_regressor.py \\
        --emb emb_ws200.npz --emb emb_ws800.npz \\
        --index ./data/records/agn_test/fold_0/test/lc_index.csv \\
        --model mlp --mlp-config agn_tune/mlp_config.json \\
        --out-dir ./agn_results_mlp
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

TARGETS = {'z': 'Z_FIT', 'm': 'LOGMBH', 'e': 'LOGEDD'}
BAND_ORDER = {'g': 0, 'r': 1, 'i': 2}
# Photometric summary statistics written by agn_to_tfrecord.py. Used both as a
# standalone control ("stats_only") and, with --with-stats, appended to the
# embeddings. They matter because the loader zero-means every window, so the
# encoder cannot see mean magnitude - which on its own predicts mass well.
STAT_COLS = ['mag_mean', 'mag_std', 'err_med', 'mag_skew', 'mag_kurt', 'n_points']

MLP_DEFAULTS = {'hidden': [512, 256], 'dropout': 0.1, 'weight_decay': 1e-4,
                'lr': 1e-3, 'batch_size': 1024, 'max_epochs': 200,
                'patience': 10, 'val_frac': 0.1}
TUNE_GRID = {'hidden': [[256], [512, 256], [512, 512, 256]],
             'dropout': [0.0, 0.1, 0.3],
             'weight_decay': [1e-5, 1e-4, 1e-3]}


# --------------------------------------------------------------------------- io
def load_embeddings(path):
    d = np.load(path, allow_pickle=True)
    return {
        'lcid': [str(x) for x in d['lcid']],
        'X': d['X'],
        'label': str(d['exp_name']) if 'exp_name' in d else os.path.basename(path),
        'window_size': int(d['window_size']) if 'window_size' in d else -1,
        'pool': str(d['pool']) if 'pool' in d else '?',
        'path': path,
    }


def _source_codes(source_ids, sids):
    pos = pd.Index(sids).get_indexer(source_ids)
    return pos


def build_stats_matrix(index, sids):
    """Per-source photometric summary features, [g|r|i] blocks, 0 for missing."""
    have = [c for c in STAT_COLS if c in index.columns]
    if not have:
        return None
    missing = [c for c in STAT_COLS if c not in index.columns]
    if missing:
        print(f'[WARN] lc_index.csv lacks {missing} - it was written by an '
              'older agn_to_tfrecord.py. Re-run the converter so the control '
              'baseline and --with-stats use the full photometric statistics.',
              file=sys.stderr)
    k = len(have)
    M = np.zeros((len(sids), 3 * k))
    src = _source_codes(index['source_id'].to_numpy(), sids)
    band = index['band_name'].map(BAND_ORDER).to_numpy()
    ok = (src >= 0) & ~pd.isna(band)
    vals = index[have].to_numpy(dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    for b in range(3):
        sel = ok & (band == b)
        M[src[sel], b * k:(b + 1) * k] = vals[sel]
    return M


def build_source_matrix(emb, index, agg='mean', band=None):
    """Collapse per-(source,band) embeddings into one row per source."""
    lut = pd.Series(np.arange(len(emb['lcid'])), index=emb['lcid'])
    idx = index[index['lcid'].isin(lut.index)]
    if idx.empty:
        raise RuntimeError(f'no lcid overlap between {emb["path"]} and the index')
    row = lut.loc[idx['lcid']].to_numpy()
    X = emb['X']
    D = X.shape[1]

    if agg == 'band':
        sel = (idx['band_name'] == band).to_numpy()
        idx, row = idx[sel], row[sel]
    sids = np.array(sorted(idx['source_id'].unique()))
    src = _source_codes(idx['source_id'].to_numpy(), sids)

    if agg in ('mean', 'band'):
        M = np.zeros((len(sids), D), dtype=np.float64)
        np.add.at(M, src, X[row].astype(np.float64))
        cnt = np.bincount(src, minlength=len(sids)).astype(np.float64)
        M /= cnt[:, None]
    elif agg == 'concat':
        M = np.zeros((len(sids), 3 * D + 3), dtype=np.float64)
        b = idx['band_name'].map(BAND_ORDER).to_numpy()
        for k in range(3):
            s = b == k
            M[src[s], k * D:(k + 1) * D] = X[row[s]]
            M[src[s], 3 * D + k] = 1.0
    else:
        raise ValueError(f'unknown --agg {agg}')
    return M, sids


# ------------------------------------------------------------------------ heads
class TorchMLP:
    """Multi-output MLP regressor. Expects standardized X and Y."""

    def __init__(self, cfg, seed, device=None, verbose=False):
        import torch
        self.torch = torch
        self.cfg = {**MLP_DEFAULTS, **(cfg or {})}
        self.seed = seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.best_epoch = None

    def _net(self, d_in, d_out):
        nn = self.torch.nn
        layers, d = [], d_in
        for h in self.cfg['hidden']:
            layers += [nn.Linear(d, h), nn.GELU()]
            if self.cfg['dropout'] > 0:
                layers.append(nn.Dropout(self.cfg['dropout']))
            d = h
        layers.append(nn.Linear(d, d_out))
        return nn.Sequential(*layers)

    def fit(self, X, Y):
        torch, c = self.torch, self.cfg
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n = len(X)
        perm = rng.permutation(n)
        n_val = max(1, int(round(c['val_frac'] * n)))
        va, tr = perm[:n_val], perm[n_val:]
        dev = self.device
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
        self.net = self._net(X.shape[1], Y.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c['lr'],
                                weight_decay=c['weight_decay'])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                           patience=max(1, c['patience'] // 3))
        lossf = torch.nn.MSELoss()
        tr_t = torch.as_tensor(tr, device=dev)
        va_t = torch.as_tensor(va, device=dev)
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator(device='cpu').manual_seed(self.seed)
        for ep in range(c['max_epochs']):
            self.net.train()
            order = tr_t[torch.randperm(len(tr_t), generator=g).to(dev)]
            for i in range(0, len(order), c['batch_size']):
                b = order[i:i + c['batch_size']]
                opt.zero_grad(set_to_none=True)
                loss = lossf(self.net(Xt[b]), Yt[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(lossf(self._predict_t(Xt[va_t]), Yt[va_t]))
            sched.step(vl)
            if vl < best - 1e-5:
                best, bad, self.best_epoch = vl, 0, ep + 1
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= c['patience']:
                    break
        self.net.load_state_dict(best_state)
        self.best_val = best
        return self

    def _predict_t(self, Xt):
        out, bs = [], 65536
        for i in range(0, len(Xt), bs):
            out.append(self.net(Xt[i:i + bs]))
        return self.torch.cat(out)

    def predict(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self._predict_t(Xt).cpu().numpy().astype(np.float64)


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:                                            # noqa: BLE001
        return False


def make_model(kind, seed, mlp_cfg=None, device=None):
    if kind == 'ridge':
        return RidgeCV(alphas=np.logspace(-2, 6, 40))
    if kind == 'mlp':
        return TorchMLP(mlp_cfg, seed, device=device)
    if kind == 'mlp-sklearn':
        return MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-2,
                            max_iter=2000, early_stopping=True,
                            random_state=seed)
    raise ValueError(kind)


def r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_predict(Xtr, Ytr, Xte, kind, seed, mlp_cfg=None, device=None):
    """Scale inside the split, fit, predict, un-scale."""
    xs, ys = StandardScaler(), StandardScaler()
    Xtr_s = xs.fit_transform(Xtr)
    Xte_s = xs.transform(Xte)
    Ytr_s = ys.fit_transform(Ytr)
    m = make_model(kind, seed, mlp_cfg, device)
    if kind == 'mlp':
        m.fit(Xtr_s, Ytr_s)
    else:
        m.fit(Xtr_s, Ytr_s if Ytr_s.shape[1] > 1 else Ytr_s.ravel())
    pred = m.predict(Xte_s)
    if pred.ndim == 1:
        pred = pred[:, None]
    return ys.inverse_transform(pred), m


def cross_val_predict(X, Y, folds, model_kind, seed, mlp_cfg=None, device=None,
                      label=''):
    """Out-of-fold predictions. Scalers (and MLP early stopping) live inside each fold."""
    oof = np.full_like(Y, np.nan, dtype=np.float64)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(kf.split(X)):
        t0 = time.time()
        oof[te], m = fit_predict(X[tr], Y[tr], X[te], model_kind, seed, mlp_cfg, device)
        if model_kind == 'mlp':
            print(f'      fold {k + 1}/{folds}: {time.time() - t0:5.1f}s, '
                  f'best epoch {m.best_epoch}', flush=True)
    return oof


# ---------------------------------------------------------------------- tuning
def run_tuning(X, Y, tgt_keys, args, device):
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_te = int(round(0.2 * len(X)))
    te, tr = perm[:n_te], perm[n_te:]
    base = json.load(open(args.mlp_config)) if args.mlp_config else {}
    rows = []
    combos = list(itertools.product(TUNE_GRID['hidden'], TUNE_GRID['dropout'],
                                    TUNE_GRID['weight_decay']))
    print(f'[TUNE] {len(combos)} settings, train {len(tr):,} / held-out {len(te):,} '
          f'sources, device={device}')
    for i, (h, dr, wd) in enumerate(combos):
        cfg = {**MLP_DEFAULTS, **base, 'hidden': h, 'dropout': dr, 'weight_decay': wd}
        t0 = time.time()
        pred, m = fit_predict(X[tr], Y[tr], X[te], 'mlp', args.seed, cfg, device)
        row = {'hidden': 'x'.join(map(str, h)), 'dropout': dr, 'weight_decay': wd,
               'best_epoch': m.best_epoch, 'seconds': round(time.time() - t0, 1)}
        for j, t in enumerate(tgt_keys):
            row[f'r2_{t}'] = r2(Y[te, j], pred[:, j])
        row['r2_mean'] = float(np.mean([row[f'r2_{t}'] for t in tgt_keys]))
        rows.append(row)
        print(f'  [{i + 1:2d}/{len(combos)}] hidden={row["hidden"]:<12} '
              f'dropout={dr:<4} wd={wd:<7g} ' +
              ' '.join(f'{t}={row[f"r2_{t}"]:.3f}' for t in tgt_keys) +
              f'  ({row["seconds"]}s, epoch {m.best_epoch})', flush=True)
    ridge_pred, _ = fit_predict(X[tr], Y[tr], X[te], 'ridge', args.seed)
    df = pd.DataFrame(rows).sort_values('r2_mean', ascending=False)
    df.to_csv(os.path.join(args.out_dir, 'mlp_tuning.csv'), index=False)
    best = df.iloc[0]
    cfg = {**MLP_DEFAULTS, **base,
           'hidden': [int(v) for v in best['hidden'].split('x')],
           'dropout': float(best['dropout']), 'weight_decay': float(best['weight_decay'])}
    path = os.path.join(args.out_dir, 'mlp_config.json')
    json.dump(cfg, open(path, 'w'), indent=2)
    print('\n' + df.head(10).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('\nridge on the same split: ' +
          ' '.join(f'{t}={r2(Y[te, j], ridge_pred[:, j]):.3f}' for j, t in enumerate(tgt_keys)))
    print(f'\n[TUNE] best: {cfg}\n[TUNE] wrote {path} and mlp_tuning.csv; '
          f'pass --mlp-config {path} to every run.')
    spread = df['r2_mean'].max() - df['r2_mean'].median()
    if spread < 0.005:
        print('[TUNE] the grid barely matters (best - median < 0.005 in R2); '
              'any of these settings is fine.')


# ------------------------------------------------------------------- bootstrap
def paired_bootstrap(oof_store, tgt_keys, n_boot, seed, chunk=50):
    """Resample shared sources with replacement; returns {label: {t: array}}.

    Vectorized: a resample is a vector of multiplicities w, so every model's
    sum of squared errors is w @ E and the target variance comes from w @ y, w @ y^2.
    """
    labels = list(oof_store)
    common = sorted(set.intersection(*[set(v[0]) for v in oof_store.values()]))
    n = len(common)
    take = {l: pd.Index(oof_store[l][0]).get_indexer(common) for l in labels}
    Yc = oof_store[labels[0]][1][take[labels[0]]]
    E = {t: np.column_stack([(oof_store[l][2][take[l], j] - Yc[:, j]) ** 2
                             for l in labels]) for j, t in enumerate(tgt_keys)}
    rng = np.random.default_rng(seed)
    out = {l: {t: [] for t in tgt_keys} for l in labels}
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        W = np.zeros((b, n), dtype=np.float64)
        for i in range(b):
            W[i] = np.bincount(rng.integers(0, n, n), minlength=n)
        for j, t in enumerate(tgt_keys):
            y = Yc[:, j]
            sy, syy = W @ y, W @ (y * y)
            den = syy - sy * sy / n
            num = W @ E[t]                                       # (b, L)
            r2b = 1.0 - num / den[:, None]
            for k, l in enumerate(labels):
                out[l][t].append(r2b[:, k])
        done += b
    return {l: {t: np.concatenate(v) for t, v in d.items()} for l, d in out.items()}, n


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--emb', action='extend', nargs='+', required=True,
                   help='.npz files from extract_embeddings.py. Accepts either '
                        'a shell glob (--emb emb_*.npz) or a repeated flag '
                        '(--emb a.npz --emb b.npz).')
    p.add_argument('--index', required=True, help='lc_index.csv')
    p.add_argument('--targets', default='z,m,e')
    p.add_argument('--agg', default='mean', choices=['mean', 'concat', 'band'])
    p.add_argument('--band', default='r', choices=['g', 'r', 'i'])
    p.add_argument('--model', default='ridge', choices=['ridge', 'mlp', 'mlp-sklearn'])
    p.add_argument('--mlp-config', default=None,
                   help='JSON of MLP settings (written by --tune)')
    p.add_argument('--device', default=None, help='torch device (default: cuda if visible)')
    p.add_argument('--tune', action='store_true',
                   help='tune the MLP on the FIRST --emb and exit (see docstring)')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seeds', type=int, default=1,
                   help='Number of CV seeds to average over. With ~1k sources '
                        'use 5; with ~170k the spread is negligible and the '
                        'paired bootstrap carries the uncertainty.')
    p.add_argument('--with-stats', action='store_true',
                   help='Append photometric summary stats to the embedding '
                        'features (recovers brightness/amplitude, which the '
                        'zero-mean windowing removes before the encoder)')
    p.add_argument('--bootstrap', type=int, default=2000,
                   help='Paired bootstrap resamples over sources for comparing '
                        'models against the control. 0 disables.')
    p.add_argument('--no-stats-baseline', action='store_true',
                   help='Skip the summary-statistics-only control row')
    p.add_argument('--keep-invalid-for-z', action='store_true',
                   help='Keep sentinel-mass sources when fitting z only')
    p.add_argument('--drop-lowz-lbol', action='store_true',
                   help='Drop sources flagged lbol_lowz_caveat from every fit')
    p.add_argument('--out-dir', default='./agn_results')
    args = p.parse_args()

    if args.model == 'mlp' or args.tune:
        if not torch_available():
            sys.exit('--model mlp / --tune need PyTorch. Install it, or use '
                     '--model mlp-sklearn (CPU, slow at this size).')
        import torch
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            print('[WARN] no GPU visible: the MLP will run on CPU. It works, '
                  'but expect minutes per fold at ~170k sources.')
    else:
        device = None
    mlp_cfg = {**MLP_DEFAULTS, **(json.load(open(args.mlp_config)) if args.mlp_config else {})}
    if args.model == 'mlp' and not args.tune:
        src = args.mlp_config or 'built-in defaults (run --tune first to choose)'
        print(f'[INFO] MLP settings from {src}: {mlp_cfg}')

    print(f'[INFO] fit_regressor: {args.seeds} CV seed(s) x {args.folds} folds, '
          f'head={args.model}, agg={args.agg}, with_stats={args.with_stats}')
    os.makedirs(args.out_dir, exist_ok=True)
    index = pd.read_csv(args.index, dtype={'source_id': str, 'lcid': str})
    tgt_keys = [t.strip() for t in args.targets.split(',') if t.strip()]
    for t in tgt_keys:
        if t not in TARGETS:
            sys.exit(f'unknown target {t}; choose from {list(TARGETS)}')

    # one label row per source
    valid_col = 'mbh_valid' if 'mbh_valid' in index.columns else (
        'label_valid' if 'label_valid' in index.columns else None)
    lab_cols = ['source_id'] + [TARGETS[t] for t in tgt_keys]
    for c in (valid_col, 'lbol_lowz_caveat'):
        if c and c in index.columns and c not in lab_cols:
            lab_cols.append(c)
    labels = index[lab_cols].drop_duplicates('source_id').set_index('source_id')
    has_lowz = 'lbol_lowz_caveat' in labels.columns

    summary, all_preds = [], []
    stats_done = False
    oof_store = {}   # label -> (source_ids, Y, oof) for the paired bootstrap
    for path in args.emb:
        t_emb = time.time()
        emb = load_embeddings(path)
        X, sids = build_source_matrix(emb, index, agg=args.agg, band=args.band)

        lab = labels.reindex(sids)
        keep = np.ones(len(sids), dtype=bool)
        if valid_col:
            need_valid = any(t in ('m', 'e') for t in tgt_keys) or \
                not args.keep_invalid_for_z
            if need_valid:
                keep &= lab[valid_col].fillna(False).to_numpy().astype(bool)
        if args.drop_lowz_lbol and has_lowz:
            keep &= ~lab['lbol_lowz_caveat'].fillna(False).to_numpy().astype(bool)
        Y = np.column_stack([lab[TARGETS[t]].to_numpy(dtype=float)
                             for t in tgt_keys])
        keep &= np.isfinite(Y).all(axis=1)
        lowz = (lab['lbol_lowz_caveat'].fillna(False).to_numpy().astype(bool)
                if has_lowz else np.zeros(len(sids), bool))

        S_all = build_stats_matrix(index, sids)
        if args.with_stats and S_all is not None:
            X = np.hstack([X, S_all])
        X, Y, sids_k, lowz_k = X[keep], Y[keep], sids[keep], lowz[keep]
        S = S_all[keep] if S_all is not None else None

        if args.tune:
            print(f'[TUNE] embedding {emb["label"]}  X={X.shape}')
            run_tuning(X, Y, tgt_keys, args, device)
            return

        print(f'\n=== {emb["label"]}  (ws={emb["window_size"]}, pool={emb["pool"]}) ===')
        print(f'    X={X.shape}  sources={len(sids_k)}  '
              f'dropped={int((~keep).sum())}  model={args.model} agg={args.agg}')

        seeds = list(range(args.seed, args.seed + args.seeds))
        oofs = [cross_val_predict(X, Y, args.folds, args.model, sd, mlp_cfg, device)
                for sd in seeds]
        oof = oofs[0]

        row = {'model': emb['label'], 'window_size': emb['window_size'],
               'pool': emb['pool'], 'agg': args.agg, 'head': args.model,
               'n_sources': len(sids_k), 'n_features': X.shape[1]}
        for j, t in enumerate(tgt_keys):
            vals = np.array([r2(Y[:, j], o[:, j]) for o in oofs])
            row[f'r2_{t}'] = float(vals.mean())
            row[f'r2_{t}_std'] = float(vals.std())
            row[f'rmse_{t}'] = float(np.sqrt(np.mean((Y[:, j] - oof[:, j]) ** 2)))
            pm = f' +/- {vals.std():.3f}' if len(vals) > 1 else ''
            print(f'    {t} ({TARGETS[t]:7s}): R2={vals.mean():6.3f}{pm}  '
                  f'RMSE={row[f"rmse_{t}"]:.3f}')
        if 'e' in tgt_keys and lowz_k.any() and (~lowz_k).sum() > 5:
            j = tgt_keys.index('e')
            row['r2_e_no_lowz'] = r2(Y[~lowz_k, j], oof[~lowz_k, j])
            print(f'    e without z<0.7 L_bol caveat ({int((~lowz_k).sum()):,} sources): '
                  f'R2={row["r2_e_no_lowz"]:6.3f}')

        # step-3 conditioning: does long context help most at high mass?
        if 'm' in tgt_keys and 'z' in tgt_keys:
            mtrue = Y[:, tgt_keys.index('m')]
            ztrue, zpred = Y[:, tgt_keys.index('z')], oof[:, tgt_keys.index('z')]
            edges = np.quantile(mtrue, [0, 1 / 3, 2 / 3, 1.0])
            names = ['low', 'mid', 'high']
            print('    z R2 by mass tercile:', end='')
            for b in range(3):
                sel = (mtrue >= edges[b]) & (
                    mtrue <= edges[b + 1] if b == 2 else mtrue < edges[b + 1])
                v = r2(ztrue[sel], zpred[sel]) if sel.sum() > 5 else np.nan
                row[f'r2_z_mass_{names[b]}'] = v
                print(f'  {names[b]}({edges[b]:.2f}-{edges[b+1]:.2f})={v:6.3f}',
                      end='')
            print()

        summary.append(row)

        # Control: the same head on hand-made summary statistics only. If the
        # embeddings do not clearly beat this, they are not contributing.
        if S is not None and not args.no_stats_baseline and not stats_done:
            print('    control (summary stats only):')
            srow = {'model': 'CONTROL: summary stats only', 'window_size': -1,
                    'pool': '-', 'agg': args.agg, 'head': args.model,
                    'n_sources': len(sids_k), 'n_features': S.shape[1]}
            soofs = [cross_val_predict(S, Y, args.folds, args.model, sd, mlp_cfg, device)
                     for sd in seeds]
            for j, t in enumerate(tgt_keys):
                vals = np.array([r2(Y[:, j], o[:, j]) for o in soofs])
                srow[f'r2_{t}'] = float(vals.mean())
                srow[f'r2_{t}_std'] = float(vals.std())
                srow[f'rmse_{t}'] = float(np.sqrt(
                    np.mean((Y[:, j] - soofs[0][:, j]) ** 2)))
                print(f'      {t}: R2={srow[f"r2_{t}"]:6.3f}')
            if 'e' in tgt_keys and lowz_k.any() and (~lowz_k).sum() > 5:
                j = tgt_keys.index('e')
                srow['r2_e_no_lowz'] = r2(Y[~lowz_k, j], soofs[0][~lowz_k, j])
            summary.append(srow)
            oof_store[srow['model']] = (sids_k, Y, soofs[0])
            spr = pd.DataFrame({'source_id': sids_k})
            for j, t in enumerate(tgt_keys):
                spr[f'{t}_true'] = Y[:, j]
                spr[f'{t}_pred'] = soofs[0][:, j]
            spr['model'] = srow['model']
            spr['window_size'] = -1
            spr['head'] = args.model
            all_preds.append(spr)
            stats_done = True

        oof_store[row['model']] = (sids_k, Y, oof)

        pr = pd.DataFrame({'source_id': sids_k})
        for j, t in enumerate(tgt_keys):
            pr[f'{t}_true'] = Y[:, j]
            pr[f'{t}_pred'] = oof[:, j]
        pr['model'] = emb['label']
        pr['window_size'] = emb['window_size']
        pr['head'] = args.model
        all_preds.append(pr)
        print(f'    ({time.time() - t_emb:.0f}s for this embedding)', flush=True)

    # ---- paired bootstrap over sources -------------------------------------
    # --seeds varies only the fold split with the embeddings held fixed, so its
    # spread is small and does NOT tell you whether two models differ. Every
    # model predicts the same sources, so resample those sources jointly and
    # look at the paired difference in R^2. This propagates the uncertainty that
    # actually matters: that these are one particular sample of AGN.
    if args.bootstrap > 0 and len(oof_store) > 1:
        labels_b = list(oof_store)
        boot, n = paired_bootstrap(oof_store, tgt_keys, args.bootstrap, args.seed)
        ref = 'CONTROL: summary stats only'
        if ref not in labels_b:
            ref = labels_b[0]

        print('\n' + '=' * 78)
        print(f'PAIRED BOOTSTRAP ({args.bootstrap} resamples of the {n:,} shared '
              f'sources)')
        print(f'delta R2 vs "{ref}"; P(better) is the fraction of resamples '
              'where the model wins')
        print('=' * 78)
        hdr = f'{"model":<50}' + ''.join(
            f'{"d" + t:>9}{"P>ref":>8}' for t in tgt_keys)
        print(hdr)
        print('-' * len(hdr))
        rows_b = []
        for lbl in labels_b:
            if lbl == ref:
                continue
            line = f'{lbl[:49]:<50}'
            rb = {'model': lbl, 'reference': ref}
            for t in tgt_keys:
                d = boot[lbl][t] - boot[ref][t]
                line += f'{d.mean():>9.3f}{(d > 0).mean():>8.2f}'
                rb[f'delta_r2_{t}'] = float(d.mean())
                rb[f'delta_r2_{t}_lo'] = float(np.percentile(d, 2.5))
                rb[f'delta_r2_{t}_hi'] = float(np.percentile(d, 97.5))
                rb[f'p_better_{t}'] = float((d > 0).mean())
            print(line)
            rows_b.append(rb)
        print('-' * len(hdr))
        print('P near 0.5 means indistinguishable; near 1.0 means a real win.')
        pd.DataFrame(rows_b).to_csv(
            os.path.join(args.out_dir, 'bootstrap_vs_reference.csv'), index=False)

    sm = pd.DataFrame(summary)
    cols = ['model', 'window_size', 'n_sources', 'n_features'] + \
           [c for c in sm.columns if c.startswith(('r2_', 'rmse_')) and not c.endswith('_std')
            and not c.startswith('r2_z_mass')]
    print('\n' + '=' * 78)
    print(sm[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('=' * 78)

    sm.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)
    pd.concat(all_preds).to_csv(
        os.path.join(args.out_dir, 'predictions.csv'), index=False)
    print(f'\n[INFO] wrote {args.out_dir}/summary.csv and predictions.csv')
    print('[INFO] predictions.csv has <t>_true / <t>_pred per source for '
          'plot_true_vs_pred.py')


if __name__ == '__main__':
    main()
