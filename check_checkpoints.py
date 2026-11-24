#!/usr/bin/env python3
"""Check that pretrained checkpoints give the same outputs in two environments,
e.g. the old Python 3.11 / TF 2.14 (Keras 2) env and the Python 3.12 / TF 2.21
(Keras 3) one.

Run `dump` once in each environment, on the same list of pretraining dirs, then
`compare` the two files. For every model it records, on a fixed synthetic batch
with a fixed mask:
  rec   the model output (the reconstruction)
  emb   the masked-mean embedding of every encoder layer, as extract_embeddings.py
        computes it (pool=concat), so a match here means matching embeddings
The policy is forced to float32 so neither side uses mixed precision. Models are
loaded with load_pt_model, i.e. exactly as the pipeline loads them (including the
Keras 2 -> Keras 3 weight-path conversion in the new env).

    # old env (acs-MHLA's .venv), then the new one
    .venv/bin/python check_checkpoints.py dump --models models.txt --out ckpt_tf214.npz
    <new-env>/bin/python check_checkpoints.py dump --models models.txt --out ckpt_tf221.npz
    python check_checkpoints.py compare ckpt_tf214.npz ckpt_tf221.npz

--models is a file with one pretraining dir per line (the models.txt that
run_extract_array.sh uses), or the dirs themselves. compare exits non-zero if any
model differs by more than --tol (default 1e-4, absolute) or loaded on one side only.
A GPU and a CPU, or two different GPUs, agree to roughly 1e-5 here, not exactly.

Needs TensorFlow for dump only.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def model_dirs(spec):
    if len(spec) == 1 and os.path.isfile(spec[0]):
        with open(spec[0]) as fh:
            return [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]
    return list(spec)


def fixed_batch(window_size, n=8, seed=1234):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, 2000, (n, window_size, 1)), axis=1)
    times -= times.mean(axis=1, keepdims=True)               # the loader zero-means
    mags = rng.normal(0, 0.3, (n, window_size, 1))
    mags -= mags.mean(axis=1, keepdims=True)
    mask = (rng.random((n, window_size, 1)) < 0.25).astype('f4')
    mask[0, window_size // 2:] = 1.                          # one padded-looking tail
    return {'input': mags.astype('f4'), 'times': times.astype('f4'), 'mask_in': mask}


def dump(args):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    import keras
    keras.mixed_precision.set_global_policy('float32')
    from presentation.pipelines.steps.model_design import load_pt_model

    out = {'_versions': np.array(f'python {sys.version.split()[0]} | TF {tf.__version__} '
                                 f'| keras {keras.__version__}')}
    for d in model_dirs(args.models):
        key = d.rstrip('/')
        try:
            model, cfg = load_pt_model(d)
            x = fixed_batch(int(cfg['window_size']))
            enc = model.get_layer('encoder')
            if hasattr(enc, 'reset_caches'):
                enc.reset_caches()
            rec = np.asarray(model(x, training=False), dtype='f8')
            if hasattr(enc, 'reset_caches'):
                enc.reset_caches()
            zs = enc(x, training=False, z_by_layer=True)
            zs = zs if isinstance(zs, (list, tuple)) else [zs]
            valid = 1. - x['mask_in']
            emb = np.concatenate([(np.asarray(z, 'f8') * valid).sum(1) / valid.sum(1)
                                  for z in zs], axis=-1)
            out[f'{key}::rec'], out[f'{key}::emb'] = rec, emb
            print(f'[ok]   {key}', flush=True)
        except Exception as e:                                # report, keep going
            out[f'{key}::error'] = np.array(f'{type(e).__name__}: {e}'[:2000])
            print(f'[FAIL] {key}: {type(e).__name__}: {str(e).splitlines()[0][:200]}',
                  flush=True)
        tf.keras.backend.clear_session()
    np.savez_compressed(args.out, **out)
    print(f'wrote {args.out} ({out["_versions"]})')


def compare(args):
    a, b = np.load(args.a), np.load(args.b)
    print(f'A: {a["_versions"]}\nB: {b["_versions"]}\n')
    keys = sorted({k.rsplit('::', 1)[0] for k in list(a.files) + list(b.files)
                   if '::' in k})
    bad = 0
    print(f'{"max |diff| rec":>15} {"max |diff| emb":>15}  model')
    for k in keys:
        errs = [f'{side}: {f[k + "::error"]}' for side, f in (('A', a), ('B', b))
                if k + '::error' in f.files or k + '::rec' not in f.files]
        if errs:
            bad += 1
            print(f'{"FAILED":>15} {"":>15}  {k}\n' + '\n'.join(f'{"":33}{e[:300]}' for e in errs))
            continue
        dr = float(np.abs(a[k + '::rec'] - b[k + '::rec']).max())
        de = float(np.abs(a[k + '::emb'] - b[k + '::emb']).max())
        flag = '' if max(dr, de) <= args.tol else '   <-- exceeds --tol'
        bad += bool(flag)
        print(f'{dr:15.2e} {de:15.2e}  {k}{flag}')
    print(f'\n{len(keys) - bad}/{len(keys)} models match within {args.tol:g}')
    sys.exit(1 if bad else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--models', nargs='+', required=True)
    d.add_argument('--out', required=True)
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    c.add_argument('--tol', type=float, default=1e-4)
    args = p.parse_args()
    dump(args) if args.cmd == 'dump' else compare(args)


if __name__ == '__main__':
    main()
