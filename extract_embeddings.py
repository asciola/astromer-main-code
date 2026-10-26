"""
Extract pooled encoder embeddings from a pretrained ASTROMER model for every
light curve in a records directory, and save them keyed by lcid.

This is step 2a of the AGN regression plan: freeze the pretrained encoder, turn
each light curve into a fixed-length feature vector, then fit a small
multi-output regressor on those vectors (see fit_regressor.py).

The pooling reproduces what get_avg_clf / get_avg_mlp do inside
build_classifier: mask out padding, then average the encoder output over time.

    x = sum(z * mask, axis=time) / sum(mask, axis=time)

--pool last    pool the final encoder layer only          -> d_model features
--pool concat  pool every layer and concatenate them      -> (L+1)*d_model
               (the un-weighted analogue of skip_avg_mlp, which learns a
                GammaWeight over the same per-layer pooled vectors)

Masking: the loader is built with probed=1.0, random=0.0, same=0.0, i.e. exactly
the values build_loader uses for clf_mode, so the encoder sees the light curve
the same way it does during classification. get_loader is called directly rather
than via build_loader(clf_mode=True) for two reasons: clf_mode needs an
objects.csv / metadata parquet to infer a class count, and it replaces the output
dict with one-hot labels, which would discard the lcid we need for joining.

Light curves longer than window_size produce several windows; their embeddings
are averaged into a single vector per lcid.

Usage:
    python -m extract_embeddings \\
        --pt-model ./presentation/results/<exp>/<ts>/pretraining \\
        --data ./data/records/agn_test/fold_0 \\
        --subset test \\
        --pool last \\
        --out emb_ws200.npz
"""
import argparse
import os
import sys
from collections import defaultdict

# CUDA_VISIBLE_DEVICES must be set before TensorFlow is imported, otherwise the
# device list may already be fixed by the time we get to main().
def _pre_parse(flag, default=None):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return default


_gpu = _pre_parse('--gpu')
if _gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu

import numpy as np
import tensorflow as tf

from presentation.pipelines.steps.model_design import load_pt_model
from src.data.loaders import get_loader
from src.models.astromer_0 import build_input as build_input_zero
from src.models.astromer_1 import build_input as build_input_base


def build_extractor(astromer, cfg, pool='last'):
    """Wrap the frozen encoder in a model that returns pooled embeddings."""
    arch = cfg.get('arch', 'base')
    if arch == 'zero':
        inp = build_input_zero(cfg['window_size'])
    else:
        inp = build_input_base(cfg['window_size'])

    encoder = astromer.get_layer('encoder')
    encoder.trainable = False

    z_by_layer = encoder(inp, z_by_layer=True)
    if not isinstance(z_by_layer, (list, tuple)):
        z_by_layer = [z_by_layer]

    mask = 1. - inp['mask_in']                      # 1 where the point is real
    denom = tf.reduce_sum(mask, axis=1)             # (batch, 1)

    def pooled(z):
        return tf.math.divide_no_nan(
            tf.reduce_sum(tf.multiply(z, mask), axis=1), denom)

    if pool == 'last':
        out = pooled(z_by_layer[-1])
    elif pool == 'concat':
        out = tf.concat([pooled(z) for z in z_by_layer], axis=-1)
    else:
        raise ValueError(f'unknown --pool {pool}')

    return tf.keras.Model(inputs=inp, outputs=out, name='embedding_extractor')


def extract(pt_dir, data_path, subset, bs, pool, normalize='zero-mean',
            allow_cpu=False):
    astromer, cfg = load_pt_model(pt_dir)
    extractor = build_extractor(astromer, cfg, pool=pool)
    print(f'[INFO] window_size={cfg["window_size"]} arch={cfg.get("arch")} '
          f'pool={pool} -> {extractor.output_shape[-1]} features', flush=True)

    # Attention is O(window^2): a 10x longer window costs 100x more memory.
    W = int(cfg['window_size'])
    heads = int(cfg.get('num_heads', 4))
    attn_gb = bs * heads * W * W * 4 / 1024 ** 3
    print(f'[INFO] attention matrix ~{attn_gb:.2f} GB per layer '
          f'(bs={bs}, heads={heads}, window={W})', flush=True)
    if attn_gb > 2.0:
        print(f'[WARN] that is large; several layers/intermediates are live at '
              f'once. If the process is Killed (SIGKILL = host OOM), drop --bs. '
              f'--bs {max(1, int(bs * 2.0 / attn_gb))} would give ~2 GB/layer.',
              flush=True)
    if not tf.config.list_physical_devices('GPU'):
        msg = ('no GPU visible - these activations would come out of host RAM. '
               f'At window {W} / bs {bs} that is ~{attn_gb:.2f} GB per layer '
               'and will likely be OOM-killed (exit -9). Run inside a GPU '
               'allocation, or pass --allow-cpu to proceed anyway.')
        if allow_cpu:
            print(f'[WARN] {msg}', flush=True)
        else:
            raise RuntimeError(msg)

    loader = get_loader(
        os.path.join(data_path, subset),
        batch_size=bs,
        window_size=cfg['window_size'],
        probed_frac=1.,     # clf_mode values: nothing hidden from the encoder
        random_frac=0.,
        same_frac=0.,
        sampling=False,
        shuffle=False,      # keep deterministic order
        repeat=1,
        num_cls=None,       # keeps 'lcid' in the output dict
        normalize=normalize,
        cache=False,
        aversion=cfg.get('arch', 'base'),
    )

    encoder = astromer.get_layer('encoder')
    can_reset = hasattr(encoder, 'reset_caches')
    if can_reset:
        print('[INFO] encoder exposes reset_caches(); clearing per batch',
              flush=True)

    sums = defaultdict(lambda: None)
    counts = defaultdict(int)
    n_windows = 0
    for inputs, outputs in loader:
        # KV-cache models (use_kv_cache=True) accumulate state across calls;
        # clear it every batch so memory does not grow without bound.
        if can_reset:
            encoder.reset_caches()
        emb = extractor(inputs, training=False).numpy()
        lcids = [x.decode() for x in outputs['lcid'].numpy().ravel()]
        for lcid, vec in zip(lcids, emb):
            if sums[lcid] is None:
                sums[lcid] = vec.astype(np.float64).copy()
            else:
                sums[lcid] += vec
            counts[lcid] += 1
            n_windows += 1

    if not sums:
        raise RuntimeError('no windows produced - check --data/--subset path')

    lcid_list = sorted(sums)
    X = np.stack([sums[k] / counts[k] for k in lcid_list]).astype(np.float32)
    W = np.array([counts[k] for k in lcid_list], dtype=np.int32)
    print(f'[INFO] {n_windows:,} windows -> {len(lcid_list):,} light curves '
          f'(mean {W.mean():.2f} windows/curve)', flush=True)
    return lcid_list, X, W, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pt-model', required=True)
    p.add_argument('--data', required=True,
                   help='Records root containing <subset>/ (e.g. .../fold_0)')
    p.add_argument('--subset', default='test')
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--pool', default='last', choices=['last', 'concat'])
    p.add_argument('--normalize', default='zero-mean')
    p.add_argument('--gpu', default='0')
    p.add_argument('--allow-cpu', action='store_true',
                   help='Proceed without a GPU. Only sensible for small '
                        'windows; large ones will exhaust host RAM.')
    p.add_argument('--out', required=True, help='Output .npz path')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    lcids, X, W, cfg = extract(args.pt_model, args.data, args.subset,
                               args.bs, args.pool, args.normalize,
                               allow_cpu=args.allow_cpu)

    np.savez_compressed(
        args.out,
        lcid=np.array(lcids),
        X=X,
        n_windows=W,
        window_size=cfg.get('window_size'),
        latent_dim=cfg.get('latent_dim') if cfg.get('latent_dim') else -1,
        num_layers=cfg.get('num_layers'),
        exp_name=cfg.get('exp_name', os.path.basename(args.pt_model)),
        pool=args.pool,
        pt_model=args.pt_model,
    )
    print(f'[INFO] wrote {args.out}  X={X.shape}')


if __name__ == '__main__':
    main()
