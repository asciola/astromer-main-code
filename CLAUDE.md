# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of **Astromer 2** (self-supervised transformer for astronomical light curves, TensorFlow 2.14) that has been extended on the `acs-MHLA` branch with:

1. **Multi-Head Latent Attention (MHLA)**, an optional KV-compressed attention layer in the encoder.
2. An **AGN/quasar regression study**: pretrain on ZTF light curves (`ztfsurvivors`), freeze the encoder, extract pooled embeddings for a held-out quasar set, and regress redshift (`z`), black-hole mass (`m`, LOGMBH), and Eddington ratio (`e`) with ridge or MLP heads. Models are compared across window size and latent size.

There is no test suite, linter, or build step. Training and embedding extraction run on a SLURM GPU cluster (Harvard FASRC, `seas_gpu` partition). The regression and plotting scripts run locally on cached `.npz`/`.csv` outputs.

## Environments (uv, not conda)

Dependencies are managed by **uv**. There are two environments, and they cannot be merged: TensorFlow 2.14 needs numpy<2 and CUDA 11, while the GPU torch build needs CUDA 12.

| | project | venv | Python | for |
|---|---|---|---|---|
| **tf** (default) | `pyproject.toml` | `.venv` | 3.11 | pretraining, finetuning, embeddings, record conversion, plotting |
| **agn** | `envs/agn-torch/pyproject.toml` | `envs/agn-torch/.venv` | 3.11 | `fit_regressor.py --model mlp` on a GPU |

```bash
uv sync                           # create/update the tf env
uv sync --project envs/agn-torch  # create/update the agn env
```

In scripts, `source activate_env.sh` (optionally `source activate_env.sh agn`) replaces the old `conda activate` two-liner. Interactively, `uv run python ...` also works and syncs first.

uv's cache and its downloaded interpreters are kept on lab storage (`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, set by `activate_env.sh`) because `/n/home07` sits at its quota. Both are large and freely regenerable; never put them back in `$HOME`.

**Python 3.11 is a ceiling, not a preference.** TensorFlow 2.14 publishes no wheels past cp311. Going to 3.12 would require TF >= 2.16, which is where `tf.keras` became Keras 3 -- and Keras 3 cannot reliably reload the existing `out.weights.h5` checkpoints (202 of them under `presentation/results`), because they were saved from subclassed `Model`s with custom `train_step`/`test_step`. Treat any Python bump past 3.11 as a TF/Keras-3 migration with a checkpoint-conversion step, not as an environment change.

Relatedly, the CUDA 11 libraries are listed explicitly in `pyproject.toml` instead of using the `tensorflow[and-cuda]` extra. That extra forces `tensorrt==8.5.3.1`, which ships a **cp310-only** wheel and was the real thing pinning this project to Python 3.10. Nothing here calls TF-TRT, so it is dropped; TF prints a harmless "Could not find TensorRT" line at import. The nvidia-*-cu11 pins are exactly what the extra resolved to, so the GPU stack is unchanged. They carry a `sys_platform == 'linux'` marker (NVIDIA ships no macOS wheels), so `uv sync` also works on an Apple Silicon laptop, giving a CPU-only TF there. That needs the explicit `tensorflow-macos` dependency: on macOS arm64 the `tensorflow` 2.14 wheel is an empty shim that requires it, and uv locks from the Linux wheel's metadata, which does not.

Add a dependency with `uv add <pkg>` (the tf env) or `uv add --project envs/agn-torch <pkg>`, then commit the changed `uv.lock`. `mlflow` and Jupyter are optional extras (`uv sync --extra notebook`), not part of the default install.

## Running code

Run everything **from the repo root**. Imports are package-absolute (`from src...`, `from presentation.pipelines.steps...`), and `presentation/` has no `__init__.py`, so pipeline scripts only work as namespace-package modules:

```bash
# Pretrain (single GPU). The output goes to presentation/results/<exp-name>/<timestamp>/pretraining/
python -m presentation.pipelines.pipeline_0.pretrain --data <records_dir> --exp-name <name> --gpu 0 \
    --window-size 200 --num-layers 6 --num-heads 4 --head-dim 64 --mask-format K
# Add MHLA:   --use-kv-cache --latent-dim 16
# Resume:     --checkpoint <path/to/.../pretraining>

# Finetune / classify (original paper pipeline)
python -m presentation.pipelines.pipeline_0.finetune --pt-model <.../pretraining> --data <records_dir> --gpu 0

# Debug run (takes 2 batches and runs 2 epochs)
python -m presentation.pipelines.pipeline_0.pretrain --data <records_dir> --debug
```

Many bash files under `presentation/pipelines/*/bash/` and `presentation/tutorials/*.md` are stale. They call `presentation.scripts.pretrain` and `presentation.scripts.test_model`, which no longer exist. The current entry points are the ones listed above. Paths in the README that start with `context/presentation/...` actually mean `presentation/...`.

Datasets for the original paper (alcock, atlas, macho, ogle): `cd data && bash get_data.sh <name>[-record]`.

## AGN regression pipeline (top-level scripts)

Each script has a detailed docstring. Read it before changing the script. The data flow is:

1. `crossmatch_labels.py` builds `qzo_labels.csv`. It decodes RA/Dec from Pan-STARRS1 objIDs and crossmatches them to the Wu & Shen DR16Q catalog. `check_labels.py` validates the result.
2. `survivor_overlap.py detect|fetch|match` writes `pretrain_overlap.csv`, which flags quasars that were already in the ZTF pretraining set (queries IRSA TAP).
3. `agn_to_tfrecord.py` converts per-source CSVs into TFRecord shards. Each (source, band) pair becomes one light curve with `lcid = "<sourceid>_b<band>"`, and the script writes `lc_index.csv` (labels, flags, photometric stats).
4. `extract_embeddings.py` (run as `python -m extract_embeddings`; it needs TF) takes a frozen encoder and masked-mean pools it over time. Windows are averaged per `lcid`, and the result is written to `.npz` (`lcid`, `X`, `exp_name`, `window_size`, `pool`). On the cluster this runs as a SLURM array over `models.txt`: see `run_extract_array.sh`.
5. `fit_regressor.py` needs no TF (MLP uses torch). It averages or concatenates bands to one row per source and runs KFold out-of-fold predictions. It also fits a "summary statistics only" **control**, adds those stats to the embeddings with `--with-stats`, and runs a paired bootstrap against the control. It writes `summary.csv`, `predictions.csv`, and `bootstrap_vs_reference.csv` to `--out-dir`. To tune the MLP once, run `--tune`, then pass the resulting `mlp_config.json` to every run with `--mlp-config`.
6. Analysis of the outputs: `summarize_results.py` (tables, including LaTeX), `plot_window_trend.py`, `plot_true_vs_pred.py`, `length_mass_gain.py`, `oof_complementarity.py`.

`cluster_results/` holds the SLURM scripts (`run_ridge*.sh`, `run_agn_mlp.sh`) and outputs pulled back from the cluster. Cluster data lives under `/n/holylabs/protopapas_lab/Lab/asciola/`.

**Model naming is parsed by the downstream scripts**. Keep the experiment names in this form:
- MHLA: `..._K_01_<latent>_<window>`, for example `ztfsurvivors_full_6_5e-5_256_K_01_16_200`
- No bottleneck: `..._base_256_<window>_K_baseline`

The regexes are in `length_mass_gain.py` and `plot_window_trend.py`.

Things to know about the data:
- The loader zero-means each window, so the encoder never sees mean magnitude. That is why the photometric-stats control is strong and why `--with-stats` exists.
- `LOGMBH == 0` / err `-1` are catalog "not measured" sentinels.
- Read IDs as strings. 18-digit PS1 IDs do not survive a float64 round trip.

## Model architecture (big picture)

- `presentation/pipelines/steps/model_design.py`: `build_model(params)` chooses `arch='zero'` (`src/models/astromer_0.py`, the original Astromer 1) or `arch='base'` (`src/models/astromer_1.py`, Astromer 2). `load_pt_model(dir)` rebuilds the model from `<dir>/config.toml` (the saved argparse namespace) and loads `<dir>/out.weights.h5`. Any new model hyperparameter must be added to the pretrain argparse, read in `build_model` (with a default for old configs that lack the key), and passed through `get_ASTROMER` → `Encoder` → `AttentionBlock`.
- `src/layers/encoders.py` `Encoder`: a Dense input projection plus a time positional encoding (`PositionalEncoder2`), followed by N `AttentionBlock`s. Called with `z_by_layer=True`, it returns a list of per-layer outputs, which the classifiers and `--pool concat` use.
- `src/layers/attblock.py` `AttentionBlock`: Pre-LN. It uses `HeadAttentionMulti` (standard) by default. When `use_cache=True` it uses `SimpleHeadAttentionMultiLatent` (MHLA, `src/layers/attention.py`), which compresses x to a shared `c_kv` with `latent_dim` dimensions. K is `c_kv` itself, shared across heads. V is up-projected from `c_kv`. Q is projected to `latent_dim` per head. The flag `--use-kv-cache` is what turns on MHLA, and it requires `--latent-dim`. Under `training=True` the block bypasses the cache logic. The inference path keeps a recurrent KV cache, so call `encoder.reset_caches()` between batches.
- Masking: `mask_in` = 1 marks masked or padded steps. Attention adds `m_alpha * mask` to the logits (`mask_format` K/Q/QK). With `trainable_mask`, `AddMSKToken` replaces masked inputs with a learned token. Pretraining loss is masked RMSE over `mask_out` (`src/losses/rmse.py`), optionally error-weighted (`--correct-loss`).
- Mixed precision: `pretrain.py` sets the global `mixed_bfloat16` policy. Layers explicitly cast tensors to matching dtypes, and `CustomModel.call` casts outputs to float32. Keep those casts when you edit layers.

## Data loading

`src/data/loaders.py` `get_loader` accepts:
- a directory of TFRecord `.record` shards, laid out as `<root>/{train,validation|val,test}/` with a `config.toml` that describes the SequenceExample schema (see `src/data/record.py`), or
- a directory or list of WebDataset-style `.tar.gz` shards (`lightcurves-*.tar.gz`, tensors stored with torch). This is why `pretrain.py` imports `torch` before `tensorflow`.

The pipeline is: filter → `to_windows` (split into `window_size` chunks, or randomly sample one with `sampling`) → normalize → mask (`probed`/`rs`/`same` fractions) → `padded_batch` → `format_inp_astromer`, which yields `({'input','times','mask_in'}, {'target','mask_out','w_error','lcid'})`. Classification mode (and embedding extraction) uses `probed=1, random=0, same=0`.

## Training loop and checkpoints

`src/training/utils.py` `train()` is a custom loop (not `model.fit`) with early stopping and TensorBoard logging. It writes these to the experiment dir:
- `out.weights.h5` and `best/out.weights.h5` for the best validation RMSE
- `latest/out.weights.h5` and `latest/optimizer_state.pkl` for resuming

All writes go to a temp file followed by `os.replace` so that a cluster preemption cannot leave a truncated file. Keep that pattern. On resume, the best weights are copied forward into the new trial directory.
