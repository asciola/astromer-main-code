# Installing AstroMer

This guide walks you through setting up the environment to run AstroMer.

Dependencies are managed with [**uv**](https://docs.astral.sh/uv/). The exact
versions are pinned in `uv.lock`, so everyone gets a byte-identical
environment.

---

## 1. Prerequisites

Install `uv`. It is a single static binary and needs no Python of its own:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(or `brew install uv`, or see the
[install docs](https://docs.astral.sh/uv/getting-started/installation/) for
other options). Make sure `~/.local/bin` is on your `PATH`.

You do **not** need conda, and you do not need to install Python yourself --
`uv` downloads the right interpreter automatically.

---

## 2. Installation Steps

### Step 1: Clone the repository

```bash
git clone https://github.com/cridonoso/astromer.git
cd astromer
```

### Step 2: Create the environment

From the repository root:

```bash
uv sync
```

That reads `pyproject.toml` / `uv.lock`, downloads CPython 3.11 if needed, and
creates `.venv` with every pinned dependency, including the CUDA 11 libraries
TensorFlow 2.14 needs for the GPU.

Python 3.11 is the highest version TensorFlow 2.14 ships wheels for; see
`CLAUDE.md` for why moving past it is a Keras 3 migration rather than a
version bump.

### Step 3: Run something

Either activate the environment the usual way:

```bash
source .venv/bin/activate
python -m presentation.pipelines.pipeline_0.pretrain --help
```

or skip activation entirely and let `uv` handle it per-command:

```bash
uv run python -m presentation.pipelines.pipeline_0.pretrain --help
```

Always run from the repository root: imports are package-absolute (`src...`,
`presentation.pipelines...`) and resolve as namespace packages.

You're all set! 🎉

---

## 3. Optional extras

The default install is what the cluster jobs need. Two groups are optional:

```bash
uv sync --extra notebook   # Jupyter Lab / notebook
uv sync --extra dev        # pytest
```

---

## 4. The second environment

`fit_regressor.py --model mlp` can run its head on a GPU, which needs a CUDA 12
PyTorch build. That cannot coexist with TensorFlow 2.14 (which requires
numpy < 2 and CUDA 11), so it lives in its own project:

```bash
uv sync --project envs/agn-torch
uv run --project envs/agn-torch python fit_regressor.py ...
```

Everything else -- pretraining, finetuning, embedding extraction, record
conversion, plotting, and the ridge regressors -- uses the root environment.

---

## 5. Adding or changing a dependency

```bash
uv add polars                            # root environment
uv add --project envs/agn-torch polars   # GPU regression environment
```

`uv` updates `pyproject.toml` and `uv.lock`. Commit both.
