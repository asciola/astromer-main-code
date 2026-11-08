#!/usr/bin/env bash
# Activate a uv-managed environment for this repo. This replaces the old
#     source ~/anaconda3/etc/profile.d/conda.sh
#     conda activate astromer
# two-liner. It must be SOURCED, not executed:
#
#     source /n/home07/asciola/astromer-main-code/activate_env.sh        # TF env
#     source /n/home07/asciola/astromer-main-code/activate_env.sh agn    # GPU torch env
#
#   tf  (default) -> .venv               pretraining, finetuning, embeddings,
#                                        record conversion, plotting
#   agn           -> envs/agn-torch/.venv  fit_regressor.py --model mlp (GPU)
#
# Create or update either with, from the repo root:
#     uv sync                           # tf
#     uv sync --project envs/agn-torch  # agn

_ae_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# On the FASRC cluster, uv's cache and its downloaded interpreters live on lab
# storage: /n/home07 is at its quota, and both are large and freely
# regenerable. Only do this where that storage actually exists, so the script
# still works on a laptop. Values already set in the environment win.
_ae_lab="/n/holylabs/protopapas_lab/Lab/asciola"
if [ -d "$_ae_lab" ]; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$_ae_lab/uv-cache}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$_ae_lab/uv-python}"
fi
unset _ae_lab
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

case "${1:-tf}" in
    tf|astromer)        _ae_venv="$_ae_root/.venv" ;         _ae_sync="uv sync" ;;
    agn|torch|agn-torch) _ae_venv="$_ae_root/envs/agn-torch/.venv"; _ae_sync="uv sync --project envs/agn-torch" ;;
    *)
        echo "activate_env.sh: unknown environment '$1' (expected 'tf' or 'agn')" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

if [ ! -f "$_ae_venv/bin/activate" ]; then
    echo "activate_env.sh: no environment at $_ae_venv" >&2
    echo "  create it from the repo root with:  $_ae_sync" >&2
    unset _ae_root _ae_venv _ae_sync
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
. "$_ae_venv/bin/activate"
unset _ae_root _ae_venv _ae_sync
