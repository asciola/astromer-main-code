#!/bin/bash
# Paired rest-frame window-duration experiment: the same sources, one 200-point
# window each, at several rest-frame durations (see restframe_windows.py), embedded
# by every window-200 model and fitted for m and e (see plot_rest_duration.py).
#
# Run on a LOGIN node, from the directory you normally submit from (the one with
# fit_regressor.py, extract_embeddings.py, presentation/results and agn_tune/):
#     bash path/to/run_rest_duration.sh
# It sources activate_env.sh (the uv TF env) itself; nothing needs activating first.
# It
#   1. writes $ROOT/agn_rest_windows/D<days>/ with restframe_windows.py (skipped if
#      the last duration's lc_index.csv already exists; needs the rest-frame records
#      from restframe_records.py in $ROOT/agn_test_rest),
#   2. lists the window-200 models (MHLA and standard attention) in models_w200.txt,
#   3. for every duration submits an extraction array into emb_rw<days>/ and, once
#      the whole array has succeeded, a ridge fit (agn_ridge_rw<days>/) and an MLP
#      fit (agn_mlp_rw<days>/), both with --targets m,e.
# When the fits finish it is the printed plot_rest_duration.py command that remains.
#
# Overrides (environment):
#   ROOT       data root            (default /n/holylabs/protopapas_lab/Lab/asciola/data)
#   DURATIONS  rest-frame days      (default "200 300 400")
#   DRY_RUN=1  print the sbatch commands instead of submitting
#   ACTIVATE   env script to source (default /n/home07/asciola/astromer-main-code/activate_env.sh)
set -euo pipefail

ACTIVATE=${ACTIVATE:-/n/home07/asciola/astromer-main-code/activate_env.sh}
set +u                       # venv activate scripts are not nounset-clean
# shellcheck disable=SC1090
source "$ACTIVATE" tf        # explicit arg: a bare source would pass on this script's $1
set -u

ROOT=${ROOT:-/n/holylabs/protopapas_lab/Lab/asciola/data}
DURATIONS=${DURATIONS:-"200 300 400"}
WIN=$ROOT/agn_rest_windows
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)     # where the run_*.sh scripts are
read -r -a DS <<< "$DURATIONS"

submit() {   # sbatch, or print it under DRY_RUN; echoes the job id
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN: $*" >&2
        echo "DRYJOB"
    else
        sbatch --parsable "$@"
    fi
}

# ---- 1. window sets ------------------------------------------------------------
if [[ ! -s "$WIN/D${DS[${#DS[@]}-1]}/fold_0/test/lc_index.csv" ]]; then
    python -c "import tensorflow" 2>/dev/null \
        || { echo "no TensorFlow after sourcing $ACTIVATE; run 'uv sync' in the repo" >&2; exit 1; }
    python restframe_windows.py --src "$ROOT/agn_test_rest/fold_0/test" --dst "$WIN" \
        --durations "${DS[@]}"
fi

# ---- 2. window-200 models ------------------------------------------------------
find presentation/results -type d -name pretraining \
    | { grep -E '(_K_01_[0-9]+_200|_base_[0-9]+_200_K_baseline)/' || true; } \
    | sort > models_w200.txt   # no match must reach the check below, not trip pipefail
N=$(wc -l < models_w200.txt)
if ! grep -q '_base_[0-9]*_200_K_baseline/' models_w200.txt; then
    echo "no window-200 standard-attention model in presentation/results" >&2
    exit 1
fi
echo "models_w200.txt: $N window-200 models"
mkdir -p logs

# ---- 3. extraction, then fits, per duration ------------------------------------
for D in "${DS[@]}"; do
    IDXD=$WIN/D$D/fold_0/test/lc_index.csv
    EXT=$(submit --array=0-$((N - 1)) \
        --export=ALL,MODELS=models_w200.txt,DATA=$WIN/D$D/fold_0,OUTDIR=emb_rw$D \
        "$HERE/run_extract_array.sh")
    # IDX/EMB/TARGETS/OUT go through the environment: --export splits on commas
    RID=$(IDX=$IDXD EMB=emb_rw$D TARGETS=m,e OUT=agn_ridge_rw$D \
        submit --dependency=afterok:$EXT --export=ALL "$HERE/run_ridge.sh")
    MLP=$(IDX=$IDXD EMB=emb_rw$D TARGETS=m,e OUT=agn_mlp_rw$D \
        submit --dependency=afterok:$EXT --export=ALL "$HERE/run_agn_mlp.sh")
    echo "D=$D d: extraction $EXT -> ridge $RID, mlp $MLP"
done

runs() { for D in "${DS[@]}"; do printf ' %s:agn_%s_rw%s/predictions.csv' "$D" "$1" "$D"; done; }
echo
echo "when the fits are done (in a shell where you ran: source $ACTIVATE):"
echo "  python plot_rest_duration.py --runs$(runs mlp) --head mlp --out-dir rest_duration"
echo "  python plot_rest_duration.py --runs$(runs ridge) --head ridge --out-dir rest_duration"
