#!/bin/bash
# Rest-frame window size / density experiment: the same sources in three window
# sets (see restframe_windows.py --cells and plot_rest_density.py)
#   p200_d300      200 points over 300 rest-frame days (contiguous)
#   p400_d600      400 points over 600 days (contiguous): same density, twice the time
#   p400_d600_t2   those 400 points thinned to every 2nd: same time, half the density
# Every set is embedded by every window-200 model; the 400-point set also by every
# window-400 model. Standard attention at window 200 is the reference in all three,
# so on p400_d600 it embeds the 400 points as two 200-point windows, averaged (each
# once: extract_embeddings.py skips the clamped repeat get_windows produces).
# Each set is then fitted for m and e with ridge and the MLP.
#
# Run on a LOGIN node, from anywhere:
#     bash path/to/run_rest_density.sh
# Like run_rest_duration.sh it changes to the repo root (the directory holding
# fit_regressor.py, extract_embeddings.py, restframe_windows.py, presentation/results
# and agn_tune/) and sources activate_env.sh itself. Outputs land there:
# models_w200.txt, models_w400.txt, models_w200_400.txt, logs/, emb_rc_<cell>/,
# agn_ridge_rc_<cell>/, agn_mlp_rc_<cell>/.
# It
#   1. writes $ROOT/agn_rest_cells/<cell>/ with restframe_windows.py --cells (skipped
#      if every cell's lc_index.csv exists; needs $ROOT/agn_test_rest from
#      restframe_records.py),
#   2. lists the window-200 and window-400 models, one per experiment (its latest
#      trial dir that has out.weights.h5),
#   3. per cell submits an extraction array and, once it has fully succeeded, a
#      ridge and an MLP fit with --targets m,e.
#
# Overrides (environment):
#   ROOT       data root            (default /n/holylabs/protopapas_lab/Lab/asciola/data)
#   DRY_RUN=1  print the sbatch commands instead of submitting
#   ACTIVATE   env script to source (default /n/home07/asciola/astromer-main-code/activate_env.sh)
#   SBATCH_EXCLUDE=<nodes>   read by sbatch itself: keep every job off those nodes
set -euo pipefail

ACTIVATE=${ACTIVATE:-/n/home07/asciola/astromer-main-code/activate_env.sh}
set +u                       # venv activate scripts are not nounset-clean
# shellcheck disable=SC1090
source "$ACTIVATE" tf        # explicit arg: a bare source would pass on this script's $1
set -u

ROOT=${ROOT:-/n/holylabs/protopapas_lab/Lab/asciola/data}
[[ "$ROOT" == /* ]] || ROOT=$PWD/$ROOT                  # still valid after the cd below
CELLDIR=$ROOT/agn_rest_cells
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)     # where the run_*.sh scripts are
SPECS=(200@300 400@600 400@600/2)                      # restframe_windows.py --cells
CELLS=(p200_d300 p400_d600 p400_d600_t2)               # the directories they become
CELL_MODELS=(models_w200.txt models_w200_400.txt models_w200.txt)

# ---- 0. work from the repo root --------------------------------------------------
NEED=(fit_regressor.py extract_embeddings.py restframe_windows.py presentation/results
      agn_tune/mlp_config.json)
has_all() { local f; for f in "${NEED[@]}"; do [[ -e "$1/$f" ]] || return 1; done; }
for d in "$PWD" "$HERE/.." "$HERE"; do
    if has_all "$d"; then cd "$d"; break; fi
done
if ! has_all .; then
    GUESS=$(cd "$HERE/.." && pwd)
    echo "no directory has all of: ${NEED[*]}" >&2
    echo "(looked in $PWD, $GUESS and $HERE). Missing from $GUESS:" >&2
    for f in "${NEED[@]}"; do [[ -e "$GUESS/$f" ]] || echo "  $f" >&2; done
    exit 1
fi
echo "working in $PWD"
# window-200 models must not count a 400-point window's clamped repeat twice
if ! grep -q 'repeated windows skipped' extract_embeddings.py; then
    echo "extract_embeddings.py here predates the repeated-window fix; update it first" >&2
    exit 1
fi

submit() {   # sbatch, or print it under DRY_RUN; echoes the job id
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN: $*" >&2
        echo "DRYJOB"
    else
        sbatch --parsable "$@"
    fi
}

# ---- 1. window sets ------------------------------------------------------------
missing=0
for c in "${CELLS[@]}"; do [[ -s "$CELLDIR/$c/fold_0/test/lc_index.csv" ]] || missing=1; done
if (( missing )); then
    python -c "import tensorflow" 2>/dev/null \
        || { echo "no TensorFlow after sourcing $ACTIVATE; run 'uv sync' in the repo" >&2; exit 1; }
    python restframe_windows.py --src "$ROOT/agn_test_rest/fold_0/test" --dst "$CELLDIR" \
        --cells "${SPECS[@]}"
fi

# ---- 2. models -----------------------------------------------------------------
# One line per EXPERIMENT: <exp>/<timestamp>/pretraining dirs that hold weights
# (a resumed run leaves its earlier trial dirs behind, some without
# out.weights.h5), and of those the latest timestamp. Paths sort by timestamp
# within an experiment, so awk keeps the last one seen.
list_models() {   # window size, output file
    find presentation/results -path '*/pretraining/out.weights.h5' \
        | sed 's#/out\.weights\.h5$##' \
        | { grep -E "(_K_01_[0-9]+_$1|_base_[0-9]+_$1_K_baseline)/" || true; } \
        | sort | awk -F/ '{ last[$(NF-2)] = $0 } END { for (e in last) print last[e] }' \
        | sort > "$2"   # no match must reach the check below, not trip pipefail
    if ! grep -q "_base_[0-9]*_$1_K_baseline/" "$2"; then
        echo "no window-$1 standard-attention model (with out.weights.h5) in presentation/results" >&2
        exit 1
    fi
    echo "$2: $(wc -l < "$2" | tr -d " ") window-$1 models; standard attention: $(grep "_base_[0-9]*_$1_K_baseline/" "$2")"
}
list_models 200 models_w200.txt
list_models 400 models_w400.txt
cat models_w200.txt models_w400.txt > models_w200_400.txt
mkdir -p logs

# ---- 3. extraction, then fits, per cell ----------------------------------------
for k in "${!CELLS[@]}"; do
    C=${CELLS[$k]}; M=${CELL_MODELS[$k]}; N=$(wc -l < "$M" | tr -d " ")
    IDXC=$CELLDIR/$C/fold_0/test/lc_index.csv
    EXT=$(submit --array=0-$((N - 1)) \
        --export=ALL,MODELS=$M,DATA=$CELLDIR/$C/fold_0,OUTDIR=emb_rc_$C \
        "$HERE/run_extract_array.sh")
    # IDX/EMB/TARGETS/OUT go through the environment: --export splits on commas
    RID=$(IDX=$IDXC EMB=emb_rc_$C TARGETS=m,e OUT=agn_ridge_rc_$C \
        submit --dependency=afterok:$EXT --export=ALL "$HERE/run_ridge.sh")
    MLP=$(IDX=$IDXC EMB=emb_rc_$C TARGETS=m,e OUT=agn_mlp_rc_$C \
        submit --dependency=afterok:$EXT --export=ALL "$HERE/run_agn_mlp.sh")
    echo "$C ($N models): extraction $EXT -> ridge $RID, mlp $MLP"
done

runs() { for C in "${CELLS[@]}"; do printf ' %s:agn_%s_rc_%s/predictions.csv' "$C" "$1" "$C"; done; }
echo
echo "when the fits are done (in a shell where you ran: source $ACTIVATE):"
echo "  python plot_rest_density.py --runs$(runs mlp) --head mlp --out-dir rest_density"
echo "  python plot_rest_density.py --runs$(runs ridge) --head ridge --out-dir rest_density"
