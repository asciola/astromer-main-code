#!/bin/bash
#SBATCH --job-name=agn_ridge
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ridge_%j.out

# same environment setup you use interactively, e.g.
# module load python

module purge
# Activate the uv-managed environment (see pyproject.toml)
source /n/home07/asciola/astromer-main-code/activate_env.sh
export PYTHONUNBUFFERED=1


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # let numpy use the cores you asked for

# FRAME=rest uses the rest-frame records written by restframe_records.py and drops
# redshift as a target (the rest-frame time axis encodes it); default is observed.
ROOT=/n/holylabs/protopapas_lab/Lab/asciola/data
# IDX, EMB, TARGETS and OUT already set in the environment win (run_rest_duration.sh
# uses that; set them in the shell, not via --export, which splits on commas).
if [[ "${FRAME:-observed}" == "rest" ]]; then
    IDX=${IDX:-$ROOT/agn_test_rest/fold_0/test/lc_index.csv}; EMB=${EMB:-emb_agn_rest}
    TARGETS=${TARGETS:-m,e}; SUFFIX=_rest
else
    IDX=${IDX:-$ROOT/agn_test/fold_0/test/lc_index.csv}; EMB=${EMB:-emb_agn}
    TARGETS=${TARGETS:-z,m,e}; SUFFIX=
fi
python fit_regressor.py --emb $EMB/*.npz --index $IDX --targets $TARGETS \
    --out-dir ${OUT:-agn_ridge$SUFFIX}
