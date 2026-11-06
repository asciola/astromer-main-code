#!/bin/bash
#SBATCH --job-name=agn_ridge_stats
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ridge_stats_%j.out

# same environment setup you use interactively, e.g.
# module load python

module purge
source ~/anaconda3/etc/profile.d/conda.sh
conda activate astromer
export PYTHONUNBUFFERED=1


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # let numpy use the cores you asked for

# FRAME=rest uses the rest-frame records written by restframe_records.py and drops
# redshift as a target (the rest-frame time axis encodes it); default is observed.
ROOT=/n/holylabs/protopapas_lab/Lab/asciola/data
if [[ "${FRAME:-observed}" == "rest" ]]; then
    IDX=$ROOT/agn_test_rest/fold_0/test/lc_index.csv; EMB=emb_agn_rest; TARGETS=m,e; SUFFIX=_rest
else
    IDX=$ROOT/agn_test/fold_0/test/lc_index.csv; EMB=emb_agn; TARGETS=z,m,e; SUFFIX=
fi
python fit_regressor.py --emb $EMB/*.npz --index $IDX --targets $TARGETS --with-stats \
    --out-dir agn_ridge_stats$SUFFIX
