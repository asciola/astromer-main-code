#!/bin/bash
#SBATCH --job-name=agn_mlp
#SBATCH --partition=seas_gpu        # <- a GPU partition from spart
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/agn_mlp_%x_%j.out

# same environment setup as your other jobs, e.g.
module purge
# Activate the uv-managed environment (see pyproject.toml)
source /n/home07/asciola/astromer-main-code/activate_env.sh agn
export PYTHONUNBUFFERED=1

# stop right away if PyTorch can't see the GPU, rather than silently
# running for hours on the CPU
python -c "import torch, sys; ok = torch.cuda.is_available(); \
print('torch', torch.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else ''); \
sys.exit(0 if ok else 1)" || { echo 'no GPU visible to torch'; exit 1; }

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
OUT=${OUT:-agn_mlp$SUFFIX}
python fit_regressor.py --emb $EMB/*.npz --index $IDX --targets $TARGETS \
    --model mlp --mlp-config agn_tune/mlp_config.json \
    ${EXTRA:-} --out-dir $OUT
