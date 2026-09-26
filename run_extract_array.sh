#!/bin/bash
# Extract embeddings for every pretrained model, one SLURM array task per model.
#
#   1. list the models, one pretraining directory per line:
#        find presentation/results -type d -name pretraining | sort > models.txt
#        wc -l models.txt                                # -> N
#   2. submit (array indices are 0-based line numbers):
#        sbatch --array=0-$(( $(wc -l < models.txt) - 1 ))%10 run_extract_array.sh
#      (%10 = at most 10 at once; drop or change it to suit the queue)
#   3. when all tasks finish:
#        python fit_regressor.py --emb emb_agn/*.npz --index "$DATA/test/lc_index.csv" ...
#
# A task whose output already exists exits immediately, so resubmitting the same
# array after failures only redoes the missing models. Check with:
#        ls emb_agn/*.npz | wc -l ;  grep -l -i "error\|killed" logs/extract_*.err
#
# Override any of these at submit time, e.g.  sbatch --export=ALL,BS=64 ...
#   MODELS  file listing pretraining dirs     (default models.txt)
#   DATA    records root containing test/      (default data/records/agn_test/fold_0)
#   OUTDIR  where the .npz files go            (default emb_agn)
#   POOL    last | concat                       (default last; concat needs ~7x RAM)
#   BS      batch size                          (default 128; lower it if a large
#                                                window is OOM-killed)
#   DRY_RUN=1  print the command instead of running it
#
# Edit the partition / time / memory / environment lines for your cluster.
#SBATCH --job-name=agn_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err

set -euo pipefail

MODELS=${MODELS:-models.txt}
DATA=${DATA:-data/records/agn_test/fold_0}
OUTDIR=${OUTDIR:-emb_agn}
POOL=${POOL:-last}
BS=${BS:-128}
TASK=${SLURM_ARRAY_TASK_ID:?"submit with sbatch --array=... (or set SLURM_ARRAY_TASK_ID)"}

# --- environment: edit to match how you normally run the TF code ------------
# module load python/3.10 cuda/12.2 cudnn
# source activate astromer
# ----------------------------------------------------------------------------

mkdir -p "$OUTDIR" logs

PT=$(sed -n "$((TASK + 1))p" "$MODELS")
if [[ -z "$PT" ]]; then
    echo "task $TASK: no line $((TASK + 1)) in $MODELS" >&2
    exit 1
fi
if [[ ! -d "$PT" ]]; then
    echo "task $TASK: not a directory: $PT" >&2
    exit 1
fi

# name the output after the experiment directory (.../<exp>/<timestamp>/pretraining)
EXP=$(basename "$(dirname "$(dirname "$PT")")")
[[ "$EXP" == "." || "$EXP" == "/" || -z "$EXP" ]] && EXP=$(basename "$PT")
OUT="$OUTDIR/emb_$(printf '%03d' "$TASK")_${EXP}.npz"

if [[ -s "$OUT" ]]; then
    echo "task $TASK: $OUT exists, skipping"
    exit 0
fi

CMD=(python -m extract_embeddings --pt-model "$PT" --data "$DATA" --subset test
     --pool "$POOL" --bs "$BS" --gpu "${CUDA_VISIBLE_DEVICES:-0}" --out "$OUT.tmp.npz")

echo "task $TASK  model $PT"
echo "  -> $OUT"
echo "  ${CMD[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

start=$(date +%s)
"${CMD[@]}"
mv "$OUT.tmp.npz" "$OUT"         # only a complete file gets the final name
echo "task $TASK done in $(( $(date +%s) - start ))s"
