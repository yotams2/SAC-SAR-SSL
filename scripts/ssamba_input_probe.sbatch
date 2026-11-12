#!/bin/bash
#SBATCH --job-name=ssamba_inprobe
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# Why can't the spatial SSAMBA encoder learn TDOA even from random init (2.86-2.93 = predict-the-mean, SAR-SSL
# random init 0.50)? Every variant here is a RANDOM-INIT supervised run (R8, 20 epochs, 1 trial, lr 1e-3 and 1e-4,
# best by val), so no pretraining is involved: it isolates the input / tokenizer. Sequence = the real clip (1.04 s for
# TDOA, 4.112 s for T60), not tiled to 10 s. Tasks: TDOA (inter-channel phase, the cue that fails) then T60 (room decay,
# a different cue, so a fix for TDOA must not cost the room tasks). All TDOA variants run before any T60 one.
# Variants, most informative first:
#   sarssl_ctrl      SAR-SSL's own encoder (CNN stem + full-band frame tokens + Conformer) at this same budget: the target
#   reimframe_stem   SSAMBA backbone with SAR-SSL's whole front end (its STFT + norm, CNN stem, 256x1 frame tokens)
#   ipd16            explicit spatial channels (log|L|, ILD, cos IPD, sin IPD), SSAMBA's 16x16 patches
#   reimframe        SAR-SSL STFT Re/Im, frame tokens, linear embedding (no stem)
#   ipdframe         IPD/ILD channels, frame tokens
#   reim16           SAR-SSL STFT Re/Im (windowed, normalised), 16x16 patches: is it just window/scale?
#   wave16_short     today's input (unwindowed, peak-normalised, 16x16) without the 10-s tiling: is it the tiling?
# Each variant goes through scripts/ssamba_spatial_downstream.sbatch (run as a plain script), which reports to the
# results branch and skips finished training on a requeue. A variant that fails does not stop the others.
#
# Usage, from the SAC-SAR-SSL root: [PROBE_TASKS="TDOA T60"] sbatch scripts/ssamba_input_probe.sbatch [variant ...]
#   (default: both tasks, all variants)
set -uo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO_ROOT"
COMMON=(--my-tmp --ds-lr 1e-3 1e-4 --ds-nepoch 20 --ssamba-fill none)
declare -A V=(
  [sarssl_ctrl]="--upstream sarssl"
  [reimframe_stem]="--ssamba-input reim --ssamba-patch frame --ssamba-stem"
  [ipd16]="--ssamba-input ipd --ssamba-patch 16x16"
  [reimframe]="--ssamba-input reim --ssamba-patch frame"
  [ipdframe]="--ssamba-input ipd --ssamba-patch frame"
  [reim16]="--ssamba-input reim --ssamba-patch 16x16"
  [wave16_short]="--ssamba-input wave --ssamba-patch 16x16"
)
ORDER=(sarssl_ctrl reimframe_stem ipd16 reimframe ipdframe reim16 wave16_short)
[ $# -gt 0 ] && ORDER=("$@")
TASKS=(${PROBE_TASKS:-TDOA T60})
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local}: tasks ${TASKS[*]}, variants ${ORDER[*]}"
for task in "${TASKS[@]}"; do
  for v in "${ORDER[@]}"; do
    [ -n "${V[$v]:-}" ] || { echo "unknown variant $v"; continue; }
    echo "===== $(date) ${task} variant $v: ${V[$v]}"
    # shellcheck disable=SC2086
    bash scripts/ssamba_spatial_downstream.sbatch "$task" random "inprobe_${v}" 8 scratchLOW "${COMMON[@]}" ${V[$v]} \
      || echo "===== ${task} variant $v FAILED (exit $?)"
  done
done
echo "$(date) all variants done"
