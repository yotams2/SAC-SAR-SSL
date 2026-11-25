#!/bin/bash
#SBATCH --job-name=v1b_hc_select
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-public-pool
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# V1b-HC (Yotam 11-25), CPU: (1) hand-crafted spatial cue file (code/v1b_hc_cues.py), (2) the V1b ESS step (stage 1,
# sigma only) on it with V1b A′'s spectral cues, (3) <out>/selected.json = spectral sigma / temp of V1b A′ (unchanged),
# spatial sigma = this ESS step, spatial temp = A′'s (0.7; no tau sweep, so tau does not confound the comparison).
# Check: the recomputed spectral sigma must reproduce A′'s (same cues, same RNG), else stop.
# Usage, from the SAC-SAR-SSL root: sbatch scripts/v1b_hc_select.sbatch [<launch after: 1>]
set -euo pipefail
source ~/sac-ssamba/env.sh
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}; cd "$ROOT"
X=/scratch200/yotams2/SAC-SAR-SSL/exp; A=$X/v1b_sigma_tau_A1; O=$X/v1b_sigma_tau_HC
mkdir -p $O
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local}"
[ -f $O/hc_cues.npy ] || (cd code && $PYTHON -u v1b_hc_cues.py --stats-json $A/cue_check/sac_cue_check.json --out $O/hc_cues)
(cd code && $PYTHON -u v1b_sigma_tau.py stage1 --rir-dir $X/rir_encoder/a1_design1 --out $O --pca-spec $A/pca/pca_spec.npz \
	--cue-file-spat $O/hc_cues.npy)
$PYTHON - "$A/selected.json" "$O/stage1.json" "$O/selected.json" <<'P'
import json, sys
a, s1, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
spec_a, spec_new = a['selected']['spec']['sigma'], s1['spec']['sigma']
assert abs(spec_new - spec_a) <= 1e-3 * spec_a, f'spectral sigma not reproduced: {spec_new} vs A′ {spec_a}'
sel = {'spec': dict(a['selected']['spec']), 'spat': {'sigma': s1['spat']['sigma'], 'temp': a['selected']['spat']['temp']}}
json.dump({'selected': sel, 'note': 'V1b-HC 11-25: spec = V1b A′ selection; spat sigma = ESS stage 1 on the hand-crafted cues, '
		   'spat temp = A′ (no sweep); old C + SAC sigma 0.613 / temp 0.15', 'stage1': s1}, open(out, 'w'), indent=1)
print(f"[select] spec sigma {spec_new:.4f} (A′ {spec_a:.4f}, reproduced); spat sigma HC {s1['spat']['sigma']:.4f} "
	  f"(top-3 {s1['spat']['top3']}, ESS {s1['spat']['avg_ess_at_best']:.1f} / target {s1['spat']['target_ess']:.1f}) "
	  f"vs old C + SAC 0.613; spat temp {sel['spat']['temp']} (A′)")
P
if [ "${1:-}" = 1 ]; then bash scripts/launch_v1bhc.sh; fi
echo "$(date) done"
