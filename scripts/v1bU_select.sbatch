#!/bin/bash
#SBATCH --job-name=v1bU_select
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# V1b-U-noanc prep (Yotam 11-25), CPU, after the A′-uniform report: (0) refuse unless A′-uniform passed A′'s pass test
# (compare.json pass_test); (1) spatial cue reduction with the V1b PCA rule (code/v1b_pca.py: centre -> PCA on the z-scored
# embedding of the paired training scenes -> k95, raised if any of the 4 probes loses > 10%; L2 in the trainer);
# (2) ESS step (stage 1, sigma only) on those cues with V1b A′'s spectral cues; (3) <out>/selected.json: spectral sigma /
# temp = V1b A′ (recomputed spectral sigma must reproduce it), spatial sigma = this ESS step, spatial temp = A′'s 0.7.
# Usage, from the SAC-SAR-SSL root: sbatch [--dependency=afterok:<rir_compare job>] scripts/v1bU_select.sbatch
set -euo pipefail
source ~/sac-ssamba/env.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
X=/scratch200/yotams2/SAC-SAR-SSL/exp; U=$X/rir_encoder/a1u_uniform; A=$X/v1b_sigma_tau_A1; O=$X/v1b_sigma_tau_U
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local}"
$PYTHON - "$U/compare.json" <<'P'
import json, sys
c = json.load(open(sys.argv[1]))
print(f"[check] A′-uniform pass test: {c['pass_test']}; verdict: {c['verdict']}; uniform / A′: {c['uniform_over_a1']}")
sys.exit(0 if c['pass_test'] else 1)
P
for f in $U/emb_paired_v2.npy $U/emb_stats.npz; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
mkdir -p $O/pca
(cd code && $PYTHON -u v1b_pca.py --rir-dir $U --out $O/pca)
(cd code && $PYTHON -u v1b_sigma_tau.py stage1 --rir-dir $U --out $O --pca-spec $A/pca/pca_spec.npz \
	--pca-spat $O/pca/pca_spat.npz --spat-cue cosine)
$PYTHON - "$A/selected.json" "$O/stage1.json" "$O/selected.json" "$O/pca/pca_spat.npz" <<'P'
import json, sys
import numpy as np
a, s1, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
k = int(np.load(sys.argv[4])['k'])
spec_a, spec_new = a['selected']['spec']['sigma'], s1['spec']['sigma']
assert abs(spec_new - spec_a) <= 1e-3 * spec_a, f'spectral sigma not reproduced: {spec_new} vs A′ {spec_a}'
sel = {'spec': dict(a['selected']['spec']), 'spat': {'sigma': s1['spat']['sigma'], 'temp': a['selected']['spat']['temp']}}
json.dump({'selected': sel, 'k_spat': k, 'note': 'V1b-U-noanc 11-25: spec = V1b A′; spat = A′-uniform embedding, '
		   'centre -> PCA (V1b rule) -> L2, sigma = ESS stage 1, temp = A′ (no sweep)', 'stage1': s1}, open(out, 'w'), indent=1)
print(f"[select] spec sigma {spec_new:.4f} (A′ {spec_a:.4f}, reproduced); spat sigma U {s1['spat']['sigma']:.4f} "
	  f"(top-3 {s1['spat']['top3']}, ESS {s1['spat']['avg_ess_at_best']:.1f} / target {s1['spat']['target_ess']:.1f}; "
	  f"A′ 0.5722); spat PCA k {k}; spat temp {sel['spat']['temp']} (A′)")
P
echo "$(date) done"
