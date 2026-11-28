# Invariance evals (post-sprint step 3, Yotam 11-28)

Frozen checkpoints, no training. Code: SAC-SAR-SSL code/invariance_eval.py, scripts/invariance_eval.sbatch (job 23817335, 53 min). Raw numbers and the full protocol: invariance.json.

**Clips.** No student trained on either set.
- **U:** 12,000 unpaired8 scenes, one clip per room, 113 WSJ0 speakers, first 4.112 s.
- **P:** paired v2 held-out rooms 12000–12499 × 4 speakers (2,000 clips).

**Embedding:** the mean over patches of JointSpatialDownstream(concat).patches, 768-d, the same one the linear-eval heads see. **Probes:** ridge, z-scored, 5-fold CV over 33 λ; SID uses one-hot ridge with argmax.

**Headline:** cross split minus a random split of the same size, averaged over both directions. For MAE a positive gap is worse; for accuracy a negative gap is worse.

## 1. Room estimation across speakers (U, spatial encoder, test MAE)
Speakers are split into two halves; the probe trains on one half and tests on the other.

| checkpoint | TDOA cross (gap %) | T60 cross (gap %) | DRR cross (gap %) | C50 cross (gap %) |
|---|---|---|---|---|
| A (swap, no SAC) | 1.414 (-0.5) | 0.060 (+2.0) | 1.543 (+0.0) | 0.541 (-0.1) |
| C 30 ep | 1.502 (+0.7) | 0.037 (-0.1) | 1.137 (+0.3) | 0.344 (-0.1) |
| V1b-A′ λ0.1 (10 ep) | 0.928 (-0.5) | 0.037 (+0.6) | 1.224 (-0.1) | 0.405 (-0.7) |
| V1b-HC λ0.1 (10 ep) | 0.481 (+0.2) | 0.043 (+0.1) | 1.232 (+0.1) | 0.408 (-0.6) |
| final noanc 30 ep s1 | 0.983 (+0.8) | 0.036 (-0.3) | 1.130 (+0.2) | 0.358 (-0.7) |
| final noanc 30 ep s2 | 0.982 (-0.4) | 0.036 (+0.2) | 1.143 (+0.4) | 0.365 (-0.6) |

- **No signal.** Every gap is within ±3.5% relative (spectral-encoder rows: in invariance.json). Room estimation from all six checkpoints already transfers across speakers, so this eval does not separate them.

## 2. SID across rooms (U, linear SID accuracy %)
Clips are split at the median T60 and at the median DRR; the probe trains on one half and tests on the other. The gap is the mean of the 4 splits.

| checkpoint | spectral: random | cross | gap | spatial (leakage): random | cross | gap |
|---|---|---|---|---|---|---|
| A (swap, no SAC) | 70.2 | 64.8 | -5.3 | 40.4 | 38.3 | -2.1 |
| C 30 ep | 86.4 | 84.4 | -2.0 | 30.6 | 27.8 | -2.9 |
| V1b-A′ λ0.1 (10 ep) | 88.3 | 87.1 | -1.2 | 25.5 | 22.1 | -3.4 |
| V1b-HC λ0.1 (10 ep) | 88.9 | 87.2 | -1.7 | 24.9 | 22.9 | -2.0 |
| final noanc 30 ep s1 | 90.4 | 89.5 | -1.0 | 16.9 | 14.8 | -2.1 |
| final noanc 30 ep s2 | 90.6 | 89.6 | -1.0 | 16.8 | 14.2 | -2.6 |

- **Spectral encoder:** A loses 5.3 points across rooms. C 30 ep loses 2.0, the V1b students 1.0 to 1.7.
  - Final noanc vs C 30 ep is 1.0 point, under the 1.5-point threshold: **a tie**.
  - A is ≥ 3.3 points worse than every other checkpoint, the only difference here above the threshold.
- **The T60 hi→lo split is the hardest:** A −11.8; C30 −4.5; final noanc −1.6 / −1.6.
- **Spatial-encoder SID (leakage)** falls from 40% (A) and 31% (C 30 ep) to 17% for the final noanc students (random split). Their spatial encoder holds less speaker information.

## 3. SID in seen vs unseen rooms (P, secondary)
Each room: renderings 0–1 train, 2–3 test (seen), vs rooms 12000–12249 train, 12250–12499 test (unseen).

| checkpoint | spectral seen | unseen | gap | spatial seen | unseen | gap |
|---|---|---|---|---|---|---|
| A (swap, no SAC) | 48.8 | 45.4 | -3.4 | 21.5 | 20.2 | -1.3 |
| C 30 ep | 65.7 | 63.0 | -2.7 | 12.7 | 14.2 | +1.5 |
| V1b-A′ λ0.1 (10 ep) | 70.9 | 67.4 | -3.5 | 9.6 | 9.9 | +0.3 |
| V1b-HC λ0.1 (10 ep) | 70.8 | 65.6 | -5.2 | 9.0 | 12.1 | +3.1 |
| final noanc 30 ep s1 | 77.0 | 72.8 | -4.2 | 5.6 | 6.1 | +0.5 |
| final noanc 30 ep s2 | 78.3 | 72.0 | -6.3 | 5.3 | 5.1 | -0.2 |

- Small sample: about 9 training clips per speaker, and every held-out utterance also occurs in training rooms. The gaps (−2.7 to −6.3) do not order the models consistently with eval 2. Not used for decisions.
