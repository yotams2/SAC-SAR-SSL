# Teacher representation analysis (28 Nov 2026)

Requested by Yotam on 28 Nov: steps 1–4. Step 5 is post-sprint.
- Code: SAC-SAR-SSL `code/teacher_geometry.py` (steps 1–3, CPU job 23815842) and `code/teacher_student_cka.py` (step 4, GPU at nice 100, job 23815779, L40S, 3 min).
- Raw numbers: [geometry.json](geometry.json) and [cka.json](cka.json).
- Plots: [sv_spectrum.png](sv_spectrum.png) and [distance_shares.png](distance_shares.png).

## Bottom line

- **A′ is not collapsed, and its distance is mostly not about the room labels.** All six labels are linearly decodable (T60 R² 0.99, C50 0.97, DRR 0.87, TDOA 0.84). Yet the directions along which the cue varies with the labels carry only **14% of the held-out pair distance**, and the reverberation span (T60, DRR, C50) carries 9%. The other **86% is residual**. So the expectation holds: the reverberation subspace has a small share and the residual dominates.
- The distance between two A′ cues is the quantity V1b's Gaussian kernel acts on. It tracks the labels only weakly: Spearman with |ΔT60| is 0.35, with |ΔDRR| 0.10, with |ΔTDOA| 0.20. Distance measured inside each label's own direction tracks them well (T60 0.95, C50 0.74, TDOA 0.62).
- **The V1b spatial students copy that geometry.** Linear CKA with A′ is 0.69 for V1b A′ λ0.1 and 0.71 for noanc 30 ep s2, against only 0.37–0.38 with oracle B. C v2 10 ep, which had no room teacher, matches oracle B more closely (0.59). One possible explanation for why V1b's T60 / DRR do not beat C, though not tested here: the kernel spends its weight on label-orthogonal A′ structure.
- **The speech teacher is the opposite case.** Its distance is speaker-dominated: a 14-dim speaker subspace, 13% of the dimensions, carries 55% of the distance, and the cue gives 94.6% linear SID. The V1b spectral students match it closely (CKA 0.86–0.93) and carry no room structure (CKA ≤ 0.01 with A′, HC and B). C v2's spectral encoder carries a lot of room structure (CKA 0.71 with oracle B at scene level).

## Setup

**Teachers.** Each is the exact V1b cue, rebuilt from cached files as `_swap_pretrain_v1.set_v1_targets` builds it:

| Teacher | Before | After = the V1b cue |
|---|---|---|
| A′ (`rir_encoder/a1_design1`) | 768-d RIR-encoder embedding (attention-pool half + late-mean half), z-scored with `emb_stats.npz` | centred, projected on `pca_spat.npz` (k = 768, a rotation), L2-normalised |
| speech (`v1_teacher_amba_s1`) | tokens z-scored per dimension, mean over the 128 tokens | `pca_spec.npz` (k = 108), L2-normalised |
| oracle B | 6 z-scored simulator parameters, Euclidean distance | same |
| HC (`v1b_sigma_tau_HC/hc_cues.npy`) | 5 hand-crafted C + SAC cues per scene, Euclidean distance | same |

**Scenes.** The same scenes as the cue check:
- *paired*: fit on paired v2 [0, 12000), score on the 500 held-out scenes [12000, 12500), using all 124,750 pairs. This is the only set HC exists for, so it is the main comparison.
- *cue set*: A′ and B are also scored on 3,000 held-out rooms (paired + unpaired8), using 10k random pairs as in the cue check. The results match the paired set closely (shares and Spearman values to within 0.01, alignment ratio 0.78 vs 0.82), so the paired numbers are the ones reported below.
- *speech*: the 1,884 utterances of the held-out scenes are scored; the other 10,616 are used for fitting. There are 113 speakers and every held-out speaker is also in the fit set.

**What was not cached, so not computed:**
- HC cues on unpaired8.
- A random-init speech teacher.
- Random-init references for volume and source–mic distance.
- SC2 labels: these clips are WSJ0 read speech, and SID is the only speech label available for them.

## 1. Collapse / spread

Effective rank = exp(entropy of s / Σs), where s are the singular values. Participation ratio = (Σλ)² / Σλ² with λ = s². Both are computed on the fit rows.

Alignment and uniformity follow Wang & Isola (α = 2, t = 2) on the L2-normalised, centred cue over held-out rows. The cues are one per scene, so there are no augmentation pairs. Positives are therefore defined as:
- spatial teachers: each scene paired with its nearest held-out scene in oracle-B space;
- speech: a random utterance of the same speaker.

The alignment ratio is the positive-pair distance divided by the random-pair distance, so 1 means the positives are no closer than a random pair.

| Teacher | dim | eff. rank | PR | top-1 / top-10 var | dims < 1% median var | alignment (ratio) | uniformity |
|---|---|---|---|---|---|---|---|
| A′ before | 768 | 253.5 | 48.3 | 7.6% / 33.8% | 0% | 1.64 (0.82) | −3.83 |
| A′ after (V1b cue) | 768 | 254.1 | 50.3 | 7.4% / 32.7% | **11.5%** | 1.64 (0.82) | −3.83 |
| speech before | 768 | 334.0 | 8.1 | 32.2% / 66.3% | 0% | 0.97 (0.48) | −3.30 |
| speech after (V1b cue) | 108 | 80.7 | 10.0 | 27.2% / 65.8% | 0% | 0.91 (0.46) | −3.24 |
| oracle B | 6 | 5.18 | 3.52 | 43.4% / 100% | 0% | 0.11 (0.06) | −2.41 |
| HC | 5 | 4.32 | 2.44 | 58.8% / 100% | 0% | 1.00 (0.50) | −2.14 |

- **A′ shows no collapse:** a third of its 768 dimensions are effective, the spectrum is flat, and uniformity is high. However, the acoustically closest room is barely closer than a random one (ratio 0.82; oracle B 0.06). The PCA rotation leaves a tail of 11.5% of dimensions with near-zero variance; section 3 explains why that matters.
- **The speech teacher is anisotropic.** Its effective rank is high but its participation ratio is only 8–10, and the top direction holds about 30% of the variance. Same-speaker utterances sit at half the random distance.
- **HC is dominated by one direction** (59% of the variance), which is the TDOA / coherence axis.

## 2. Linear interpretability (ridge)

Ridge probes follow the standard protocol, `rir_encoder.ridge_cv`: z-scored X, an intercept, 33 λ values, and 5-fold cross-validated MAE. SID uses a one-hot multi-output ridge, with λ chosen by cross-validated accuracy. Table values are test MAE, with R² in parentheses.

| Cue (paired set) | TDOA (samples) | T60 (s) | DRR (dB) | C50 (dB) | volume (m³) | src–mic dist (m) |
|---|---|---|---|---|---|---|
| mean baseline | 2.88 | 0.180 | 2.67 | 1.75 | 119 | 1.31 |
| A′ before | 0.98 (0.84) | 0.013 (0.994) | 0.94 (0.87) | 0.30 (0.97) | 60.5 (0.71) | 0.44 (0.89) |
| A′ after (V1b cue) | 0.99 (0.84) | 0.018 (0.989) | 0.94 (0.87) | 0.30 (0.97) | 61.8 (0.70) | 0.46 (0.88) |
| HC | 0.45 (0.93) | 0.125 (0.46) | 2.00 (0.47) | 0.87 (0.73) | 113 (0.07) | 1.24 (0.11) |
| oracle B | 0 (1.0) | 0 | 0 | 0 | 0 | 0 |

- **Random-init reference** (from rir_encoder `probes.json`, cue set, raw embedding): TDOA 2.37, T60 0.0137, DRR 1.03, C50 0.367. Trained A′ on the same set (my rerun reproduces it exactly): 1.03 / 0.0135 / 0.93 / 0.30.
  - Training mainly buys TDOA. The random-init encoder already reads out T60 almost as well.
- **Speech teacher:** SID accuracy is **99.1% before and 94.6% after** PCA k = 108 + L2, against 1.0% for the majority class.

## 3. Distance geometry

Method:
1. For each held-out pair, take the cue difference d and split ‖d‖² into the part inside each label's subspace, the part inside the joint span of all labels, and the residual.
2. Report the mean share per pair. The ratio of sums is in the json and agrees to within 0.01.
3. As a chance level, a random m-dimensional subspace gets m/d.

**One deviation from the request.** You asked for the ridge-probe direction itself, i.e. the weight vector w / sd in cue space (the "readout"). For A′ that direction puts **99.9–100% of its weight on the 11.5% of near-zero-variance PCA-tail dimensions**. The protocol z-scores each dimension, and that divides the tail by its tiny sd. As a result the readout subspaces carry **0.00%** of the distance (residual 100%), even though the label prediction is exactly that projection (in-subspace Spearman 0.98 for T60). This shows *where the labels can be read*, not *where the distance is*.

So the main numbers use the **encoding pattern** Cov(X)·w (Haufe et al. 2014): the direction along which the cue actually moves when the label changes. Both variants are in the json, as `readout` and `pattern`.

| Mean share of ‖d‖² (paired, pattern) | TDOA | T60 | DRR | C50 | volume | dist | reverb span (T60, DRR, C50) | all 6 labels | **residual** |
|---|---|---|---|---|---|---|---|---|---|
| A′ (768-d; chance 0.13% per label) | 2.3% | 5.5% | 3.5% | 5.1% | 2.7% | 2.4% | **9.1%** (chance 0.4%) | 14.3% (chance 0.8%) | **85.7%** |
| A′, readout directions | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100% |
| HC (5-d; chance 20%) | 26.3% | 32.8% | 40.9% | 37.3% | 35.9% | 41.6% | 63.6% (chance 60%) | 100% | 0 (5 dims fully spanned) |
| oracle B (6-d; chance 16.7%) | 20.7% | 29.6% | 31.4% | 32.1% | 24.8% | 29.9% | 53.6% (chance 50%) | 100% | 0 (by construction) |

- **Cue set (3,000 rooms), A′:** reverb span 9.7%, all six labels 14.6%, residual 85.4%.
- **The single-label shares overlap** because the directions are not orthogonal. In B, for example, T60, DRR and C50 are strongly correlated.
- For HC and B the joint span is the whole space, so their residual is 0 trivially. The per-label shares are the informative numbers there.

**Spearman of the in-subspace distance with the true label distance |Δlabel|**, using pattern directions. Values in parentheses are for the full cue distance.

| | TDOA | T60 | DRR | C50 | volume | dist |
|---|---|---|---|---|---|---|
| A′ | 0.62 (0.20) | **0.95 (0.35)** | 0.51 (0.10) | 0.74 (0.24) | 0.23 (0.05) | 0.58 (0.09) |
| HC | 0.92 (0.42) | 0.15 (0.11) | 0.20 (0.15) | 0.28 (0.21) | 0.04 (0.01) | 0.03 (0.02) |
| oracle B | 0.95 (0.31) | 0.75 (0.44) | 0.77 (0.48) | 0.87 (0.55) | 0.83 (0.38) | 0.75 (0.45) |

- With readout directions, A′'s in-subspace values are higher: 0.77 / 0.98 / 0.78 / 0.94 / 0.59 / 0.79.
- **A′ contains a good room metric** (its T60 direction ranks pairs at 0.95), but the full distance V1b uses is drowned by the residual. A full-distance correlation of 0.35 with T60 and 0.10 with DRR is weaker than oracle B's (0.44 / 0.48).
- **HC's distance is TDOA plus coherence.** Reverb is weakly present, even in its own label directions, which fits HC λ1's poor T60 (0.125).

**Speech teacher (SID).**
- The span of all 113 probe weight vectors has rank 108 = the whole cue space, so it is degenerate. I report the top singular directions of the pattern matrix instead:
  - the 14 directions holding 90% of its energy (13.0% of dimensions) carry **55.5%** of the distance;
  - the top 2 (1.9%) carry 25.9%.
- Readout directions, 90% energy: 44 directions, 12.2% of the distance against 40.7% chance. This is the same tail effect as for A′.
- Same- versus different-speaker discrimination by distance: AUC **0.93** on the full cue, 0.95 inside the 14-dim speaker subspace. Spearman is not useful here, because 99% of pairs are different-speaker.
- **The speech cue's geometry is speaker-dominated**, the opposite of A′.

## 4. Teacher → student CKA

**Inputs.**
- Clips: the 2,000 held-out paired-v2 clips (500 scenes × 4 renderings), first 4.112 s.
- Student embedding: the per-encoder mean-pooled patches that the lineareval head and ridge_dump see.
- Teacher cue per clip: its scene's A′ / HC / B cue, and its utterance's speech cue.
- Staging: clips and checkpoints copied node-local.

**Results.** Linear CKA, clip level. Debiased CKA (unbiased HSIC) agrees to within 0.01 everywhere (cka.json), except that V1b spectral × room cues drop to about 0.00.

| Student encoder | A′ | HC | oracle B | speech |
|---|---|---|---|---|
| V1b A′ λ0.1, spatial | **0.690** | 0.593 | 0.384 | 0.007 |
| V1b-HC λ0.1, spatial | 0.389 | **0.832** | 0.275 | 0.004 |
| C v2 10 ep, spatial | 0.456 | 0.372 | **0.586** | 0.047 |
| V1b A′ noanc 30 ep s2, spatial | **0.705** | 0.581 | 0.374 | 0.005 |
| V1b A′ λ0.1, spectral | 0.006 | 0.004 | 0.003 | **0.860** |
| V1b-HC λ0.1, spectral | 0.006 | 0.005 | 0.003 | **0.867** |
| C v2 10 ep, spectral | 0.231 | 0.173 | 0.435 | 0.413 |
| V1b A′ noanc 30 ep s2, spectral | 0.005 | 0.003 | 0.002 | **0.933** |

- **Scene level** (each scene's 4 clips averaged):
  - spatial × A′: 0.70 / 0.71 for the two V1b A′ runs;
  - V1b-HC spatial × HC: 0.84;
  - C v2: spatial × B 0.62, **spectral × B 0.71**.
- **Teacher × teacher:** A′ ~ HC 0.41, A′ ~ B 0.37, HC ~ B 0.31, any room cue ~ speech ≤ 0.01.
- Each V1b student takes on its own teacher's geometry, to a degree well above how much the teachers share with each other (0.31–0.41).
- The spatial student matches A′ about twice as closely as it matches oracle B. Going from 10 to 30 epochs slightly increases the match to A′ (0.69 → 0.71).
- The V1b spectral encoders keep no room structure. C v2's spectral encoder keeps a lot, which matches the leakage numbers in RUNS.md.

## Caveats

- The readout-versus-pattern split arises from the protocol's per-dimension z-scoring of a PCA-rotated cue. On the unrotated A′ "before" embedding, no dimension has near-zero variance. A ridge without z-scoring would give directions closer to the pattern ones.
- Oracle B and HC are fully spanned by their own label directions, so their residual is 0 by construction. Compare them through the per-label shares and the Spearman values.
- Single seed for every teacher and every student. Step 4 uses one checkpoint per run (`*_final.pth`) and 2,000 clips. The 30-ep s1 run is not included.
