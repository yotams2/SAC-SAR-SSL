# Phase 0 diagnostics + Phase 1 proposal (Yotam 11-28: improve the core method, no training until a plan is approved)

Core method: the spectral encoder sees x(s, r′) and the spatial encoder x(s′, r); a shared decoder rebuilds x(s, r); SAC aligns each encoder's pooled embedding with a teacher's similarity (dry speech / RIR).

**Checkpoints:**
- C v2 10 ep (pilotC_v2r);
- V1b-A′-noanc 10 ep (pilotc_v1bA_noanc_l01);
- C 30 ep (pilotC30_v2r);
- final noanc 30 ep s1 / s2 (pilotc_v1bA30noanc_l01_s{1,2}).

All runs were eval-only or data-only, at nice 100. Validation is the fixed check-4 triplets: 2,000 targets from the 500 held-out paired-v2 scenes.

## Summary

1. **The spatial encoder supplies almost none of the target it is trained on.**
   - ILD is 4% explained, cos IPD 22%, sin IPD 16%.
   - The per-frame ILD target is mostly unpredictable from the room: 30% noise, 67% source × room interaction, 3% room.
2. **Clip-level (time-averaged) ILD, IPD and IC are 81–85% room-determined, with ~0 noise.** Those are the targets a room encoder can own (D2). This is the strongest lead (P2).
3. **The decoder uses per-frequency-band room information, not a single global vector.**
   - Time-pooling the spatial tokens within each frequency row costs +0.01 to +0.03.
   - A single mean vector costs +0.14 to +0.18 (D3).
4. **A low-dimensional PCA view of the RIR teacher does not make its distances label-aligned.** ρ DRR stays ≤ 0.06 for every k (D4). A teacher fix needs a different objective, not a smaller projection.
5. **The checkpoints barely differ in how they fit the swap target** (±0.002 in every column). SAC changes the encoders' embeddings, not the swap fit.

---

## D1. Who explains the swap target?

**Loss normalisation.**
- The target is the ipd16 image of x(s, r): log|L|, ILD = log|R| − log|L|, cos IPD and sin IPD, over 256 bins × 256 frames.
- Each channel is z-scored with the spatial stream's global per-channel input mean and std, so the raw MSE of a global-mean predictor is about 1.
- The loss is the mean of the 4 per-channel MSEs. The IPD channels are weighted per bin by min(√(|L||R|) / clip mean, 5).
- The table below uses check 4's bin-centred scale (code/swap_shufdiag.py): error / Σ W (T − μ_f)², where μ_f is each frequency bin's mean. **1.0 = the per-bin mean predictor** (the mean predictor you asked for).
- On the raw scale, the per-bin mean already explains part of two channels: bin-centred variance / global variance = logL 0.74, ILD 1.00, cosIPD 0.82, sinIPD 1.00.

**Correction to the RUNS.md reading.**
- 0.572 → 0.607 is the training log's roll-by-one diagnostic. The validation loader is unshuffled and the k = 4 clips of a scene are adjacent, so for 3 of every 4 targets the "spatially shuffled" donor is from the *same* scene.
- With donors from another scene (check 4), the final run's swap loss rises **0.572 → 0.727** (s2: 0.572 → 0.730). The spectral shuffle gives 0.87, both shuffled 0.95.
- **ILD ≈ 0.96 is confirmed:** 4% of ILD's bin-centred variance is predicted, even with the true inputs.

Final noanc 30 ep s1, bin-centred error, all bins (the other four checkpoints are within ±0.02, see d1_check4_*.json):

| condition | logL | ILD | cosIPD | sinIPD |
|---|---|---|---|---|
| true inputs | 0.32 | 0.96 | 0.78 | 0.84 |
| spectral shuffled | 1.63 | 1.00 | 0.95 | 0.97 |
| spatial shuffled | 0.39 | 1.05 | 1.19 | 1.18 |
| both shuffled | 1.66 | 1.05 | 1.19 | 1.13 |
| mean predictor (per bin) | 1 | 1 | 1 | 1 |

Cost of the spatial shuffle per band (0–2 / 2–4 / 4–6 / 6–8 kHz):

| channel | cost |
|---|---|
| logL | +0.06 / +0.06 / +0.07 / +0.08 |
| ILD | +0.07 / +0.10 / +0.09 / +0.08 |
| cosIPD | +0.44 / +0.43 / +0.35 / +0.33 |
| sinIPD | +0.32 / +0.36 / +0.34 / +0.31 |

Error with true inputs, per band: cosIPD 0.74 / 0.79 / 0.82 / 0.83, ILD 0.95–0.97 in every band.

- **Spectral side:** it supplies log|L|, 68% explained. It is the only stream that matters for logL.
- **Spatial side:** it supplies only IPD, and only 16–22% of it. With a wrong room, IPD gets *worse than the per-bin prior* (1.19), because the decoder confidently predicts the donor's IPD.
- **ILD** is unexplained by either stream.

## D2. Predictability ceiling of the target

Code: code/data_generation/d2_ceiling.py (job 23818492, 11 min).
- **Re-rendering:** every clip is rebuilt from its dry source and the stored RIR, at the stored SNR. The original noise seed reproduces the stored wav to −68…−74 dB, i.e. quantisation error.
- **Second draw:** each clip is rendered again with an independent noise draw.
- **Groups:** 500 scene groups (held-out scenes × their 4 utterances) and 500 utterance groups (× their 4 scenes).
- **Split:** variance around each bin's mean = noise (½ E[(f_a − f_b)²]) + room main + speech main + interaction (source × room). Group shares are corrected by k/(k−1).
- **Ceiling** = 1 − noise share: what a decoder that knew s and r exactly could reach.

**Resolutions:**
- **frame:** the current per-frame ipd16.
- **seg16:** 256-ms segments (one decoder patch column), computed from segment-summed spectra.
- **clip:** the whole 4.1-s clip, one value per frequency bin.

For seg16 and clip: ILD = log Σ|R|² − log Σ|L|²; IPD = angle(Σ L R*); IC = |Σ L R*| / √(Σ|L|² Σ|R|²).

| resolution | channel | ceiling | noise | room main | speech main | interaction | room main per band |
|---|---|---|---|---|---|---|---|
| frame | logL | 0.92 | 0.08 | 0.04 | 0.68 | 0.21 | 0.05 / 0.04 / 0.03 / 0.03 |
| frame | ILD | 0.70 | 0.30 | 0.03 | 0.00 | 0.67 | 0.03 / 0.03 / 0.03 / 0.02 |
| frame | cosIPD | 0.94 | 0.06 | 0.33 | 0.06 | 0.55 | 0.30 / 0.33 / 0.35 / 0.37 |
| frame | sinIPD | 0.94 | 0.06 | 0.29 | 0.04 | 0.61 | 0.24 / 0.30 / 0.34 / 0.36 |
| seg16 | logL | 1.00 | 0.00 | 0.05 | 0.91 | 0.03 | 0.06 / 0.06 / 0.04 / 0.05 |
| seg16 | ILD | 0.92 | 0.08 | 0.39 | 0.00 | 0.52 | 0.36 / 0.46 / 0.40 / 0.33 |
| seg16 | cosIPD | 0.86 | 0.14 | 0.45 | 0.00 | 0.40 | 0.58 / 0.54 / 0.41 / 0.32 |
| seg16 | sinIPD | 0.84 | 0.16 | 0.39 | 0.00 | 0.46 | 0.44 / 0.47 / 0.37 / 0.29 |
| seg16 | IC | 0.92 | 0.08 | 0.28 | 0.19 | 0.45 | 0.39 / 0.33 / 0.20 / 0.15 |
| clip | logL | 1.00 | 0.00 | 0.11 | 0.86 | 0.02 | 0.14 / 0.12 / 0.09 / 0.11 |
| clip | ILD | 1.00 | 0.00 | **0.85** | 0.00 | 0.15 | 0.80 / 0.89 / 0.86 / 0.84 |
| clip | cosIPD | 0.99 | 0.01 | **0.85** | 0.00 | 0.14 | 0.90 / 0.89 / 0.84 / 0.78 |
| clip | sinIPD | 0.99 | 0.01 | **0.83** | 0.00 | 0.16 | 0.85 / 0.87 / 0.83 / 0.79 |
| clip | IC | 1.00 | 0.00 | **0.81** | 0.03 | 0.16 | 0.88 / 0.85 / 0.77 / 0.69 |

- **Per-frame ILD**: 30% is noise and 67% is source × room interaction. That interaction is the reverberant tail acting on this exact waveform, which neither encoder sees (the spatial encoder has another utterance in the room; the spectral encoder has this utterance in another room). The room alone explains 3%. That is why the decoder gets 4%.
- **Per-frame IPD**: the room explains about 31%. The models reach 16–22%, below the room's own share.
- **Time-averaged ILD, IPD and IC are room properties** (81–85% room main, ≤1% noise). 256-ms segments sit in between (28–45%).
- **logL is speech** at every resolution (68–91%). The current target already carries content through log|L| (D1: the spectral shuffle cost is +1.3), so a reverberant log-mel target (P1) would add little new information.

Caveat: the seg16 cos/sin IPD are unweighted, so low-energy segments add noise (14–16%, against 6% for the magnitude-weighted per-frame IPD).

## D3. Does the decoder use the spatial tokens?

Code: code/d3_decoder_use.py (job 23818449). Same triplets and scale as D1. The attention shares below are corrected by 1/16, because the first run's normaliser was wrong; the script is now fixed.

**Spatial tokens replaced, error (mean of the 4 channels):**

| checkpoint | normal | per-band time mean | global mean | other scene |
|---|---|---|---|---|
| C v2 10 ep | 0.747 | 0.761 | 0.927 | 0.929 |
| V1b-A′-noanc 10 ep | 0.745 | 0.758 | 0.900 | 0.930 |
| C 30 ep | 0.726 | 0.751 | 0.872 | 0.946 |
| final noanc s1 | 0.728 | 0.750 | 0.866 | 0.950 |
| final noanc s2 | 0.727 | 0.755 | 0.876 | 0.951 |

- **Per-band time mean:** each spatial token is replaced by the mean over the time columns of its frequency row. It costs +0.01 to +0.03 (cos IPD +0.02–0.04). The decoder hardly uses the spatial tokens' time structure; a per-band (16-row) room summary carries nearly everything it uses.
- **Global mean:** one vector, broadcast. It costs +0.14 to +0.18, almost as much as a wrong room. Most of the cost is cos IPD, 0.78 → 1.10–1.20. A single global vector, as the decoder is trained now, does not carry the per-band phase.
  - Caveat: both pooled inputs are off the training distribution, so these costs are upper bounds.
- **Spectral side, for contrast:** the per-band time mean costs +0.6 to +0.8 on logL, the global mean +0.8 to +1.0. Content needs time structure.

**Cross-attention share on the spatial tokens** (uniform = 0.67, from 257 of 386 memory tokens):

| checkpoint | layer 1 | layer 2 | layer 3 | layer 4 |
|---|---|---|---|---|
| C v2 10 ep | 0.43 | 0.70 | 0.68 | 0.79 |
| V1b-A′-noanc 10 ep | 0.47 | 0.69 | 0.68 | 0.67 |
| C 30 ep | 0.37 | 0.71 | 0.76 | 0.85 |
| final noanc s1 | 0.37 | 0.65 | 0.69 | 0.91 |
| final noanc s2 | 0.42 | 0.67 | 0.87 | 0.91 |

- Layer 1 attends mostly to the spectral tokens; the last layer mostly to the spatial ones (0.91 after 30 ep). The share is flat across query frequency bands.
- The spatial cls token gets about 1%.
- So the attention does reach the spatial tokens, but what it reads from them is a per-band summary.

## D4. Does a low-dimensional view of the teachers give label-aligned distances?

Code: code/d4_topk.py (CPU, 2 min). PCA fitted on the fit rows, top-k projection, L2. Same metrics as gate 1.

**Room teacher A′-pad** (paired held-out set, all pairs; residual outside the 6-label pattern span):

| k | TDOA | T60 | DRR | C50 | ρ T60 | ρ DRR | ρ TDOA | ρ reverb | residual |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2.20 | 0.065 | 2.50 | 1.08 | 0.31 | 0.03 | 0.21 | 0.15 | (degenerate) |
| 8 | 2.03 | 0.056 | 2.23 | 0.99 | 0.26 | 0.05 | 0.22 | 0.14 | (degenerate) |
| 16 | 1.98 | 0.047 | 2.07 | 0.90 | 0.23 | 0.05 | 0.22 | 0.13 | 61% |
| 32 | 1.85 | 0.041 | 1.94 | 0.82 | 0.20 | 0.05 | 0.20 | 0.12 | 78% |
| 108 | 1.50 | 0.030 | 1.26 | 0.54 | 0.18 | 0.06 | 0.19 | 0.12 | 91% |
| 768 | 1.12 | 0.020 | 0.97 | 0.33 | 0.18 | 0.06 | 0.19 | 0.12 | 93% |

- **A′-pad:** ρ T60 rises at small k (0.31 at k = 4), but ρ DRR never exceeds 0.06, and the probes collapse (DRR 2.50 at k = 4 against 0.97).
- **Residual is not meaningful at k ≤ 8:** six label directions span the whole space, so it reads 0–20% by construction.
- **A′, the original, unpadded teacher:** ρ T60 0.35–0.40 and ρ DRR 0.07–0.11 for every k (full table in d4_topk.json).
- **Reading:** the label directions are not the teacher's top-variance directions. A bottleneck on the existing objective would keep the decay or length axis and drop DRR.

**Dry-speech teacher (pooled):**

| k | SID acc | AUC (different vs same speaker) | residual vs SID span |
|---|---|---|---|
| 4 | 0.12 | 0.87 | 25% |
| 8 | 0.27 | 0.90 | 28% |
| 16 | 0.60 | 0.92 | 34% |
| 32 | 0.78 | 0.93 | 37% |
| 108 (the V1b cue) | 0.95 | 0.93 | 45% |
| 768 | 0.99 | 0.93 | 49% |

- The speech teacher's distance is already speaker-aligned at k ≥ 16 (AUC 0.92–0.93).
- Its residual is about half the room teacher's.
- Down to k = 32 it loses almost no AUC but 17 points of SID accuracy. A small bottleneck is possible on the speech side; it is not a fix the room side needs.

---

## Phase 1 proposal (one change per pilot; nothing runs until you pick)

**Base:** V1b-A′-noanc 10 ep's recipe (paired v2, 10 ep, seed 1, anchors off). Every pilot is compared with V1b-A′-noanc 10 ep (decision row) and C v2 10 ep (reported).

**Cost per pilot:**
- training ≈ 4 GPU-h (10 ep × 375 steps × 3.9 s);
- the full eval set ≈ 15–25 GPU-h (three spatial tasks at 2.5–8 h each, plus speech probes, leakage, ridge, check 4);
- about one day of wall time per pair of pilots.

At most 2 GPU jobs at a time.

**Pre-registered pass rule (all pilots), against V1b-A′-noanc 10 ep, single seed:**
- **Spatial** (frozen spatial encoder; TDOA, T60, DRR, plus C50 where run): no task more than 5% worse, **and** T60 or DRR at least 5% better. TDOA alone is not enough, since TDOA is already the method's strength.
- **Speech** (spectral encoder, linear SC2 / SID): neither more than 1.5 points worse.
- **Mechanism** (check 4 + D3 rerun on the pilot): the new component is predicted (bin-centred error < 0.5), and the spatial shuffle costs > 0.3 on it, i.e. the spatial encoder supplies it.
- A pass gets a second seed before any combination or 30-ep run.

**P2. Room-determined targets (first; slot A).**
- **Change:** add clip-level per-bin ILD, cos/sin IPD and IC of x(s, r) (256 × 4 values, z-scored) to the swap target. A second output head on the decoder pools its 256 query states per frequency row. The four new channels together get one channel's weight. The per-frame ipd16 target stays as it is.
- **Motivation:** D1 (ILD 4%, IPD 16–22% explained) and D2 (these clip-level quantities are 81–85% room-determined; per-frame ILD is 3%).
- **Expected effect:**
  - The spatial encoder gets a gradient it can actually satisfy, carrying reverberation (IC, ILD) and direction (IPD). T60 / DRR / C50 should improve; they are the method's weak point (0.089 / 1.41 / 0.565 against C 30 ep's 0.070 / 1.27 / 0.387).
  - TDOA should hold, since the clip-level IPD is the direct-path phase.
- **Cost:** 4 + about 20 GPU-h.
- **Variant P2b,** only if P2 passes: replace per-frame ILD (70% ceiling, 3% room) with the seg16 ILD.

**P3. Per-band pooled spatial conditioning (second; slot B, in parallel with P2).**
- **Change:** the decoder sees 16 per-frequency-row means of the spatial tokens (plus cls) instead of the 257 tokens.
- **Why not a single FiLM vector:** D3 shows a single global vector loses the per-band phase (+0.14–0.18), while per-band means lose almost nothing (+0.01–0.03) even untrained.
- **Expected effect:**
  - The spatial encoder is pushed to put the room into a few band summaries close to the pooled embedding that SAC and the probes read, so T60 / DRR on the linear probe may improve.
  - Risk: TDOA, if any time-resolved phase is used. D3 says little is.
- **Cost:** 4 + about 20 GPU-h, and a smaller decoder memory (386 → 146 tokens), so training is slightly faster.

**P4. Room teacher with an effect objective (third; short teacher jobs, gated before any student run).**
- **Change:** train the RIR encoder so that, given a dry-speech clip, its embedding predicts the clip-level ILD / IPD / IC and band-energy decay of dry ∗ RIR (padded RIRs). The targets are D2's room-determined quantities; labels are used only to evaluate.
- **Why not a bottleneck:** D4 shows that a low-dimensional projection of A′-pad does not fix ρ DRR at any k; 1c showed that an easy contrastive term makes it worse.
- **Cost:** about 0.5–1 GPU-h per teacher plus CPU geometry. Gate 1 as before (ρ T60, ρ DRR ≥ 0.35, residual < 80%, probes within 5% of A′-pad).
- If it passes, it replaces A′ in a V1b pilot, which goes through the same pass rule.

**P1. Reverberant log-mel in the target (deprioritised).**
- D2: per-frame logL is 68% speech main and 4% room main.
- D1: the target already carries content through log|L|, where the spectral shuffle cost is +1.3 and the spectral side gets 68%.
- A log-mel channel would mostly duplicate that. Proposed only as an ablation after P2.

**P5. Token-level speech targets (later, as you planned).**
- D4: the pooled speech teacher is already speaker-aligned (AUC 0.93) with half the room teacher's residual, so the speech side is not the bottleneck now.

**Order:** P2 ∥ P3 (two slots) → P4 teacher jobs in the first free slot → second seed of whichever passes → the P2 + P3 combination, if both pass.
