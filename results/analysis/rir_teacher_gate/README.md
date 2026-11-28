# Gate 1: RIR teacher with meaningful distances (post-sprint step 1, Yotam 11-28)

Rows are RIR teachers. Probes: rir_encoder ridge, test MAE on 3,000 held-out rooms (TDOA samples, T60 s, DRR dB, C50 dB). ρ: Spearman of the full V1b-cue distance (centred, z-scored embedding, L2) with |Δlabel| over all pairs of the 500 held-out paired-v2 rooms; reverb factor = |ΔPC1| of oracle B's 5 non-TDOA dims. Residual: share of the pair distance outside the span of the 6 label pattern directions. Effective rank and uniformity: held-out rows (step 1 of teacher_geometry.py). Code: SAC-SAR-SSL code/rir_gate.py, teacher_geometry.py --spatial-only --pca centre.

Gate (Yotam): ρ T60 and ρ DRR ≥ ~0.35 each; residual clearly below A′'s ~86% (operationalised here as < 80%); probes no worse than A′-pad (here: within +5% MAE).

**Outcome (11-28): gate 1 not passed; step 1c stopped after its first arm (Yotam: post-sprint plan stopped). No teacher passes.**

## 1b: A′-pad (A′ design 1 on the noise-padded RIRs, same objective / seed)

| teacher | TDOA | T60 | DRR | C50 | ρ ΔTDOA | ρ ΔT60 | ρ ΔDRR | ρ ΔC50 | ρ reverb | residual | eff. rank | uniformity | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A′ | 1.028 | 0.0135 | 0.926 | 0.300 | 0.20 | 0.35 | 0.10 | 0.23 | 0.19 | 85.7% | 254 / 768 | -3.83 | fail: rho_T60>=0.35, rho_DRR>=0.35, residual<0.80 |
| A′-pad | 1.147 | 0.0169 | 0.933 | 0.303 | 0.19 | 0.18 | 0.06 | 0.13 | 0.12 | 92.8% | 340 / 768 | -3.93 | fail: rho_T60>=0.35, rho_DRR>=0.35, residual<0.80 |
| oracle B | 0 | 0 | 0 | 0 | 0.31 | 0.44 | 0.48 | 0.55 | 0.61 | 0.0% | 5 / 6 | -2.41 |  |

- **Padding removes the length shortcut.** T60 from the observed RIR length is now 0.191 s, the mean predictor (from the true rir_len it would be 0.031). Random-init T60 rises from 0.0137 to 0.0241.
- **A′-pad still reads the labels** (T60 0.0169, DRR 0.93, C50 0.30, all close to A′), so T60 is learned from the decay, not the length. TDOA is 11% worse (1.15 vs 1.03).
- **But its distance is less meaningful than A′'s**:
  - ρ T60 0.35 → 0.18, ρ DRR 0.10 → 0.06, ρ reverb 0.19 → 0.12;
  - residual 85.7% → 92.8%; the label span halves (14.3% → 7.2%; reverb span 9.1% → 3.9%);
  - effective rank 254 → 340.
- Reading: part of A′'s T60 ρ (and of its label-aligned distance) came from the length shortcut. Without it, the reconstruction objective spreads the distance over more, mostly label-unrelated directions. 1c (A′-metric) starts from this weaker base.

## 1c: A′-metric, arm w0.1 crop (the only arm run; w1 crop and w0.1 no-crop were cancelled before they started)

A′-pad plus an NT-Xent term (weight 0.1) on the pooled 768-d embedding. The two views of each RIR differ in fresh −60 dB padding noise, a tail crop starting at max(4096, 0.8·rir_len), and a ±6 dB gain. Reconstruction runs on the unaugmented batch. Same data, seed and 100 epochs as A′-pad. Run: exp/rir_encoder/a1m_w01 (job 23817318, 35 min on an L40S); geometry exp/teacher_geometry_post/a1m_w01 (job 23817319).

| teacher | TDOA | T60 | DRR | C50 | ρ ΔTDOA | ρ ΔT60 | ρ ΔDRR | ρ ΔC50 | ρ reverb | residual | eff. rank | uniformity | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A′ | 1.028 | 0.0135 | 0.926 | 0.300 | 0.20 | 0.35 | 0.10 | 0.23 | 0.19 | 85.7% | 254 / 768 | -3.83 | fail: ρ T60, ρ DRR, residual |
| A′-pad | 1.147 | 0.0169 | 0.933 | 0.303 | 0.19 | 0.18 | 0.06 | 0.13 | 0.12 | 92.8% | 340 / 768 | -3.93 | fail: ρ T60, ρ DRR, residual |
| A′-metric w0.1 crop | 1.239 | 0.0195 | 1.114 | 0.403 | 0.07 | 0.07 | 0.02 | 0.05 | 0.04 | 97.9% | 439 / 768 | -3.97 | fail: ρ T60, ρ DRR, residual, probes |
| random init (padded) | 2.376 | 0.0241 | 0.977 | 0.353 | | | | | | | | | |
| oracle B | 0 | 0 | 0 | 0 | 0.31 | 0.44 | 0.48 | 0.55 | 0.61 | 0.0% | 5 / 6 | -2.41 | |

- **It is worse than A′-pad on every gate column.**
  - ρ T60 falls from 0.18 to 0.07, ρ DRR from 0.06 to 0.02, ρ reverb from 0.12 to 0.04.
  - The residual rises from 92.8% to 97.9%, and the effective rank from 340 to 439.
  - The Spearman of its concat-pool cosine distance against oracle B's distance falls from 0.21 to 0.08; mean-pool stays at 0.37 (A′-pad 0.35).
- **The probes degrade too.** TDOA +8%, T60 +15%, DRR +19%, C50 +33% against A′-pad. DRR and C50 are now worse than the random-init encoder (1.11 vs 0.98; 0.40 vs 0.35).
- **Why:** the contrastive task was too easy. NT-Xent fell from 0.81 to 0.05 by epoch 10 and to 0.027 by the end. The two views share the whole early part of the RIR, so the encoder can match them from fine detail that is unique to each RIR. That adds instance-specific directions (higher rank, larger residual) rather than room-level ones. The views did not differ in anything that the room labels ignore.
- Not run: w1 crop, w0.1 no crop. The w0.1 result gives no reason to expect a stronger weight to help, since the term was already solved.

Files: gate.json (all rows; rebuilt with code/rir_gate.py, which reads geometry.json + probes.json), probes_ and geometry_ for a1_pad60 and a1m_w01.
