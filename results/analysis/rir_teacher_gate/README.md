# Gate 1: RIR teacher with meaningful distances (post-sprint step 1, Yotam 11-28)

Rows are RIR teachers. Probes: rir_encoder ridge, test MAE on 3,000 held-out rooms (TDOA samples, T60 s, DRR dB, C50 dB). ρ: Spearman of the full V1b-cue distance (centred, z-scored embedding, L2) with |Δlabel| over all pairs of the 500 held-out paired-v2 rooms; reverb factor = |ΔPC1| of oracle B's 5 non-TDOA dims. Residual: share of the pair distance outside the span of the 6 label pattern directions. Effective rank and uniformity: held-out rows (step 1 of teacher_geometry.py). Code: SAC-SAR-SSL code/rir_gate.py, teacher_geometry.py --spatial-only --pca centre.

Gate (Yotam): ρ T60 and ρ DRR ≥ ~0.35 each; residual clearly below A′'s ~86% (operationalised here as < 80%); probes no worse than A′-pad (here: within +5% MAE).

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
