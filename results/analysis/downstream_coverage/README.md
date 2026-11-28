# Downstream coverage, key rows (Yotam, 28 Nov 2026)

The table covers every downstream task for the six rows Yotam kept on 28 Nov.
- **Spatial tasks** read the spatial encoder: frozen lineareval, R8, 20 epochs, lr 1e-3 / 1e-4 (best by val), the new eval code. Values are test MAE, lower is better.
- **Speech tasks** read the spectral encoder: linear probe, accuracy % (ER: unweighted accuracy %), higher is better.
- **Seeds and jobs:** one seed per row, except the 30-ep mean of s1 and s2.
- **Sources:** C50 / ABS from jobs 23816219–55 (summary.json), ER from 23816221–57 (sac-ssamba exp/_probes/*_spec_er.jsonl). Everything else is from RUNS.md.

| Row | TDOA | T60 | DRR | C50 | ABS | SC2 | SID | ER (UA) |
|---|---|---|---|---|---|---|---|---|
| A (unpaired-4s v1) | 2.57 | 0.084 | 2.32 | 0.816 | 0.0693 | 14.8 | 34.0 | 54.3 |
| C 30 ep (baseline for 30-ep rows) | 2.47 | **0.070** | **1.27** | **0.387** | 0.0746 | 23.8 | 45.9 | **55.9** |
| V1b A′ λ0.1, 10 ep | 1.49 | 0.090 | 1.43 | 0.557 | 0.0762 | 22.3 | 46.1 | 54.0 |
| V1b-HC λ0.1, 10 ep | **1.32** | 0.074 | 1.46 | 0.498 | **0.0587** | 19.5 | 46.1 | 55.7 |
| V1b-A′-noanc 30 ep s1 | 1.33 | 0.092 | 1.42 | 0.534 | 0.0762 | 22.5 | **47.5** | 55.3 |
| V1b-A′-noanc 30 ep s2 | 1.38 | 0.086 | 1.39 | 0.597 | 0.0730 | **24.6** | 44.4 | 53.8 |
| *noanc 30 ep, mean of s1 / s2* | *1.36* | *0.089* | *1.41* | *0.565* | *0.0746* | *23.6* | *46.0* | *54.6* |

**Final row against C 30 ep** (mean of the two seeds):
- **Better:** TDOA, 1.36 vs 2.47 (45% lower).
- **Worse:**
  - T60: 0.089 vs 0.070;
  - DRR: 1.41 vs 1.27;
  - C50: 0.565 vs 0.387, about 46% higher error, so C50 joins T60 and DRR as reverb tasks where the room teacher loses to C.
- **Level:** ABS 0.0746 vs 0.0746, and SC2 / SID within 0.2 points.
- **ER:** −1.3 points, below the 1.5-point tie threshold, so a tie.
- The two seeds differ by 0.06 on C50 and 1.5 points on ER.

**Notes:**
- ABS spans only 0.059–0.076 across all rows, the HC row is the best on it, and a from-scratch random-init SSAMBA got 0.0625 (11-17, ctrl_ipd16_n256, not frozen, so not directly comparable). ABS barely separates the encoders.
- **Leakage cells:** cancelled at Yotam's request, so none for C50 / ABS / ER.
- **Tasks not covered:**
  - ESC-50 / UrbanSound8K / AudioSet / Speech Commands v1: no binaural rendering;
  - LOCATA / ACE real-room sets: not staged on scratch.
