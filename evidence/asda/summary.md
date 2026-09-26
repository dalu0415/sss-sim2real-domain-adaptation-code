# ASDA archived-run evidence

Complete ASDA comparator (ASDA-SEL-01/A), 5 seeds x 5 source folds, fixed 40 epochs, prediction from the
average of epochs 36-40 on all 553 target images. Values in percent; mean +/- sample SD over the five seed
means (each seed averages its five fold metrics). Recomputed by `tests/test_asda.py` from `per_run/`.

## Manuscript Table 6 (ASDA row) and Table E.2

| Metric | Mean +/- SD |
|---|---:|
| Macro-F1 | 41.91 +/- 20.15 |
| Airplane F1 | 19.27 +/- 2.82 |
| Airplane PR-AUC (AP) | 32.16 +/- 11.39 |
| Airplane precision | 53.89 +/- 30.41 |
| Airplane recall | 39.21 +/- 41.67 |
| Ship F1 | 64.55 +/- 42.52 |
| Ship precision | 92.34 +/- 4.86 |
| Ship recall | 67.91 +/- 45.46 |

Source-only and source-only+AdaBN rows of Table 6, recomputed from the seed values in `reference_arms.json`:

| Reference | Macro-F1 | Airplane F1 | Airplane PR-AUC |
|---|---:|---:|---:|
| #1 source-only | 62.69 +/- 1.87 | 38.59 +/- 1.95 | 37.57 +/- 1.99 |
| #2 source-only+AdaBN | 68.58 +/- 1.07 | 44.66 +/- 1.64 | 44.29 +/- 2.24 |

## Manuscript Table E.3

| Seed / contrast | Macro-F1 | Airplane F1 | Airplane PR-AUC |
|---|---:|---:|---:|
| 0 | 55.00 | 16.04 | 35.85 |
| 1 | 31.03 | 22.63 | 26.36 |
| 2 | 56.71 | 19.29 | 42.33 |
| 3 | 55.42 | 16.93 | 41.09 |
| 4 | 11.37 | 21.44 | 15.18 |
| ASDA - source-only | -20.79 [-46.24, +4.67] | -19.33 [-23.73, -14.92] | -5.40 [-19.09, +8.29] |
| (source-only+AdaBN) - ASDA | +26.67 [+2.30, +51.05] | +25.39 [+20.53, +30.26] | +12.13 [-0.45, +24.70] |

Contrasts: mean paired difference in percentage points with the unadjusted paired-t 95% interval
(n = 5 seeds, df = 4); reference seed values are in `reference_arms.json`.

## Prediction bias (Section 3.6)

- 17 runs predicted mostly ship (airplane predictions per run: 3-20 of 553).
- 8 runs predicted mostly airplane (airplane predictions per run: 539-552 of 553).
- Every run assigned at least 95% of its target predictions to one class (smallest share 96.4%).

## Cost records (Table 5a, Table F.1, Appendices E.5 and F)

- Training loop per run: median 468.60 s [Q1 389.14, Q3 470.27], range 258.61-474.03 s (n = 25).
- Target prediction, last five checkpoints: median 6.92 s [Q1 6.81, Q3 6.99], range 6.55-7.34 s (n = 25).
- CUDA peaks, the same in all 25 training runs: 7.034 GiB allocated, 11.260 GiB reserved (1 GiB = 2^30 bytes).
- Analytic training budget: 836.657 x 10^12 FLOPs per run (8.37 x 10^14 in the text): 2 FLOPs per
  multiply-accumulate, backward counted as twice forward, convolutional and linear layers plus the conditional
  outer product (320 x 2 x 512 multiply-accumulates per update, not included in the per_run counts), 240
  updates. The per-update convolutional and linear forward counts are in each `per_run` record.
- Analytic target prediction: 10.029 x 10^12 FLOPs (553 images, five checkpoints).

## Order of events

- The training stage for all 25 runs ended at 2026-09-17T04:25:01.730499+00:00 (UTC).
- Target labels were first read for final scoring at 2026-09-17T04:26:21.010491+00:00 (UTC), after all training had finished.
