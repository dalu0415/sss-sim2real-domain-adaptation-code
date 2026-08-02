# E2 ablation summary -- arm3

- arm dataset ID: arm3_internal_structure
- n_seed present: 12 / 12  (seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
- separate real-to-real reference macro-F1: 0.890
- torch=2.9.1+cu128 cuda=12.8

## 逐 seed 指标 (full-R 测试集)

| seed | macro_f1 | balanced_accuracy | f1_ship | recall_ship | f1_airplane | recall_airplane | aircraft_recall | accuracy | loss↓ | plateau_ok | nan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5518 | 0.7226 | 0.7487 | 0.6119 | 0.3548 | 0.8333 | 0.8333 | 0.6383 | True | False | False |
| 1 | 0.6440 | 0.7828 | 0.8436 | 0.7474 | 0.4444 | 0.8182 | 0.8182 | 0.7559 | True | False | False |
| 2 | 0.6301 | 0.7090 | 0.8622 | 0.7967 | 0.3981 | 0.6212 | 0.6212 | 0.7758 | True | False | False |
| 3 | 0.6207 | 0.7028 | 0.8546 | 0.7844 | 0.3868 | 0.6212 | 0.6212 | 0.7649 | True | False | False |
| 4 | 0.6076 | 0.6406 | 0.8779 | 0.8419 | 0.3372 | 0.4394 | 0.4394 | 0.7939 | True | False | False |
| 5 | 0.4644 | 0.6228 | 0.6542 | 0.5031 | 0.2745 | 0.7424 | 0.7424 | 0.5316 | True | False | False |
| 6 | 0.5357 | 0.6547 | 0.7616 | 0.6427 | 0.3099 | 0.6667 | 0.6667 | 0.6456 | True | False | False |
| 7 | 0.5860 | 0.7027 | 0.8099 | 0.7084 | 0.3622 | 0.6970 | 0.6970 | 0.7071 | True | False | False |
| 8 | 0.5535 | 0.6590 | 0.7877 | 0.6817 | 0.3194 | 0.6364 | 0.6364 | 0.6763 | True | False | False |
| 9 | 0.6431 | 0.7317 | 0.8651 | 0.7967 | 0.4211 | 0.6667 | 0.6667 | 0.7812 | True | False | False |
| 10 | 0.5996 | 0.6884 | 0.8364 | 0.7556 | 0.3628 | 0.6212 | 0.6212 | 0.7396 | True | False | False |
| 11 | 0.4895 | 0.6854 | 0.6649 | 0.5072 | 0.3140 | 0.8636 | 0.8636 | 0.5497 | True | False | False |

## 均值 ± std (跨 12 seed)

| metric | mean | std |
|---|---|---|
| macro_f1 | 0.5772 | 0.0565 |
| balanced_accuracy | 0.6919 | 0.0421 |
| f1_ship | 0.7972 | 0.0731 |
| recall_ship | 0.6982 | 0.1075 |
| f1_airplane | 0.3571 | 0.0475 |
| recall_airplane | 0.6856 | 0.1120 |
| aircraft_recall | 0.6856 | 0.1120 |
| accuracy | 0.6967 | 0.0857 |

## 收敛 sanity
- loss 末<首: 12/12 ; 无 NaN: 12/12 ; plateau_ok: 0/12

_注: 本 summary 只汇总, 不做配对 Δ 判定 (判定在 analyze_paired_delta.py)。_
