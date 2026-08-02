# E2 ablation summary -- arm0

- arm dataset ID: arm0_native_baseline
- n_seed present: 12 / 12  (seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
- separate real-to-real reference macro-F1: 0.890
- torch=2.9.1+cu128 cuda=12.8

## 逐 seed 指标 (full-R 测试集)

| seed | macro_f1 | balanced_accuracy | f1_ship | recall_ship | f1_airplane | recall_airplane | aircraft_recall | accuracy | loss↓ | plateau_ok | nan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.3091 | 0.5661 | 0.3803 | 0.2382 | 0.2379 | 0.8939 | 0.8939 | 0.3165 | True | False | False |
| 1 | 0.5500 | 0.6905 | 0.7632 | 0.6386 | 0.3368 | 0.7424 | 0.7424 | 0.6510 | True | False | False |
| 2 | 0.3659 | 0.5633 | 0.4955 | 0.3388 | 0.2364 | 0.7879 | 0.7879 | 0.3924 | True | False | False |
| 3 | 0.4650 | 0.6127 | 0.6614 | 0.5133 | 0.2686 | 0.7121 | 0.7121 | 0.5371 | True | False | False |
| 4 | 0.5383 | 0.6968 | 0.7412 | 0.6057 | 0.3355 | 0.7879 | 0.7879 | 0.6275 | True | False | False |
| 5 | 0.3533 | 0.5958 | 0.4538 | 0.2977 | 0.2527 | 0.8939 | 0.8939 | 0.3689 | True | False | False |
| 6 | 0.2777 | 0.5586 | 0.3208 | 0.1930 | 0.2346 | 0.9242 | 0.9242 | 0.2803 | True | False | False |
| 7 | 0.1592 | 0.5050 | 0.1044 | 0.0554 | 0.2139 | 0.9545 | 0.9545 | 0.1627 | True | False | False |
| 8 | 0.2260 | 0.5475 | 0.2218 | 0.1253 | 0.2302 | 0.9697 | 0.9697 | 0.2260 | True | False | False |
| 9 | 0.3968 | 0.6035 | 0.5353 | 0.3737 | 0.2582 | 0.8333 | 0.8333 | 0.4286 | True | False | False |
| 10 | 0.4112 | 0.5862 | 0.5730 | 0.4148 | 0.2494 | 0.7576 | 0.7576 | 0.4557 | True | False | False |
| 11 | 0.2025 | 0.4883 | 0.1996 | 0.1129 | 0.2054 | 0.8636 | 0.8636 | 0.2025 | True | False | False |

## 均值 ± std (跨 12 seed)

| metric | mean | std |
|---|---|---|
| macro_f1 | 0.3546 | 0.1208 |
| balanced_accuracy | 0.5845 | 0.0602 |
| f1_ship | 0.4542 | 0.2057 |
| recall_ship | 0.3256 | 0.1840 |
| f1_airplane | 0.2550 | 0.0400 |
| recall_airplane | 0.8434 | 0.0824 |
| aircraft_recall | 0.8434 | 0.0824 |
| accuracy | 0.3874 | 0.1537 |

## 收敛 sanity
- loss 末<首: 12/12 ; 无 NaN: 12/12 ; plateau_ok: 0/12

_注: 本 summary 只汇总, 不做配对 Δ 判定 (判定在 analyze_paired_delta.py)。_
