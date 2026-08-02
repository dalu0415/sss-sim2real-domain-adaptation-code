# E2 ablation summary -- arm2

- arm dataset ID: arm2_selected_source
- n_seed present: 12 / 12  (seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
- separate real-to-real reference macro-F1: 0.890
- torch=2.9.1+cu128 cuda=12.8

## 逐 seed 指标 (full-R 测试集)

| seed | macro_f1 | balanced_accuracy | f1_ship | recall_ship | f1_airplane | recall_airplane | aircraft_recall | accuracy | loss↓ | plateau_ok | nan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5354 | 0.6792 | 0.7463 | 0.6160 | 0.3245 | 0.7424 | 0.7424 | 0.6311 | True | False | False |
| 1 | 0.6210 | 0.6534 | 0.8849 | 0.8522 | 0.3571 | 0.4545 | 0.4545 | 0.8047 | True | False | False |
| 2 | 0.4944 | 0.6018 | 0.7239 | 0.5975 | 0.2649 | 0.6061 | 0.6061 | 0.5986 | True | False | False |
| 3 | 0.6025 | 0.6905 | 0.8390 | 0.7598 | 0.3661 | 0.6212 | 0.6212 | 0.7432 | True | False | False |
| 4 | 0.5758 | 0.6620 | 0.8197 | 0.7331 | 0.3319 | 0.5909 | 0.5909 | 0.7161 | True | False | False |
| 5 | 0.5169 | 0.6348 | 0.7423 | 0.6181 | 0.2915 | 0.6515 | 0.6515 | 0.6221 | True | False | False |
| 6 | 0.6095 | 0.6337 | 0.8856 | 0.8583 | 0.3333 | 0.4091 | 0.4091 | 0.8047 | True | False | False |
| 7 | 0.5774 | 0.6540 | 0.8273 | 0.7474 | 0.3274 | 0.5606 | 0.5606 | 0.7251 | True | False | False |
| 8 | 0.4561 | 0.6542 | 0.6214 | 0.4600 | 0.2909 | 0.8485 | 0.8485 | 0.5063 | True | False | False |
| 9 | 0.5886 | 0.6667 | 0.8348 | 0.7577 | 0.3423 | 0.5758 | 0.5758 | 0.7360 | True | False | False |
| 10 | 0.5171 | 0.5888 | 0.7765 | 0.6776 | 0.2578 | 0.5000 | 0.5000 | 0.6564 | True | False | False |
| 11 | 0.4855 | 0.6091 | 0.7023 | 0.5667 | 0.2687 | 0.6515 | 0.6515 | 0.5769 | True | False | False |

## 均值 ± std (跨 12 seed)

| metric | mean | std |
|---|---|---|
| macro_f1 | 0.5484 | 0.0522 |
| balanced_accuracy | 0.6440 | 0.0300 |
| f1_ship | 0.7837 | 0.0759 |
| recall_ship | 0.6870 | 0.1142 |
| f1_airplane | 0.3131 | 0.0354 |
| recall_airplane | 0.6010 | 0.1146 |
| aircraft_recall | 0.6010 | 0.1146 |
| accuracy | 0.6768 | 0.0891 |

## 收敛 sanity
- loss 末<首: 12/12 ; 无 NaN: 12/12 ; plateau_ok: 0/12

_注: 本 summary 只汇总, 不做配对 Δ 判定 (判定在 analyze_paired_delta.py)。_
