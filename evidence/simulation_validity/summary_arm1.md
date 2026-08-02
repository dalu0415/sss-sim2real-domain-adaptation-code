# E2 ablation summary -- arm1

- arm dataset ID: arm1_low_level_realism
- n_seed present: 12 / 12  (seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
- separate real-to-real reference macro-F1: 0.890
- torch=2.9.1+cu128 cuda=12.8

## 逐 seed 指标 (full-R 测试集)

| seed | macro_f1 | balanced_accuracy | f1_ship | recall_ship | f1_airplane | recall_airplane | aircraft_recall | accuracy | loss↓ | plateau_ok | nan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5704 | 0.6444 | 0.8237 | 0.7433 | 0.3172 | 0.5455 | 0.5455 | 0.7197 | True | False | False |
| 1 | 0.5437 | 0.7009 | 0.7475 | 0.6140 | 0.3399 | 0.7879 | 0.7879 | 0.6347 | True | False | False |
| 2 | 0.5374 | 0.6512 | 0.7666 | 0.6509 | 0.3082 | 0.6515 | 0.6515 | 0.6510 | True | False | False |
| 3 | 0.5688 | 0.6949 | 0.7895 | 0.6776 | 0.3481 | 0.7121 | 0.7121 | 0.6817 | True | False | False |
| 4 | 0.6358 | 0.7031 | 0.8716 | 0.8152 | 0.4000 | 0.5909 | 0.5909 | 0.7884 | True | False | False |
| 5 | 0.5044 | 0.6446 | 0.7141 | 0.5770 | 0.2947 | 0.7121 | 0.7121 | 0.5931 | True | False | False |
| 6 | 0.3951 | 0.6432 | 0.5121 | 0.3470 | 0.2780 | 0.9394 | 0.9394 | 0.4177 | True | False | False |
| 7 | 0.4927 | 0.6664 | 0.6807 | 0.5298 | 0.3046 | 0.8030 | 0.8030 | 0.5624 | True | False | False |
| 8 | 0.4146 | 0.6169 | 0.5628 | 0.4004 | 0.2663 | 0.8333 | 0.8333 | 0.4521 | True | False | False |
| 9 | 0.5088 | 0.6532 | 0.7166 | 0.5791 | 0.3009 | 0.7273 | 0.7273 | 0.5967 | True | False | False |
| 10 | 0.6033 | 0.7150 | 0.8264 | 0.7331 | 0.3802 | 0.6970 | 0.6970 | 0.7288 | True | False | False |
| 11 | 0.3856 | 0.5722 | 0.5300 | 0.3717 | 0.2411 | 0.7727 | 0.7727 | 0.4195 | True | False | False |

## 均值 ± std (跨 12 seed)

| metric | mean | std |
|---|---|---|
| macro_f1 | 0.5134 | 0.0772 |
| balanced_accuracy | 0.6588 | 0.0390 |
| f1_ship | 0.7118 | 0.1146 |
| recall_ship | 0.5866 | 0.1452 |
| f1_airplane | 0.3149 | 0.0439 |
| recall_airplane | 0.7311 | 0.1026 |
| aircraft_recall | 0.7311 | 0.1026 |
| accuracy | 0.6038 | 0.1176 |

## 收敛 sanity
- loss 末<首: 12/12 ; 无 NaN: 12/12 ; plateau_ok: 0/12

_注: 本 summary 只汇总, 不做配对 Δ 判定 (判定在 analyze_paired_delta.py)。_
