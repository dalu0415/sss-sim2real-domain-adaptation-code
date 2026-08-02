# Simulation-validity E2 paired-difference report

- per_run 目录: `evidence/simulation_validity/per_run`
- 配对结构: 同一组 12 seed (SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) 跑全 4 臂; 每 (臂,seed) 在固定全 R 测试集算一点。
- 主 CI = 配对 t (mean(d) ± t(11,0.975)·std(d,ddof=1)/√12); 交叉核 = 对 12 个 d_i 配对 bootstrap (B=10000)。
- 固定判定规则: primary 95% paired-t CI 排除 0 时标为 EFFECTIVE；包含 0 时只标为 INCONCLUSIVE。
- 未作多重比较校正；secondary 指标只能作为描述性结果。
- ceiling macro-F1 = 0.890，来自单独的 KLSG-II 5-fold OOF 监督参照。

## Cell 完整性

- present=48  missing=0 (期望 48 = 4臂×12seed)

## Target image-byte fingerprint consistency

- R_signature_md5 唯一集合 (应只 1 个): ['b4f9077323250ff506ac6580cf82d022']
- 全部 full cell 使用同一组目标输入图片字节。

## 6 个预先指定的比较块 × 配对 Δ

### 0->1  (low-level treatment increment (both classes))

- 数据源: per_run/ (全 490 张)

- 主判指标:
  - **macro_f1**: Δ=0.1588  SE=0.0249  |Δ|/SE=6.38  (n=12)
    - 主CI(配对t): [0.1040, 0.2136]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.1120, 0.2061]  一致=True (endpoint_rel_gap lo=0.15 hi=0.14)
    - **判定: EFFECTIVE** (improve(+))
- 照报指标:
  - **f1_ship**: Δ=0.2576  SE=0.0449  |Δ|/SE=5.73  (n=12)
    - 主CI(配对t): [0.1587, 0.3565]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.1744, 0.3436]  一致=True (endpoint_rel_gap lo=0.16 hi=0.13)
    - **判定: EFFECTIVE** (improve(+))
  - **recall_ship**: Δ=0.2610  SE=0.0408  |Δ|/SE=6.40  (n=12)
    - 主CI(配对t): [0.1712, 0.3507]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.1831, 0.3380]  一致=True (endpoint_rel_gap lo=0.13 hi=0.14)
    - **判定: EFFECTIVE** (improve(+))
  - **f1_airplane**: Δ=0.0600  SE=0.0096  |Δ|/SE=6.25  (n=12)
    - 主CI(配对t): [0.0389, 0.0811]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0427, 0.0781]  一致=True (endpoint_rel_gap lo=0.18 hi=0.14)
    - **判定: EFFECTIVE** (improve(+))
  - **recall_airplane**: Δ=-0.1124  SE=0.0311  |Δ|/SE=3.62  (n=12)
    - 主CI(配对t): [-0.1807, -0.0440]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.1730, -0.0543]  一致=True (endpoint_rel_gap lo=0.11 hi=0.15)
    - **判定: EFFECTIVE** (degrade(-))
  - **balanced_accuracy**: Δ=0.0743  SE=0.0126  |Δ|/SE=5.89  (n=12)
    - 主CI(配对t): [0.0465, 0.1021]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0506, 0.0983]  一致=True (endpoint_rel_gap lo=0.15 hi=0.14)
    - **判定: EFFECTIVE** (improve(+))

### 1->2  (ship morphology increment)

- 数据源: per_run/ (全 490 张)

- 主判指标:
  - **f1_ship**: Δ=0.0719  SE=0.0370  |Δ|/SE=1.94  (n=12)
    - 主CI(配对t): [-0.0097, 0.1534]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0070, 0.1456]  一致=False (endpoint_rel_gap lo=0.20 hi=0.10)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)
    - ⚠ 两CI判定不一致 (主t与bootstrap对'是否含0'不同) -> 标注复核, 仍以主t为准。
  - **recall_ship**: Δ=0.1004  SE=0.0519  |Δ|/SE=1.94  (n=12)
    - 主CI(配对t): [-0.0137, 0.2146]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0096, 0.2040]  一致=False (endpoint_rel_gap lo=0.20 hi=0.09)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)
    - ⚠ 两CI判定不一致 (主t与bootstrap对'是否含0'不同) -> 标注复核, 仍以主t为准。
- 照报指标:
  - **macro_f1**: Δ=0.0350  SE=0.0243  |Δ|/SE=1.44  (n=12)
    - 主CI(配对t): [-0.0185, 0.0885]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.0096, 0.0821]  一致=True (endpoint_rel_gap lo=0.17 hi=0.12)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)
  - **balanced_accuracy**: Δ=-0.0148  SE=0.0135  |Δ|/SE=1.09  (n=12)
    - 主CI(配对t): [-0.0446, 0.0150]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.0416, 0.0089]  一致=True (endpoint_rel_gap lo=0.10 hi=0.20)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)

### 2->3  (internal-structure increment over all 400 ships)

- 数据源: per_run/ (全 490 张)

- 主判指标:
  - **f1_ship**: Δ=0.0136  SE=0.0245  |Δ|/SE=0.55  (n=12)
    - 主CI(配对t): [-0.0404, 0.0675]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.0314, 0.0615]  一致=True (endpoint_rel_gap lo=0.17 hi=0.11)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)
  - **recall_ship**: Δ=0.0111  SE=0.0373  |Δ|/SE=0.30  (n=12)
    - 主CI(配对t): [-0.0709, 0.0931]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.0578, 0.0832]  一致=True (endpoint_rel_gap lo=0.16 hi=0.12)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)
- 照报指标:
  - **macro_f1**: Δ=0.0288  SE=0.0170  |Δ|/SE=1.69  (n=12)
    - 主CI(配对t): [-0.0086, 0.0662]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.0035, 0.0610]  一致=True (endpoint_rel_gap lo=0.14 hi=0.14)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)
  - **balanced_accuracy**: Δ=0.0479  SE=0.0141  |Δ|/SE=3.40  (n=12)
    - 主CI(配对t): [0.0169, 0.0789]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0216, 0.0739]  一致=True (endpoint_rel_gap lo=0.15 hi=0.16)
    - **判定: EFFECTIVE** (improve(+))

### 2->3 (structure subset)  (internal-structure increment over 263 changed ships)

- **数据源: per_run_structure_subset/ (训练=90 airplane + 263 张实际结构差异 ship)**
- _注: 原内部代号为 268-subset，冻结清单实际为 263 张差异船和 137 张 no-op 船。_

- 主判指标:
  - **f1_ship**: Δ=0.0430  SE=0.0227  |Δ|/SE=1.90  (n=12)
    - 主CI(配对t): [-0.0068, 0.0929]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0022, 0.0862]  一致=False (endpoint_rel_gap lo=0.18 hi=0.13)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)
    - ⚠ 两CI判定不一致 (主t与bootstrap对'是否含0'不同) -> 标注复核, 仍以主t为准。
  - **recall_ship**: Δ=0.0501  SE=0.0281  |Δ|/SE=1.79  (n=12)
    - 主CI(配对t): [-0.0116, 0.1119]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.0010, 0.1040]  一致=True (endpoint_rel_gap lo=0.17 hi=0.13)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)
- 照报指标:
  - **macro_f1**: Δ=0.0293  SE=0.0132  |Δ|/SE=2.22  (n=12)
    - 主CI(配对t): [0.0003, 0.0584]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0049, 0.0543]  一致=True (endpoint_rel_gap lo=0.16 hi=0.14)
    - **判定: EFFECTIVE** (improve(+))
  - **balanced_accuracy**: Δ=0.0124  SE=0.0072  |Δ|/SE=1.73  (n=12)
    - 主CI(配对t): [-0.0033, 0.0282]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.0010, 0.0260]  一致=True (endpoint_rel_gap lo=0.15 hi=0.14)
    - **判定: INCONCLUSIVE(95% paired-t CI includes zero)** (n/a)

### 0->3  (all-stage endpoint relative to arm0)

- 数据源: per_run/ (全 490 张)

- 主判指标:
  - **macro_f1**: Δ=0.2226  SE=0.0300  |Δ|/SE=7.41  (n=12)
    - 主CI(配对t): [0.1565, 0.2886]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.1664, 0.2794]  一致=True (endpoint_rel_gap lo=0.15 hi=0.14)
    - **判定: EFFECTIVE** (improve(+))
- 照报指标:
  - **f1_ship**: Δ=0.3430  SE=0.0528  |Δ|/SE=6.50  (n=12)
    - 主CI(配对t): [0.2269, 0.4592]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.2469, 0.4441]  一致=True (endpoint_rel_gap lo=0.17 hi=0.13)
    - **判定: EFFECTIVE** (improve(+))
  - **recall_ship**: Δ=0.3725  SE=0.0441  |Δ|/SE=8.46  (n=12)
    - 主CI(配对t): [0.2756, 0.4695]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.2888, 0.4543]  一致=True (endpoint_rel_gap lo=0.14 hi=0.16)
    - **判定: EFFECTIVE** (improve(+))
  - **f1_airplane**: Δ=0.1021  SE=0.0144  |Δ|/SE=7.08  (n=12)
    - 主CI(配对t): [0.0704, 0.1339]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0736, 0.1271]  一致=True (endpoint_rel_gap lo=0.10 hi=0.21)
    - **判定: EFFECTIVE** (improve(+))
  - **recall_airplane**: Δ=-0.1578  SE=0.0370  |Δ|/SE=4.26  (n=12)
    - 主CI(配对t): [-0.2394, -0.0763]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.2260, -0.0871]  一致=True (endpoint_rel_gap lo=0.16 hi=0.13)
    - **判定: EFFECTIVE** (degrade(-))
  - **balanced_accuracy**: Δ=0.1073  SE=0.0203  |Δ|/SE=5.29  (n=12)
    - 主CI(配对t): [0.0626, 0.1520]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0663, 0.1432]  一致=True (endpoint_rel_gap lo=0.08 hi=0.20)
    - **判定: EFFECTIVE** (improve(+))

### 0->2  (selected arm2 source relative to arm0)

- 数据源: per_run/ (全 490 张)

- 主判指标:
  - **macro_f1**: Δ=0.1938  SE=0.0320  |Δ|/SE=6.05  (n=12)
    - 主CI(配对t): [0.1233, 0.2642]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.1360, 0.2562]  一致=True (endpoint_rel_gap lo=0.18 hi=0.11)
    - **判定: EFFECTIVE** (improve(+))
- 照报指标:
  - **f1_ship**: Δ=0.3294  SE=0.0553  |Δ|/SE=5.95  (n=12)
    - 主CI(配对t): [0.2077, 0.4512]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.2309, 0.4373]  一致=True (endpoint_rel_gap lo=0.19 hi=0.11)
    - **判定: EFFECTIVE** (improve(+))
  - **recall_ship**: Δ=0.3614  SE=0.0495  |Δ|/SE=7.30  (n=12)
    - 主CI(配对t): [0.2524, 0.4704]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.2748, 0.4603]  一致=True (endpoint_rel_gap lo=0.21 hi=0.09)
    - **判定: EFFECTIVE** (improve(+))
  - **f1_airplane**: Δ=0.0581  SE=0.0113  |Δ|/SE=5.16  (n=12)
    - 主CI(配对t): [0.0333, 0.0829]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0366, 0.0791]  一致=True (endpoint_rel_gap lo=0.13 hi=0.15)
    - **判定: EFFECTIVE** (improve(+))
  - **recall_airplane**: Δ=-0.2424  SE=0.0340  |Δ|/SE=7.13  (n=12)
    - 主CI(配对t): [-0.3172, -0.1676]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [-0.3106, -0.1831]  一致=True (endpoint_rel_gap lo=0.09 hi=0.21)
    - **判定: EFFECTIVE** (degrade(-))
  - **balanced_accuracy**: Δ=0.0595  SE=0.0174  |Δ|/SE=3.42  (n=12)
    - 主CI(配对t): [0.0213, 0.0977]  (hardcoded_t(11,0.975)=2.200985)
    - 交叉核(配对bootstrap B=10000): [0.0264, 0.0908]  一致=True (endpoint_rel_gap lo=0.14 hi=0.18)
    - **判定: EFFECTIVE** (improve(+))

## 天花板 gap 轨迹 (每臂 macro-F1 -> ceiling=0.890)

| arm | macro_f1 mean | std | gap_to_ceiling | n_seed |
|---|---|---|---|---|
| 0 | 0.3546 | 0.1208 | 0.5354 | 12 |
| 1 | 0.5134 | 0.0772 | 0.3766 | 12 |
| 2 | 0.5484 | 0.0522 | 0.3416 | 12 |
| 3 | 0.5772 | 0.0565 | 0.3128 | 12 |

## format-probe 轨迹 (每臂合成 vs KLSG 低级可分性 AUC; DIRECTION-ONLY)

_Arm2 and arm3 are interpreted directionally; their AUC values are not evaluated against a new pass/fail threshold._

| arm | ship AUC | airplane AUC |
|---|---|---|
| 0 | 0.9432 | 0.8164 |
| 1 | 0.8228 | 0.8229 |
| 2 | 0.7247 | 0.8229 |
| 3 | 0.7295 | 0.8229 |

---
_The code reports the frozen numerical rule only. Interpretation, target-aware selection limits, and the distinction between primary and secondary metrics are documented in src/simulation_validity/README.md._
