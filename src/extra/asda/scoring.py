"""Scoring used for the archived ASDA runs: per-run adjudication metrics and seed-paired t intervals.

The executed evaluator imported these functions from two helper modules of the working project that are not part of
this repository. Their code is reproduced here unchanged (only docstrings and comments differ), so the public
evaluator computes exactly what produced the archived ASDA metrics and summary. tests/test_asda.py checks that they
agree with the paper's scoring in src/evaluation.
"""
import math
import statistics

import numpy as np
from scipy import stats

CLASSES = ["airplane", "ship"]
AIRPLANE, SHIP = 0, 1
SEEDS5 = [0, 1, 2, 3, 4]
DF = len(SEEDS5) - 1
T_CRIT = float(stats.t.ppf(0.975, DF))  # two-sided 95% paired-t critical value, df=4


def adjudication_metrics(y_true, y_pred, probs):
    """Symmetric per-run metrics: macro-F1, per-class precision/recall/F1/support, 2x2 confusion matrix
    (rows=true, cols=pred, order airplane, ship) and airplane average precision (None if one class).
    zero_division=0 throughout; accuracy is deliberately not reported."""
    from sklearn.metrics import (
        f1_score, precision_recall_fscore_support, confusion_matrix, average_precision_score,
    )
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); probs = np.asarray(probs)
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    per_class = {c: {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f[i]), "support": int(sup[i])} for i, c in enumerate(CLASSES)}
    y_air = (y_true == AIRPLANE).astype(int)
    if 0 < int(y_air.sum()) < len(y_air):
        airplane_pr_auc = float(average_precision_score(y_air, probs[:, AIRPLANE]))
    else:
        airplane_pr_auc = None
    return {
        "macro_f1": float(macro_f1), "per_class": per_class,
        "confusion_matrix": cm.tolist(), "cm_axis": "rows=true, cols=pred, order=[airplane, ship]",
        "airplane_pr_auc": airplane_pr_auc,
    }


def seed_value_map(arm, metric):
    """Seed -> per-seed value of one metric for an aggregate arm record."""
    out = {}
    for pt in arm["per_seed_points"]:
        out[pt["seed"]] = pt["values"][metric]
    return out


def paired_delta(arm_a, arm_b, metric, seeds=SEEDS5):
    """Seed-matched differences a[s] - b[s] with mean, sample SD and paired-t 95% interval (df=4, not bootstrap)."""
    va = seed_value_map(arm_a, metric)
    vb = seed_value_map(arm_b, metric)
    deltas = [va[s] - vb[s] for s in seeds]
    n = len(deltas)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    lo = mean - T_CRIT * se
    hi = mean + T_CRIT * se
    return dict(
        metric=metric,
        seeds=list(seeds),
        deltas=deltas,
        mean=mean,
        sd=sd,
        ci_lo=lo,
        ci_hi=hi,
        n=n,
        t_crit=T_CRIT,
        df=DF,
        ci_method="paired_t",
    )
