"""Credit-risk evaluation metrics.

Lenders speak in Gini and KS, data scientists in AUC. We report both, plus
PR-AUC (honest on imbalanced data) and Brier score (probability accuracy,
which matters once PD drives pricing).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: the largest gap between the cumulative share of
    defaulters and of good payers as the score threshold moves.
    0 = no separation, 1 = perfect separation. Equals max(TPR - FPR)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def credit_metrics(y_true, y_score) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = roc_auc_score(y_true, y_score)
    return {
        "n": int(len(y_true)),
        "default_rate": round(float(y_true.mean()), 4),
        "roc_auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 5),
    }
