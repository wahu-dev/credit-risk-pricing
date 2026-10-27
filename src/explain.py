"""Explainability: SHAP contributions and plain-language reason codes.

SHAP splits one applicant's score into a contribution per feature, measured in
log-odds of default:
    log-odds(PD) = base value + sum(contribution of each feature)
A positive contribution pushed this applicant's risk UP; negative pushed it down.

LightGBM and XGBoost compute exact TreeSHAP natively (`pred_contrib`), the same
algorithm as the `shap` library's TreeExplainer, so no extra dependency.

Ensembles: members are combined in log-odds space. For the champion (average of
LightGBM and XGBoost probabilities) we average the members' log-odds
contributions. That decomposes the average of the members' log-odds exactly,
which tracks the averaged probability very closely (checked and reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.boosting import KINDS
from src.ensemble import BoostedMember, ScorecardMember

BASE = "_base_value"


def member_contributions(member, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature log-odds contributions for one member, plus the base value.
    Columns: every column of X (0 where the member does not use it) + BASE."""
    out = pd.DataFrame(0.0, index=X.index, columns=list(X.columns) + [BASE])
    if isinstance(member, BoostedMember):
        Xm = X[member.columns]
        if member.kind == "lightgbm":
            values = member.model_.predict(Xm, pred_contrib=True)
        else:
            import xgboost as xgb
            values = member.model_.get_booster().predict(xgb.DMatrix(Xm), pred_contribs=True)
        out[member.columns] = values[:, :-1]
        out[BASE] = values[:, -1]
    elif isinstance(member, ScorecardMember):
        # Linear model: contribution = coefficient x standardised value
        Z = member.prep_.transform(X[member.features])[member.used_]
        coefs = member.model_.coef_[0]
        out[member.used_] = Z.to_numpy() * coefs
        out[BASE] = float(member.model_.intercept_[0])
    else:
        raise TypeError(f"No contribution method for {type(member).__name__}")
    return out


def champion_contributions(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Contributions for whichever model won stage 2, in log-odds."""
    champion = bundle["champion"]
    members = bundle["members"]
    if champion == "average_boosted":
        parts = [member_contributions(members[k], X) for k in KINDS]
        return sum(parts) / len(parts)
    if champion == "stacked":
        stacker = bundle["stacker"]
        total = None
        for name, w in zip(stacker.members, stacker.model_.coef_[0]):
            part = member_contributions(members[name], X) * w
            total = part if total is None else total + part
        total[BASE] += float(stacker.model_.intercept_[0])
        return total
    return member_contributions(members[champion], X)


def log_odds_from_contributions(contrib: pd.DataFrame) -> np.ndarray:
    return contrib.sum(axis=1).to_numpy()


def readable(feature: str, labels: dict[str, str]) -> str:
    return labels.get(feature, feature.replace("_", " ").capitalize())


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _about_typical(value, typical, tolerance: float = 0.05) -> bool:
    """True when the value rounds to the typical value or is within 5% of it."""
    if pd.isna(typical):
        return False
    v, t = float(value), float(typical)
    return _fmt(v) == _fmt(t) or abs(v - t) <= tolerance * max(abs(t), 1e-9)


def binary_features(X: pd.DataFrame) -> set[str]:
    """Numeric features whose known values are only 0 and 1 (yes/no flags)."""
    out = set()
    for col in X.select_dtypes(include=[np.number]).columns:
        values = set(pd.unique(X[col].dropna()))
        if values and values <= {0, 1}:
            out.add(col)
    return out


def reason_codes(contrib: pd.DataFrame, X: pd.DataFrame, typical: pd.Series, labels: dict[str, str],
                 top_k: int = 3, categories: dict[str, list[str]] | None = None,
                 binary: set[str] | None = None) -> pd.DataFrame:
    """Top risk-increasing factors per applicant, in plain language.

    Only positive contributions count: a reason for a decline must be something
    that made the applicant look riskier. Each reason compares the applicant's
    value with the typical (median) value of the training population.

    Factors where the applicant is about typical are skipped. SHAP measures a
    push relative to the average *prediction*, so even median values get small
    pushes; "your employment length is about typical" is not a usable reason
    for a customer or a credit officer. An applicant with nothing unusual gets
    no reasons, which is the honest answer.
    """
    categories = categories or {}
    binary = binary or set()
    features = [c for c in contrib.columns if c != BASE]
    values = contrib[features].to_numpy()
    rows = []
    for i in range(len(contrib)):
        order = np.argsort(-values[i])
        reasons = []
        for j in order:
            if values[i, j] <= 0 or len(reasons) == top_k:
                break
            f = features[j]
            v = X.iloc[i][f]
            t = typical.get(f, np.nan)
            if f in categories:
                code = int(v) if pd.notna(v) else -1
                name = categories[f][code] if 0 <= code < len(categories[f]) else "unknown"
                comparison = f"is '{name}'"
            elif pd.isna(v):
                comparison = "is missing"
            elif f in binary:
                if pd.notna(t) and float(v) == round(float(t)):
                    continue  # same as most applicants: not a reason
                comparison = ": yes" if float(v) == 1 else ": no"
            elif _about_typical(v, t):
                continue
            else:
                comparison = f"is {'higher' if float(v) > float(t) else 'lower'} than typical ({_fmt(v)} vs {_fmt(t)})"
            reasons.append(f"{readable(f, labels)}{'' if comparison.startswith(':') else ' '}{comparison}")
        row = {"row": contrib.index[i]}
        for k in range(top_k):
            row[f"reason_{k + 1}"] = reasons[k] if k < len(reasons) else ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("row")


def global_importance(contrib: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Mean absolute contribution per feature: how much each one moves scores
    on average, in either direction."""
    features = [c for c in contrib.columns if c != BASE]
    imp = contrib[features].abs().mean().sort_values(ascending=False)
    out = imp.rename("mean_abs_log_odds").reset_index().rename(columns={"index": "feature"})
    out["label"] = [readable(f, labels) for f in out["feature"]]
    out["share"] = out["mean_abs_log_odds"] / out["mean_abs_log_odds"].sum()
    return out
