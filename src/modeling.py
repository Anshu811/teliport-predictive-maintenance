"""RUL modeling utilities: metrics, engine-level cross-validation, LightGBM training,
seed bagging and calibrated prediction intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

CRITICAL_RUL = 30          # "act now" zone used for weighting and critical-zone metrics
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                  random_state=42, n_jobs=-1, verbose=-1)


# ---------------- metrics ----------------
def nasa_score(y_true, y_pred):
    """Official PHM08 score: late predictions (d > 0) are punished harder than early ones."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred, float) - np.asarray(y_true, float)) ** 2)))


def evaluate(y_true, y_pred, critical=CRITICAL_RUL):
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    crit_true, crit_pred = y <= critical, p <= critical
    tp = int((crit_true & crit_pred).sum())
    return {
        "n": len(y),
        "RMSE": round(rmse(y, p), 2),
        "MAE": round(float(np.mean(np.abs(p - y))), 2),
        "NASA_score": round(nasa_score(y, p), 1),
        f"RMSE_RUL<={critical}": round(rmse(y[crit_true], p[crit_true]), 2) if crit_true.any() else np.nan,
        "critical_recall": round(tp / max(int(crit_true.sum()), 1), 3),
        "critical_precision": round(tp / max(int(crit_pred.sum()), 1), 3),
        "late_pred_%": round(100 * float(np.mean(p - y > 10)), 1),   # >10 cycles too optimistic
    }


# ---------------- training helpers ----------------
def make_lgbm(**overrides):
    return lgb.LGBMRegressor(**{**LGB_PARAMS, **overrides})


def sample_weights(rul, critical=CRITICAL_RUL, w=2.0):
    """Mild-imbalance handling: near-failure rows are rarer, so they count double."""
    return np.where(np.asarray(rul) <= critical, w, 1.0)


def val_rows(df, min_cycle=20, max_rul=150):
    """Rows that look like real test engines: some history, and within planning horizon."""
    return df[(df["cycle"] >= min_cycle) & (df["RUL"] <= max_rul)]


def fit_model(df, features, make_model, cap, weighted=True):
    model = make_model()
    y = df["RUL"].clip(upper=cap)
    if weighted:
        model.fit(df[features], y, sample_weight=sample_weights(df["RUL"]))
    else:
        model.fit(df[features], y)
    return model


def predict(model, df, features):
    return np.clip(model.predict(df[features]), 0, None)


def cross_validate(df, features, make_model, cap, weighted=True, n_splits=5):
    """GroupKFold by engine: no engine ever appears in both train and validation."""
    parts = []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=df["unit"])):
        model = fit_model(df.iloc[a], features, make_model, cap, weighted)
        v = val_rows(df.iloc[b])
        parts.append(pd.DataFrame({"unit": v["unit"].to_numpy(), "cycle": v["cycle"].to_numpy(),
                                   "fold": fold, "y_true": v["RUL"].to_numpy(),
                                   "y_pred": predict(model, v, features)}))
    oof = pd.concat(parts, ignore_index=True)
    return evaluate(oof["y_true"], oof["y_pred"]), oof


# ---------------- v2: seed bagging + uncertainty ----------------
class SeedBag:
    """Average of several LightGBM models trained with different random seeds.
    Removes 'lucky seed' variance, so reported results are stable and reproducible."""

    def __init__(self, params=None, seeds=(0, 1, 2, 3, 4)):
        self.params = dict(params or {})
        self.seeds = tuple(seeds)

    def fit(self, X, y, sample_weight=None):
        self.models_ = [make_lgbm(**{**self.params, "random_state": s}).fit(X, y, sample_weight=sample_weight)
                        for s in self.seeds]
        self.feature_names_in_ = list(X.columns)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)

    def predict_each(self, X):
        return np.vstack([m.predict(X) for m in self.models_])

    @property
    def feature_importances_(self):
        return np.mean([m.booster_.feature_importance("gain") for m in self.models_], axis=0)


class RULUncertainty:
    """Prediction intervals from out-of-fold residuals (split-conformal style), computed
    separately per predicted-RUL band because errors are small near failure, larger far away."""

    def __init__(self, coverage=0.90, edges=(0, 20, 40, 60, 80, 100, np.inf), min_points=30):
        self.coverage = coverage
        self.edges = tuple(edges)
        self.min_points = min_points

    def fit(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        r = y_true - y_pred
        lo_q, hi_q = (1 - self.coverage) / 2, 1 - (1 - self.coverage) / 2
        band = np.digitize(y_pred, self.edges[1:-1])
        g_lo, g_hi = np.quantile(r, lo_q), np.quantile(r, hi_q)
        self.lo_, self.hi_ = [], []
        for b in range(len(self.edges) - 1):
            rb = r[band == b]
            ok = len(rb) >= self.min_points
            self.lo_.append(float(np.quantile(rb, lo_q)) if ok else float(g_lo))
            self.hi_.append(float(np.quantile(rb, hi_q)) if ok else float(g_hi))
        return self

    def interval(self, y_pred):
        y_pred = np.asarray(y_pred, float)
        band = np.digitize(y_pred, self.edges[1:-1])
        lower = np.clip(y_pred + np.asarray(self.lo_)[band], 0, None)
        upper = y_pred + np.asarray(self.hi_)[band]
        return lower, upper

    def table(self):
        labels = [f"{self.edges[i]:.0f}-{self.edges[i+1]:.0f}" for i in range(len(self.edges) - 1)]
        return pd.DataFrame({"pred_RUL_band": labels, "lower_offset": np.round(self.lo_, 1),
                             "upper_offset": np.round(self.hi_, 1)})
