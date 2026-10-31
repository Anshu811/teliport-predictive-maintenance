"""Unsupervised degradation (anomaly) detection for C-MAPSS.

Learns what a HEALTHY engine looks like (rows far from failure) and scores how far each
reading has drifted from it. Uses the "drift from own early-life baseline" features, so
deviation in ANY direction counts -> works for both fault modes (FD003/FD004) without labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

HEALTHY_RUL = 125      # same boundary as the RUL cap: > 125 cycles from failure = healthy

SENSOR_NAMES = {
    "s2": "LPC outlet temperature (T24)", "s3": "HPC outlet temperature (T30)",
    "s4": "LPT outlet temperature (T50)", "s7": "HPC outlet pressure (P30)",
    "s8": "Fan speed (Nf)", "s9": "Core speed (Nc)",
    "s11": "HPC static pressure (Ps30)", "s12": "Fuel flow / Ps30 ratio (phi)",
    "s13": "Corrected fan speed (NRf)", "s14": "Corrected core speed (NRc)",
    "s15": "Bypass ratio (BPR)", "s17": "Bleed enthalpy (htBleed)",
    "s20": "HPT coolant bleed (W31)", "s21": "LPT coolant bleed (W32)",
}


class HealthMonitor:
    def __init__(self, method="mahalanobis", suffix="_dbase", quantile=0.99,
                 persistence=5, healthy_rul=HEALTHY_RUL, random_state=42):
        self.method = method
        self.suffix = suffix
        self.quantile = quantile
        self.persistence = persistence
        self.healthy_rul = healthy_rul
        self.random_state = random_state

    def _X(self, df):
        return df[self.features_].to_numpy(dtype=float)

    def _raw_score(self, X):
        if self.method == "mahalanobis":
            d = X - self.cov_.location_
            return np.einsum("ij,jk,ik->i", d, self.cov_.precision_, d)
        return -self.iforest_.score_samples(X)

    def fit(self, df, sensors):
        self.sensors_ = list(sensors)
        self.features_ = [f"{s}{self.suffix}" for s in self.sensors_]
        X = self._X(df[df["RUL"] > self.healthy_rul])          # healthy rows only
        if self.method == "mahalanobis":
            self.cov_ = LedoitWolf().fit(X)
        else:
            self.iforest_ = IsolationForest(n_estimators=300, random_state=self.random_state).fit(X)
        self.threshold_ = float(np.quantile(self._raw_score(X), self.quantile))
        return self

    def transform(self, df):
        """anomaly_score > 1 = outside the healthy envelope; alarm = score > 1 for k cycles in a row."""
        out = df[["unit", "cycle"]].copy()
        out["anomaly_score"] = self._raw_score(self._X(df)) / self.threshold_
        raw = (out["anomaly_score"] > 1.0).astype(int)
        streak = raw.groupby(out["unit"]).rolling(self.persistence, min_periods=self.persistence).sum()
        out["alarm"] = streak.reset_index(level=0, drop=True).reindex(out.index).fillna(0) >= self.persistence
        return out

    def top_contributors(self, row, n=3):
        """Which sensors push this reading out of the healthy envelope (Mahalanobis only)."""
        d = np.asarray(row[self.features_], dtype=float) - self.cov_.location_
        contrib = np.clip(d * (self.cov_.precision_ @ d), 0, None)
        share = contrib / max(contrib.sum(), 1e-9)
        order = np.argsort(contrib)[::-1][:n]
        return [(self.sensors_[i], SENSOR_NAMES.get(self.sensors_[i], self.sensors_[i]),
                 round(float(100 * share[i]), 1)) for i in order]


def detection_report(df, flags, healthy_rul=HEALTHY_RUL, critical=30):
    """Lead time and false alarms, measured on run-to-failure engines."""
    m = df[["unit", "cycle", "RUL"]].join(flags[["anomaly_score", "alarm"]])
    healthy = m[m["RUL"] > healthy_rul]
    first = m[m["alarm"]].groupby("unit")["RUL"].max()           # RUL when the first alarm fired
    n_units = m["unit"].nunique()
    true_alarms = first[first <= healthy_rul]
    return {
        "engines": int(n_units),
        "false_alarm_rate_%": round(float(100 * healthy["alarm"].mean()), 2),
        "engines_with_early_alarm_%": round(float(100 * (first > healthy_rul).sum() / n_units), 1),
        "detected_before_failure_%": round(float(100 * m.groupby("unit")["alarm"].any().mean()), 1),
        f"detected_before_RUL{critical}_%": round(float(100 * (first > critical).sum() / n_units), 1),
        "median_lead_time_cycles": float(true_alarms.median()) if len(true_alarms) else np.nan,
    }
