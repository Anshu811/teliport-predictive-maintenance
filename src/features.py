"""Data engineering for C-MAPSS: cleaning, regime normalization, causal rolling features.

Every feature at cycle t uses only data from cycles <= t, so the exact same code
works for batch training and for live streaming data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from cmapss_data import OP_COLS, SUBSET_INFO

SELECTED_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
                    "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
DEFAULT_RUL_CAP = 125


class CMAPSSPreprocessor:
    """sklearn-style fit/transform. Fit on TRAIN only, then reuse for test / new data."""

    def __init__(self, n_regimes=1, sensors=SELECTED_SENSORS, mean_windows=(5, 15, 30),
                 std_window=10, slope_windows=(10, 30), ewm_span=10,
                 baseline_cycles=10, clip_z=6.0, spike_z=4.0, random_state=42):
        self.n_regimes = n_regimes
        self.sensors = list(sensors)
        self.mean_windows = tuple(mean_windows)
        self.std_window = std_window
        self.slope_windows = tuple(slope_windows)
        self.ewm_span = ewm_span
        self.baseline_cycles = baseline_cycles
        self.clip_z = clip_z
        self.spike_z = spike_z
        self.random_state = random_state

    @classmethod
    def for_subset(cls, fd, **kw):
        return cls(n_regimes=SUBSET_INFO[fd]["conditions"], **kw)

    # ---------- cleaning ----------
    def _clean(self, df):
        cols = OP_COLS + self.sensors
        out = df.copy()
        out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")       # bad strings -> NaN
        out = (out.drop_duplicates(subset=["unit", "cycle"], keep="last")  # duplicate readings
                  .sort_values(["unit", "cycle"]).reset_index(drop=True))  # out-of-order data
        out[cols] = out.groupby("unit")[cols].ffill()                      # carry last good reading
        out[cols] = out.groupby("unit")[cols].bfill()                      # gap at the very start
        out[cols] = out[cols].fillna(self.fill_values_)                    # whole column missing
        return out

    # ---------- fit ----------
    def fit(self, df):
        self.fill_values_ = df[OP_COLS + self.sensors].median(numeric_only=True)
        clean = self._clean(df)
        ops = clean[OP_COLS].to_numpy(dtype=float)
        self.op_scaler_ = StandardScaler().fit(ops)
        self.kmeans_ = KMeans(n_clusters=self.n_regimes, n_init=10,
                              random_state=self.random_state).fit(self.op_scaler_.transform(ops))
        reg = self.kmeans_.labels_
        self.mu_ = clean[self.sensors].groupby(reg).mean()
        sd = clean[self.sensors].groupby(reg).std()
        self.sd_ = sd.where(sd > 1e-8, 1.0).fillna(1.0)
        self.feature_names_ = None
        return self

    def regimes(self, df):
        ops = df[OP_COLS].to_numpy(dtype=float)
        return self.kmeans_.predict(self.op_scaler_.transform(ops))

    # ---------- transform ----------
    def transform(self, df):
        out = self._clean(df)
        S = self.sensors
        reg = self.regimes(out)
        out["regime"] = reg
        # regime-wise z-score, then hard clip of extreme values
        z = (out[S].to_numpy() - self.mu_.loc[reg].to_numpy()) / self.sd_.loc[reg].to_numpy()
        z = np.clip(z, -self.clip_z, self.clip_z)
        Z = pd.DataFrame(z, columns=S, index=out.index)
        unit = out["unit"]
        # spike filter (causal Hampel-style): isolated outliers -> rolling median of last 5 cycles
        med = Z.groupby(unit).rolling(5, min_periods=1).median().reset_index(level=0, drop=True)
        Z = Z.mask((Z - med).abs() > self.spike_z, med)
        g = Z.groupby(unit)
        feats = {}

        for s in S:                                        # current (cleaned) value
            feats[f"{s}_z"] = Z[s]
        ewm = g.transform(lambda x: x.ewm(span=self.ewm_span, adjust=False).mean())
        for s in S:                                        # smoothed level (noise filter)
            feats[f"{s}_ewm"] = ewm[s]
        for w in self.mean_windows:                        # rolling averages
            m = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            for s in S:
                feats[f"{s}_mean{w}"] = m[s]
        sd = g.rolling(self.std_window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        for s in S:                                        # recent volatility
            feats[f"{s}_std{self.std_window}"] = sd[s]

        t = out["cycle"].astype(float)                     # rolling linear-regression slope = trend
        T = pd.concat([Z, Z.mul(t, axis=0).add_suffix("_tx")], axis=1)
        T["t"], T["tt"] = t, t * t
        gT = T.groupby(unit)
        for w in self.slope_windows:
            r = gT.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            var_t = (r["tt"] - r["t"] ** 2).to_numpy()
            for s in S:
                cov = (r[f"{s}_tx"] - r["t"] * r[s]).to_numpy()
                feats[f"{s}_slope{w}"] = np.where(var_t > 1e-9, cov / np.where(var_t > 1e-9, var_t, 1.0), 0.0)

        # drift from the engine's own early-life baseline (handles different initial wear)
        pos = out.groupby("unit").cumcount()
        early = Z.where(np.repeat((pos.to_numpy() < self.baseline_cycles)[:, None], len(S), axis=1))
        base = early.groupby(unit).expanding().mean().reset_index(level=0, drop=True).groupby(unit).ffill()
        for s in S:
            feats[f"{s}_dbase"] = ewm[s] - base[s]

        F = pd.DataFrame(feats, index=out.index)
        F.insert(0, "cycle", out["cycle"].astype(float))
        self.feature_names_ = list(F.columns)
        meta = [c for c in ["unit", "subset", "regime", "RUL"] if c in out.columns]
        return pd.concat([out[meta], F], axis=1)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def add_targets(df, cap=DEFAULT_RUL_CAP):
    out = df.copy()
    out["RUL_capped"] = out["RUL"].clip(upper=cap)
    return out


def last_cycle(df):
    """One row per engine: the most recent cycle (what we predict on for test engines)."""
    return df.groupby("unit", sort=True).tail(1).reset_index(drop=True)
