"""Sensor-fault detection by analytical redundancy ("virtual sensors").

When an engine wears, its sensors drift TOGETHER in a physically consistent way. When one SENSOR
breaks (sudden bias, stuck value, drift), it stops agreeing with the other 13. For every sensor we
learn a small model that predicts it from the other sensors; a persistent, large disagreement means
that sensor (not the engine) is suspect. The suspect sensor can then be replaced by its virtual
value to give a backup RUL. Nothing is corrected silently: the agent only uses the backup to
ESCALATE (never to downgrade) and always sends the engine to human review.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


class SensorHealth:
    def __init__(self, quantile=0.9995, persistence=5, alpha=1.0, n_splits=3):
        self.quantile = quantile
        self.persistence = persistence
        self.alpha = alpha
        self.n_splits = n_splits

    # ---------- fit on TRAINING engines only ----------
    def fit(self, feats, prep):
        self.sensors_ = list(prep.sensors)
        S = self.sensors_
        units = feats["unit"].to_numpy()
        oof = np.zeros((len(feats), len(S)))
        self.slow_, self.fast_, self.sigma_ = {}, {}, {}
        for j, s in enumerate(S):
            others = [o for o in S if o != s]
            Xe, ye = feats[[f"{o}_ewm" for o in others]].to_numpy(), feats[f"{s}_ewm"].to_numpy()
            for a, b in GroupKFold(n_splits=self.n_splits).split(Xe, groups=units):     # unseen-engine residuals
                oof[b, j] = ye[b] - Ridge(alpha=self.alpha).fit(Xe[a], ye[a]).predict(Xe[b])
            self.slow_[s] = Ridge(alpha=self.alpha).fit(Xe, ye)                        # detector (smoothed)
            Xz, yz = feats[[f"{o}_z" for o in others]].to_numpy(), feats[f"{s}_z"].to_numpy()
            self.fast_[s] = Ridge(alpha=self.alpha).fit(Xz, yz)                        # virtual sensor (per cycle)
            self.sigma_[s] = float(oof[:, j].std() + 1e-6)
        z = np.abs(oof / np.array([self.sigma_[s] for s in S]))
        self.threshold_ = float(np.quantile(z.max(axis=1), self.quantile))
        return self

    # ---------- detection ----------
    def scores(self, feats):
        S = self.sensors_
        out = {}
        for s in S:
            others = [o for o in S if o != s]
            pred = self.slow_[s].predict(feats[[f"{o}_ewm" for o in others]].to_numpy())
            out[s] = (feats[f"{s}_ewm"].to_numpy() - pred) / self.sigma_[s]
        return pd.DataFrame(out, index=feats.index)

    def last_suspects(self, feats):
        """{unit: (sensor, score)} for engines whose worst sensor disagreed for `persistence` cycles in a row."""
        sc = self.scores(feats).abs()
        worst = sc.max(axis=1)
        over = (worst > self.threshold_).astype(int)
        streak = over.groupby(feats["unit"]).rolling(self.persistence, min_periods=self.persistence).sum()
        streak = streak.reset_index(level=0, drop=True).reindex(feats.index).fillna(0)
        flagged = streak >= self.persistence
        last_idx = feats.groupby("unit").tail(1).index
        res = {}
        for i in last_idx:
            if flagged.loc[i]:
                s = sc.loc[i].idxmax()
                res[int(feats.loc[i, "unit"])] = (s, round(float(sc.loc[i, s]), 1))
        return res

    # ---------- backup input: replace the suspect sensor by its virtual value ----------
    def repair_raw(self, raw, feats, prep, unit, sensor):
        S = self.sensors_
        others = [o for o in S if o != sensor]
        f = feats[feats["unit"] == unit]
        zhat = self.fast_[sensor].predict(f[[f"{o}_z" for o in others]].to_numpy())
        reg = f["regime"].to_numpy()
        virtual = zhat * prep.sd_.loc[reg, sensor].to_numpy() + prep.mu_.loc[reg, sensor].to_numpy()
        vmap = pd.Series(virtual, index=f["cycle"].astype(int).to_numpy())
        eng = raw[raw["unit"] == unit].copy()
        eng[sensor] = eng["cycle"].astype(int).map(vmap).to_numpy()
        return eng
