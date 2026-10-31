"""Fleet scoring: raw sensor history -> one snapshot per engine (RUL range + health + root cause)."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from features import last_cycle
from modeling import predict


def load_models(fd, models_dir):
    models_dir = Path(models_dir)
    unc_path = models_dir / f"uncertainty_{fd}.joblib"
    return (joblib.load(models_dir / f"preprocessor_{fd}.joblib"),
            joblib.load(models_dir / f"lgbm_{fd}.joblib"),
            joblib.load(models_dir / f"health_monitor_{fd}.joblib"),
            joblib.load(unc_path) if unc_path.exists() else None)


def deployed_version(fd, models_dir):
    """Registry version written by MLOps deploy(); 'local' if the models were trained in a notebook."""
    path = Path(models_dir) / f"deployed_{fd}.json"
    return f"registry v{json.loads(path.read_text())['version']}" if path.exists() else "local"


def score_fleet(raw_df, fd, models_dir, model_version=None):
    """raw_df = sensor history (unit, cycle, settings, sensors) for any number of engines."""
    prep, rul_model, hm, unc = load_models(fd, models_dir)
    model_version = model_version or deployed_version(fd, models_dir)
    feats = prep.transform(raw_df)
    flags = hm.transform(feats)
    last = last_cycle(feats)
    last_flags = flags.groupby("unit").tail(1).set_index("unit")
    snap = pd.DataFrame({
        "subset": fd,
        "unit": last["unit"].to_numpy(),
        "cycles_seen": last["cycle"].astype(int).to_numpy(),
        "pred_RUL": np.round(predict(rul_model, last, prep.feature_names_), 1),
    })
    if unc is not None:                                   # 90% prediction interval
        lower, upper = unc.interval(snap["pred_RUL"].to_numpy())
        snap["rul_lower"], snap["rul_upper"] = np.round(lower, 1), np.round(upper, 1)
    else:
        snap["rul_lower"] = snap["rul_upper"] = snap["pred_RUL"]
    snap["anomaly_score"] = np.round(last_flags.loc[snap["unit"], "anomaly_score"].to_numpy(), 2)
    snap["alarm"] = last_flags.loc[snap["unit"], "alarm"].to_numpy().astype(bool)
    snap["top_sensors"] = [hm.top_contributors(row) for _, row in last.iterrows()]
    snap["model_version"] = model_version
    return snap
