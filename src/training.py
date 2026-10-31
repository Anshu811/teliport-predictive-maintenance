"""ONE training recipe used everywhere: notebooks, MLOps retraining and the end-to-end pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from anomaly import HealthMonitor
from features import CMAPSSPreprocessor, DEFAULT_RUL_CAP
from modeling import RULUncertainty, SeedBag, cross_validate, fit_model, make_lgbm

DEFAULT_MONITOR = {"method": "mahalanobis", "quantile": 0.99, "persistence": 5}


def load_config(models_dir, filename, fd, default):
    path = Path(models_dir) / filename if models_dir else None
    if path and path.exists():
        return json.loads(path.read_text()).get(fd, default)
    return default


def train_pipeline(fd, train_raw, models_dir=None, params=None, monitor_cfg=None,
                   cap=DEFAULT_RUL_CAP, seeds=(0, 1, 2, 3, 4), calib_folds=3):
    """train_raw: run-to-failure history with an RUL column. Returns every fitted component."""
    params = params if params is not None else load_config(models_dir, "best_params.json", fd, {})
    monitor_cfg = monitor_cfg or load_config(models_dir, "monitor_config.json", fd, DEFAULT_MONITOR)

    prep = CMAPSSPreprocessor.for_subset(fd)
    tr = prep.fit_transform(train_raw)
    F = prep.feature_names_

    _, oof = cross_validate(tr, F, lambda: make_lgbm(**params), cap, n_splits=calib_folds)
    unc = RULUncertainty().fit(oof["y_true"], oof["y_pred"])          # calibrated on unseen engines
    model = fit_model(tr, F, lambda: SeedBag(params, seeds), cap, weighted=True)
    hm = HealthMonitor(**monitor_cfg).fit(tr, prep.sensors)
    return {"prep": prep, "model": model, "unc": unc, "hm": hm, "features": F,
            "params": params, "monitor_cfg": monitor_cfg, "train_features": tr}


def save_pipeline(p, fd, models_dir):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(p["prep"], models_dir / f"preprocessor_{fd}.joblib")
    joblib.dump(p["model"], models_dir / f"lgbm_{fd}.joblib")
    joblib.dump(p["unc"], models_dir / f"uncertainty_{fd}.joblib")
    joblib.dump(p["hm"], models_dir / f"health_monitor_{fd}.joblib")
    stale = models_dir / f"deployed_{fd}.json"          # MLOps deploy() rewrites it right after
    if stale.exists():
        stale.unlink()
