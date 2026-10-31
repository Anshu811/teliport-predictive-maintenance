"""MLOps: MLflow experiment tracking, model registry, drift check and gated retraining flow.

Promotion decisions use a fixed GATE set of training engines that no model ever trains on.
The NASA test set is logged for reporting only and never drives a decision.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from anomaly import detection_report
from cmapss_data import COLUMNS, OP_COLS, SENSOR_COLS
from features import last_cycle
from modeling import evaluate, val_rows
from serving import RULPipelineModel
from training import save_pipeline, train_pipeline

EXPERIMENT = "teliport-predictive-maintenance"
GATE_FRACTION = 0.15
SRC_FILES = ["cmapss_data.py", "features.py", "modeling.py", "anomaly.py", "serving.py"]


def clean_key(k):
    """MLflow metric names cannot contain '<=' or '%'."""
    return k.replace("<=", "_le_").replace("%", "pct")


def model_name(fd):
    return f"teliport_rul_{fd}"


def setup_mlflow(project):
    project = Path(project)
    mlflow.set_tracking_uri(f"sqlite:///{(project / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(project / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


def split_gate(train_raw, fraction=GATE_FRACTION, seed=42):
    """Fixed, reproducible set of engines kept out of ALL training -> fair champion/challenger gate."""
    units = np.array(sorted(train_raw["unit"].unique()))
    gate = np.random.default_rng(seed).choice(units, size=max(1, round(fraction * len(units))), replace=False)
    is_gate = train_raw["unit"].isin(gate)
    return train_raw[~is_gate].copy(), train_raw[is_gate].copy()


# ---------------- data checks ----------------
def validate_batch(df):
    """Schema + sanity checks before new data is allowed into training."""
    issues = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
    else:
        if df[SENSOR_COLS + OP_COLS].isna().mean().max() > 0.2:
            issues.append("more than 20% missing values in a column")
        if df.duplicated(["unit", "cycle"]).any():
            issues.append(f"{int(df.duplicated(['unit', 'cycle']).sum())} duplicate (unit, cycle) rows (will be de-duplicated)")
        if (df["cycle"] < 1).any():
            issues.append("cycle numbers < 1")
    blocking = [i for i in issues if "duplicate" not in i]
    return {"ok": not blocking, "issues": issues, "rows": len(df), "engines": int(df["unit"].nunique())}


def psi(ref, new, bins=10):
    """Population Stability Index: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    n = np.histogram(new, edges)[0] / len(new) + 1e-6
    return float(np.sum((n - r) * np.log(n / r)))


def drift_report(prep, ref_raw, new_raw):
    ref = prep.transform(ref_raw)
    new = prep.transform(new_raw)
    rep = pd.Series({s: psi(ref[f"{s}_z"], new[f"{s}_z"]) for s in prep.sensors}).round(3)
    return rep.sort_values(ascending=False)


# ---------------- evaluation ----------------
def evaluate_on_gate(p, gate_raw):
    feats = val_rows(p["prep"].transform(gate_raw))
    pred = np.clip(p["model"].predict(feats[p["features"]]), 0, None)
    lower, upper = p["unc"].interval(pred)
    y = feats["RUL"].to_numpy()
    m = {clean_key(k): v for k, v in evaluate(y, pred).items() if k != "n"}
    m["interval_coverage_pct"] = round(100 * float(np.mean((y >= lower) & (y <= upper))), 1)
    return m


def evaluate_on_test(p, test_raw, test_rul):
    te = last_cycle(p["prep"].transform(test_raw)).merge(test_rul[["unit", "RUL"]], on="unit")
    pred = np.clip(p["model"].predict(te[p["features"]]), 0, None)
    return {clean_key(k): v for k, v in evaluate(te["RUL"], pred).items() if k != "n"}


def _offline_predict(p, raw):
    last = last_cycle(p["prep"].transform(raw))
    pred = np.clip(p["model"].predict(last[p["features"]]), 0, None)
    lower, upper = p["unc"].interval(pred)
    return pd.DataFrame({"unit": last["unit"].to_numpy(), "pred_RUL": np.round(pred, 1),
                         "rul_lower": np.round(lower, 1), "rul_upper": np.round(upper, 1)})


# ---------------- train + log + register ----------------
def train_and_log(fd, train_raw, gate_raw, test_raw, test_rul, models_dir, run_name, tags=None, extra_params=None):
    p = train_pipeline(fd, train_raw, models_dir=models_dir)
    gate_m = evaluate_on_gate(p, gate_raw)
    test_m = evaluate_on_test(p, test_raw, test_rul)
    det = detection_report(p["train_features"], p["hm"].transform(p["train_features"]))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"subset": fd, **(tags or {})})
        mlflow.log_params({"subset": fd, "train_engines": int(train_raw["unit"].nunique()),
                           "gate_engines": int(gate_raw["unit"].nunique()), "train_rows": len(train_raw),
                           "n_features": len(p["features"]), "n_seeds": len(p["model"].seeds),
                           **{f"lgb_{k}": v for k, v in p["params"].items()},
                           **{f"monitor_{k}": v for k, v in p["monitor_cfg"].items()}, **(extra_params or {})})
        mlflow.log_metrics({f"gate_{k}": float(v) for k, v in gate_m.items()})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_m.items()})
        mlflow.log_metric("anomaly_false_alarm_pct", float(det["false_alarm_rate_%"]))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = {"preprocessor": tmp / "preprocessor.joblib", "rul_model": tmp / "rul_model.joblib",
                     "uncertainty": tmp / "uncertainty.joblib", "health_monitor": tmp / "health_monitor.joblib"}
            for key, obj in [("preprocessor", p["prep"]), ("rul_model", p["model"]),
                             ("uncertainty", p["unc"]), ("health_monitor", p["hm"])]:
                joblib.dump(obj, paths[key])
            src = Path(__file__).resolve().parent
            example = test_raw[test_raw["unit"].isin(test_raw["unit"].unique()[:3])]
            info = mlflow.pyfunc.log_model(
                name="rul_pipeline", python_model=RULPipelineModel(),
                artifacts={k: str(v) for k, v in paths.items()},
                code_paths=[str(src / f) for f in SRC_FILES],
                signature=infer_signature(example, _offline_predict(p, example)),
                registered_model_name=model_name(fd))
        version = info.registered_model_version
    return {"run_id": run.info.run_id, "version": str(version), "gate": gate_m, "test": test_m, "pipeline": p}


def set_champion(client, fd, version):
    client.set_registered_model_alias(model_name(fd), "champion", str(version))


def get_champion(client, fd):
    try:
        mv = client.get_model_version_by_alias(model_name(fd), "champion")
    except Exception:
        return None
    return {"version": str(mv.version), "run_id": mv.run_id, "metrics": client.get_run(mv.run_id).data.metrics}


def deploy(result, fd, models_dir):
    """Copy the champion pipeline to the folder that inference / dashboard / agent read from."""
    save_pipeline(result["pipeline"], fd, models_dir)
    (Path(models_dir) / f"deployed_{fd}.json").write_text(json.dumps(
        {"subset": fd, "version": result["version"], "run_id": result["run_id"],
         "deployed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))


# ---------------- retraining flow ----------------
def retrain_on_new_data(fd, current_train_raw, gate_raw, new_batch_raw, test_raw, test_rul, client,
                        models_dir, max_rmse_increase=0.0, max_critical_rmse_increase=0.5):
    """New run-to-failure data arrived -> validate -> drift check -> retrain challenger ->
    compare with champion on the GATE engines -> promote + deploy only if not worse."""
    log = {"subset": fd, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    v = validate_batch(new_batch_raw)
    log["validation"] = v
    if not v["ok"]:
        log["decision"] = "REJECTED_BAD_DATA"
        return log

    new = new_batch_raw.copy()
    new["unit"] = new["unit"] + int(max(current_train_raw["unit"].max(), gate_raw["unit"].max()))
    new["RUL"] = new.groupby("unit")["cycle"].transform("max") - new["cycle"]
    combined = pd.concat([current_train_raw, new], ignore_index=True)

    champ = get_champion(client, fd)
    drift = drift_report(joblib.load(Path(models_dir) / f"preprocessor_{fd}.joblib"), current_train_raw, new)
    log["drift_max_psi"] = float(drift.max())
    log["drift_top"] = drift.head(3).to_dict()

    res = train_and_log(fd, combined, gate_raw, test_raw, test_rul, models_dir,
                        run_name=f"{fd}-retrain-{datetime.now():%Y%m%d-%H%M%S}",
                        tags={"trigger": "new_data", "stage": "challenger"},
                        extra_params={"new_engines": v["engines"], "drift_max_psi": round(float(drift.max()), 3)})
    ch = res["gate"]
    log["challenger"] = {"version": res["version"], "gate_RMSE": ch["RMSE"], "gate_RMSE_RUL_le_30": ch["RMSE_RUL_le_30"]}

    if champ is None:
        better, reason = True, "no champion yet"
    else:
        c_rmse, c_crit = champ["metrics"]["gate_RMSE"], champ["metrics"]["gate_RMSE_RUL_le_30"]
        log["champion"] = {"version": champ["version"], "gate_RMSE": c_rmse, "gate_RMSE_RUL_le_30": c_crit}
        better = (ch["RMSE"] <= c_rmse + max_rmse_increase) and \
                 (ch["RMSE_RUL_le_30"] <= c_crit + max_critical_rmse_increase)
        reason = (f"gate RMSE {ch['RMSE']} vs champion {c_rmse}; "
                  f"gate critical-zone RMSE {ch['RMSE_RUL_le_30']} vs {c_crit}")

    with mlflow.start_run(run_id=res["run_id"]):
        mlflow.set_tag("promotion_decision", "PROMOTED" if better else "KEPT_CHAMPION")
        mlflow.log_dict({k: val for k, val in log.items() if k != "validation"}, "retraining_log.json")

    if better:
        set_champion(client, fd, res["version"])
        deploy(res, fd, models_dir)
        log["decision"] = f"PROMOTED v{res['version']} to champion and deployed ({reason})"
    else:
        log["decision"] = f"KEPT champion v{champ['version']} ({reason})"
    log["test_RMSE_report_only"] = res["test"]["RMSE"]
    return log
