"""Teliport - end-to-end predictive-maintenance pipeline. One command, no manual steps.

  python run_pipeline.py                         # FULL: raw data -> features -> models -> MLflow registry
                                                 #       -> retraining demo -> agent -> work orders -> reports
  python run_pipeline.py --subsets FD001 FD003   # full run for selected subsets only
  python run_pipeline.py --mode score --subset FD001 --input data/incoming/new_readings.csv
                                                 # SCORE: new sensor readings -> deployed champion -> work orders
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
warnings.filterwarnings("ignore")

import pandas as pd                                                     # noqa: E402
import requests                                                         # noqa: E402

from cmapss_data import SUBSETS, load_subset, _read_txt                 # noqa: E402
from inference import score_fleet                                       # noqa: E402
from triage_agent import TriageAgent, TriageConfig, SEVERITY_ORDER      # noqa: E402

DATA_DIR = PROJECT / "data" / "raw" / "CMAPSS"
INCOMING = PROJECT / "data" / "incoming"
MODELS = PROJECT / "models"
REPORTS = PROJECT / "reports"
WO_DIR = PROJECT / "work_orders"
URLS = ["https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
        "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"]
EXPECTED = [f"{k}_FD00{i}.txt" for i in range(1, 5) for k in ("train", "test", "RUL")]

log = logging.getLogger("teliport")


def setup_logging():
    REPORTS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(REPORTS / "pipeline.log", mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)
    logging.getLogger("mlflow").setLevel(logging.ERROR)


# ---------------- stage 0: data ----------------
def ensure_dataset():
    if all((DATA_DIR / f).exists() for f in EXPECTED):
        log.info("Dataset present: %s", DATA_DIR)
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DATA_DIR.parent / "_tmp"
    for url in URLS:
        try:
            log.info("Downloading C-MAPSS from %s", url)
            zpath = DATA_DIR.parent / "cmapss.zip"
            with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
                r.raise_for_status()
                with open(zpath, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp)
            done = set()
            while True:
                inner = [p for p in tmp.rglob("*.zip") if p not in done]
                if not inner:
                    break
                for p in inner:
                    with zipfile.ZipFile(p) as z:
                        z.extractall(p.with_suffix(""))
                    done.add(p)
            for name in EXPECTED:
                hits = list(tmp.rglob(name))
                if hits:
                    shutil.copy(hits[0], DATA_DIR / name)
            zpath.unlink(missing_ok=True)
            if all((DATA_DIR / f).exists() for f in EXPECTED):
                log.info("Dataset downloaded")
                return
        except Exception as e:                                          # try the next mirror
            log.warning("Download failed (%s)", e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise SystemExit("Could not download the dataset - place the 12 C-MAPSS txt files in data/raw/CMAPSS")


# ---------------- stage 1: agent pre-flight ----------------
def agent_self_test():
    """The agent must pass its rule tests before it is allowed to create work orders."""
    agent = TriageAgent(TriageConfig(critical_margin=0.0))
    base = dict(subset="FD001", unit=1, cycles_seen=150, anomaly_score=0.5, alarm=False, top_sensors=[])
    cases = [(dict(pred_RUL=8, alarm=True), "CRITICAL", False), (dict(pred_RUL=35, alarm=True), "HIGH", False),
             (dict(pred_RUL=60, alarm=True), "MEDIUM", False), (dict(pred_RUL=118), "LOW", False),
             (dict(pred_RUL=90, alarm=True), "MEDIUM", False), (dict(pred_RUL=120, alarm=True), "MEDIUM", True),
             (dict(pred_RUL=15), "CRITICAL", True), (dict(pred_RUL=50, alarm=True, cycles_seen=12), "MEDIUM", True),
             (dict(pred_RUL=20, alarm=True), "CRITICAL", False), (dict(pred_RUL=75), "MEDIUM", False),
             (dict(pred_RUL=26, rul_lower=17, rul_upper=35, alarm=True), "CRITICAL", False),
             (dict(pred_RUL=26, rul_lower=22, rul_upper=35, alarm=True), "HIGH", False)]
    for over, sev, review in cases:
        d = agent.assess({**base, **over})
        if d.severity != sev or d.needs_human_review != review:
            raise SystemExit(f"Agent self-test FAILED for {over}: got {d.severity}/{d.needs_human_review}")
    log.info("Agent self-test: %d/%d rule tests passed", len(cases), len(cases))


# ---------------- stage 2-4: train, register, deploy, retrain ----------------
def reset_tracking():
    try:
        (PROJECT / "mlflow.db").unlink(missing_ok=True)
    except PermissionError:
        raise SystemExit("mlflow.db is locked - close the MLflow UI and any notebook that uses MLflow, then rerun.")
    shutil.rmtree(PROJECT / "mlruns", ignore_errors=True)
    for f in MODELS.glob("deployed_*.json"):
        f.unlink()
    log.info("MLflow tracking store reset (fresh, reproducible registry)")


def train_register_deploy(subsets, demo_retrain):
    from mlops import (deploy, retrain_on_new_data, set_champion, setup_mlflow, split_gate, train_and_log)
    client = setup_mlflow(PROJECT)
    rows, splits = [], {}
    for fd in subsets:
        t0 = time.time()
        train, test, rul = load_subset(fd, DATA_DIR)
        train = train.drop(columns=["subset"])
        pool = train[train["unit"] <= 80] if (fd == "FD001" and demo_retrain) else train
        tr_part, gate = split_gate(pool)
        splits[fd] = (tr_part, gate)
        res = train_and_log(fd, tr_part, gate, test, rul, MODELS, run_name=f"{fd}-initial",
                            tags={"trigger": "pipeline", "promotion_decision": "INITIAL_CHAMPION"})
        set_champion(client, fd, res["version"])
        deploy(res, fd, MODELS)
        rows.append({"subset": fd, "champion": f"v{res['version']}", "train_engines": tr_part["unit"].nunique(),
                     "gate_engines": gate["unit"].nunique(), "gate_RMSE": res["gate"]["RMSE"],
                     "gate_coverage_%": res["gate"]["interval_coverage_pct"],
                     "test_RMSE": res["test"]["RMSE"], "test_NASA_score": res["test"]["NASA_score"],
                     "test_RMSE_RUL<=30": res["test"]["RMSE_RUL_le_30"]})
        log.info("%s v%s trained, registered, deployed | gate RMSE %.2f | test RMSE %.2f | %.0fs",
                 fd, res["version"], res["gate"]["RMSE"], res["test"]["RMSE"], time.time() - t0)

    retrain_log = None
    if demo_retrain and "FD001" in subsets:
        train, test, rul = load_subset("FD001", DATA_DIR)
        batch = train[train["unit"] > 80].drop(columns=["RUL", "subset"]).copy()
        batch["unit"] -= 80
        INCOMING.mkdir(parents=True, exist_ok=True)
        batch.to_csv(INCOMING / "FD001_batch_001.csv", index=False)
        log.info("New data arrived: %s (%d engines)", INCOMING / "FD001_batch_001.csv", batch["unit"].nunique())
        tr_part, gate = splits["FD001"]
        retrain_log = retrain_on_new_data("FD001", tr_part, gate, pd.read_csv(INCOMING / "FD001_batch_001.csv"),
                                          test, rul, client, MODELS)
        log.info("Retraining decision: %s", retrain_log["decision"])
        if retrain_log["decision"].startswith("PROMOTED"):
            row = next(r for r in rows if r["subset"] == "FD001")
            row.update({"champion": f"v{retrain_log['challenger']['version']}",
                        "gate_RMSE": retrain_log["challenger"]["gate_RMSE"],
                        "test_RMSE": retrain_log["test_RMSE_report_only"]})
    return pd.DataFrame(rows).set_index("subset"), retrain_log


# ---------------- stage 5: score fleet + agent ----------------
def score_and_triage(raw_by_subset, fresh=True):
    snaps = pd.concat([score_fleet(raw, fd, MODELS) for fd, raw in raw_by_subset.items()], ignore_index=True)
    if fresh:
        shutil.rmtree(WO_DIR, ignore_errors=True)
    decisions = TriageAgent(TriageConfig(conservative_mode="critical"), out_dir=WO_DIR).run(snaps)
    decisions["model_version"] = snaps["model_version"].to_numpy()
    return decisions


def evaluate_agent(decisions, subsets):
    cfg = TriageConfig()
    truth = pd.concat([load_subset(fd, DATA_DIR)[2].assign(subset=fd) for fd in subsets])[["subset", "unit", "RUL"]]
    ev = decisions.merge(truth, on=["subset", "unit"])
    band = lambda r: ("CRITICAL" if r <= cfg.critical_rul else "HIGH" if r <= cfg.high_rul
                      else "MEDIUM" if r <= cfg.medium_rul else "LOW")
    s, t = ev["severity"].map(SEVERITY_ORDER), ev["RUL"].apply(band).map(SEVERITY_ORDER)
    return {"engines": len(ev),
            "truly_critical_engines": int((ev.RUL <= cfg.critical_rul).sum()),
            "critical_missed": int(((ev.RUL <= cfg.critical_rul) & (s < 3)).sum()),
            "exact_severity_match_%": round(100 * float((s == t).mean()), 1),
            "within_one_level_%": round(100 * float(((s - t).abs() <= 1).mean()), 1),
            "work_orders": int(ev["work_order_id"].notna().sum()),
            "work_orders_failing_before_due": int((ev.work_order_id.notna() & (ev.RUL < ev.due_within_cycles)).sum()),
            "human_review_rate_%": round(100 * float(ev["needs_human_review"].mean()), 1)}


# ---------------- modes ----------------
def run_full(subsets, demo_retrain):
    t0 = time.time()
    ensure_dataset()
    agent_self_test()
    reset_tracking()
    models, retrain_log = train_register_deploy(subsets, demo_retrain)
    decisions = score_and_triage({fd: load_subset(fd, DATA_DIR)[1] for fd in subsets})
    agent_eval = evaluate_agent(decisions, subsets)
    agent_eval["sensor_flags"] = int((decisions["sensor_suspect"].fillna("") != "").sum())

    models.to_csv(REPORTS / "pipeline_model_summary.csv")
    decisions.to_csv(REPORTS / "triage_decisions_test.csv", index=False)
    summary = {"finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "runtime_minutes": round((time.time() - t0) / 60, 1), "subsets": subsets,
               "models": models.reset_index().to_dict("records"),
               "retraining": {k: v for k, v in (retrain_log or {}).items() if k != "validation"},
               "agent": agent_eval,
               "severity_counts": decisions["severity"].value_counts().to_dict()}
    (REPORTS / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 78)
    print(" TELIPORT PIPELINE - SUMMARY")
    print("=" * 78)
    print(models.to_string())
    if retrain_log:
        print(f"\nRetraining : {retrain_log['decision']}")
    print("\nAgent      : " + " | ".join(f"{k}={v}" for k, v in agent_eval.items()))
    print(f"Outputs    : {WO_DIR} , {REPORTS / 'pipeline_summary.json'}")
    print(f"Runtime    : {summary['runtime_minutes']} min")
    print("=" * 78)


def run_score(fd, input_path):
    path = Path(input_path)
    raw = pd.read_csv(path) if path.suffix == ".csv" else _read_txt(path)
    log.info("Scoring %d engines from %s with deployed %s model", raw["unit"].nunique(), path, fd)
    decisions = score_and_triage({fd: raw}, fresh=False)
    out = REPORTS / f"score_{fd}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    decisions.to_csv(out, index=False)
    print(decisions[["unit", "severity", "pred_RUL", "rul_lower", "rul_upper", "alarm",
                     "wo_status", "work_order_id"]].to_string(index=False))
    log.info("Decisions saved to %s | work orders in %s", out, WO_DIR)


def main():
    ap = argparse.ArgumentParser(description="Teliport predictive-maintenance pipeline")
    ap.add_argument("--mode", choices=["full", "score"], default="full")
    ap.add_argument("--subsets", nargs="+", default=SUBSETS, choices=SUBSETS)
    ap.add_argument("--subset", choices=SUBSETS, help="score mode: which fleet's model to use")
    ap.add_argument("--input", help="score mode: sensor readings file (C-MAPSS txt or csv)")
    ap.add_argument("--no-demo-retrain", action="store_true", help="train FD001 on all engines, skip the demo")
    args = ap.parse_args()
    setup_logging()
    if args.mode == "full":
        run_full(list(args.subsets), demo_retrain=not args.no_demo_retrain)
    else:
        if not (args.subset and args.input):
            ap.error("--mode score needs --subset and --input")
        run_score(args.subset, args.input)


if __name__ == "__main__":
    main()
