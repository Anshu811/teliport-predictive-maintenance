"""Teliport - Predictive Maintenance monitoring dashboard (Streamlit).

Run from the project root:   streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from cmapss_data import SUBSETS, load_subset, _read_txt          # noqa: E402
from inference import load_models, score_fleet                 # noqa: E402
from triage_agent import TriageAgent, TriageConfig, SEVERITY_ORDER  # noqa: E402

DATA_DIR = PROJECT / "data" / "raw" / "CMAPSS"
MODELS = PROJECT / "models"
REPORTS = PROJECT / "reports"
WO_DIR = PROJECT / "work_orders"
SEV_COLORS = {"CRITICAL": "#d62728", "HIGH": "#ff7f0e", "MEDIUM": "#f2c80f", "LOW": "#2ca02c"}
LOG_TICKS = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]

st.set_page_config(page_title="Teliport - Fleet Health", page_icon="🛠️", layout="wide")


# ---------------- data loading (cached) ----------------
@st.cache_data(show_spinner=False)
def load_test_fleet(fd):
    _, test, rul = load_subset(fd, DATA_DIR)
    return test, rul


@st.cache_data(show_spinner="Scoring fleet with deployed models...")
def score(fd, raw_df, stamp):
    snaps = score_fleet(raw_df, fd, MODELS)
    decisions = TriageAgent(TriageConfig()).run(snaps)
    out = decisions.merge(snaps[["unit", "top_sensors", "model_version"]], on="unit")
    out["sev_rank"] = out["severity"].map(SEVERITY_ORDER)
    return out.sort_values(["sev_rank", "pred_RUL"], ascending=[False, True]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def engine_history(fd, raw_df, unit, stamp):
    prep, model, hm, unc = load_models(fd, MODELS)
    eng = raw_df[raw_df["unit"] == unit]
    feats = prep.transform(eng)
    pred = np.clip(model.predict(feats[prep.feature_names_]), 0, None)
    lower, upper = unc.interval(pred) if unc is not None else (pred, pred)
    flags = hm.transform(feats)
    hist = pd.DataFrame({"cycle": feats["cycle"].to_numpy(), "pred_RUL": pred, "rul_lower": lower,
                         "rul_upper": upper, "anomaly_score": flags["anomaly_score"].to_numpy(),
                         "alarm": flags["alarm"].to_numpy()})
    sensors = feats[["cycle"] + [f"{s}_ewm" for s in prep.sensors]].rename(
        columns={f"{s}_ewm": s for s in prep.sensors})
    return hist, sensors


def models_stamp():
    """Changes whenever MLOps deploys new models -> refreshes the cache automatically."""
    return tuple(sorted((p.name, p.stat().st_mtime) for p in MODELS.glob("*.joblib")))


def read_mlflow_history():
    try:
        import mlflow
        from mlops import EXPERIMENT, model_name
        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT / 'mlflow.db').as_posix()}")
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["attributes.start_time ASC"])
        client = mlflow.tracking.MlflowClient()
        champs = {}
        for fd in SUBSETS:
            try:
                champs[fd] = f"v{client.get_model_version_by_alias(model_name(fd), 'champion').version}"
            except Exception:
                champs[fd] = "-"
        return runs, champs
    except Exception as e:                       # MLflow not initialised yet
        return None, {"error": str(e)}


# ---------------- sidebar ----------------
st.sidebar.title("🛠️ Teliport")
st.sidebar.caption("Self-healing machines: predictive maintenance for a connected fleet")
fd = st.sidebar.selectbox("Fleet (C-MAPSS subset)", SUBSETS, index=0)
source = st.sidebar.radio("Sensor data source", ["NASA test fleet", "Upload new readings (txt/csv)"])

raw, truth = load_test_fleet(fd)
if source.startswith("Upload"):
    up = st.sidebar.file_uploader("C-MAPSS format: unit, cycle, 3 settings, 21 sensors", type=["txt", "csv"])
    if up is not None:
        if up.name.endswith(".csv"):
            raw = pd.read_csv(up)
        else:
            tmp = PROJECT / "data" / "incoming" / "_upload.txt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(up.getvalue())
            raw = _read_txt(tmp)
        truth = None
        st.sidebar.success(f"Loaded {raw['unit'].nunique()} engines, {len(raw)} rows")
show_truth = truth is not None and st.sidebar.checkbox("Show true RUL (evaluation only)", value=False)

stamp = models_stamp()
fleet = score(fd, raw, stamp)
deployed = MODELS / f"deployed_{fd}.json"
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Deployed model:** {fleet['model_version'].iloc[0]}")
if deployed.exists():
    st.sidebar.caption(f"deployed at {json.loads(deployed.read_text())['deployed_at']}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Fleet overview", "🔍 Engine detail", "🧾 Work orders", "⚙️ Model monitoring"])

# ---------------- tab 1: fleet overview ----------------
with tab1:
    counts = fleet["severity"].value_counts()
    c = st.columns(6)
    c[0].metric("Engines", len(fleet))
    c[1].metric("🔴 Critical", int(counts.get("CRITICAL", 0)))
    c[2].metric("🟠 High", int(counts.get("HIGH", 0)))
    c[3].metric("🟡 Medium", int(counts.get("MEDIUM", 0)))
    c[4].metric("🚨 Anomaly alarms", int(fleet["alarm"].sum()))
    c[5].metric("👤 Needs human review", int(fleet["needs_human_review"].sum()))

    left, right = st.columns([1, 2])
    sev = (fleet["severity"].value_counts().reindex(list(SEV_COLORS), fill_value=0)
           .rename_axis("severity").reset_index(name="engines"))
    left.plotly_chart(px.bar(sev, x="severity", y="engines", color="severity", color_discrete_map=SEV_COLORS,
                             title="Engines by severity").update_layout(showlegend=False), width="stretch")
    fig = px.scatter(fleet, x="pred_RUL", y="anomaly_score", color="severity", color_discrete_map=SEV_COLORS,
                     hover_data=["unit", "rul_lower", "rul_upper", "suspected_component"], log_y=True,
                     title="Risk map: predicted RUL vs anomaly score")
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="healthy threshold", annotation_position="bottom left")
    fig.update_yaxes(tickvals=LOG_TICKS)
    fig.update_xaxes(autorange="reversed")
    right.plotly_chart(fig, width="stretch")

    view = fleet[["unit", "cycles_seen", "severity", "pred_RUL", "rul_lower", "rul_upper", "anomaly_score",
                  "alarm", "suspected_component", "sensor_suspect", "needs_human_review", "wo_status",
                  "due_within_cycles"]]
    if show_truth:
        view = view.merge(truth[["unit", "RUL"]].rename(columns={"RUL": "true_RUL"}), on="unit")
    st.subheader("Engines ranked by risk")
    st.dataframe(view, width="stretch", hide_index=True, height=420)

# ---------------- tab 2: engine detail ----------------
with tab2:
    unit = st.selectbox("Engine", fleet["unit"].tolist(),
                        format_func=lambda u: f"engine {u}  ({fleet.loc[fleet.unit == u, 'severity'].iloc[0]})")
    row = fleet[fleet["unit"] == unit].iloc[0]
    hist, sensors = engine_history(fd, raw, unit, stamp)

    m = st.columns(5)
    m[0].metric("Severity", row["severity"])
    m[1].metric("Predicted RUL", f"{row['pred_RUL']:.0f} cycles")
    m[2].metric("90% range", f"{row['rul_lower']:.0f} – {row['rul_upper']:.0f}")
    m[3].metric("Anomaly score", f"{row['anomaly_score']:.2f}", "ALARM" if row["alarm"] else "normal",
                delta_color="inverse" if row["alarm"] else "off")
    m[4].metric("Suspected component", row["suspected_component"].split(" (")[0])
    if isinstance(row.get("sensor_suspect"), str) and row["sensor_suspect"]:
        st.warning(f"Sensor {row['sensor_suspect']} disagrees with the other sensors: verify the sensor "
                   "before acting. The engine was sent to human review.")

    a, b = st.columns(2)
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_upper"], line=dict(width=0), showlegend=False))
    f1.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_lower"], fill="tonexty", line=dict(width=0),
                            fillcolor="rgba(31,119,180,0.2)", name="90% interval"))
    f1.add_trace(go.Scatter(x=hist["cycle"], y=hist["pred_RUL"], name="predicted RUL", line=dict(color="#1f77b4")))
    if show_truth:
        true_now = int(truth.loc[truth.unit == unit, "RUL"].iloc[0])
        f1.add_trace(go.Scatter(x=hist["cycle"], y=true_now + hist["cycle"].max() - hist["cycle"],
                                name="true RUL", line=dict(color="black", dash="dot")))
    for name, y in [("CRITICAL", 20), ("HIGH", 40), ("MEDIUM", 75)]:
        f1.add_hline(y=y, line_dash="dot", line_color=SEV_COLORS[name], annotation_text=name)
    f1.update_layout(title="RUL prediction over the engine's life", xaxis_title="cycle", yaxis_title="RUL (cycles)")
    a.plotly_chart(f1, width="stretch")

    f2 = px.line(hist, x="cycle", y="anomaly_score", log_y=True, title="Anomaly score (health monitor)")
    f2.add_hline(y=1.0, line_dash="dash", annotation_text="threshold", annotation_position="bottom left")
    f2.update_yaxes(tickvals=LOG_TICKS)
    alarm_pts = hist[hist["alarm"]]
    f2.add_trace(go.Scatter(x=alarm_pts["cycle"], y=alarm_pts["anomaly_score"], mode="markers",
                            marker=dict(color="red", size=5), name="alarm"))
    b.plotly_chart(f2, width="stretch")

    top = pd.DataFrame(row["top_sensors"], columns=["sensor", "description", "share of deviation %"])
    c1, c2 = st.columns([1, 2])
    c1.markdown("**Root cause: top contributing sensors**")
    c1.dataframe(top, hide_index=True, width="stretch")
    pick = c2.multiselect("Sensor trends (smoothed, regime-normalized)", sensors.columns[1:].tolist(),
                          default=top["sensor"].tolist())
    if pick:
        c2.plotly_chart(px.line(sensors, x="cycle", y=pick), width="stretch")

# ---------------- tab 3: work orders ----------------
with tab3:
    wo_files = sorted(WO_DIR.glob(f"WO-{fd}-*.json"))
    if not wo_files:
        st.info("No work orders on disk for this fleet yet. Run the agent / pipeline to generate them.")
    else:
        orders = [json.loads(p.read_text()) for p in wo_files]
        wo = pd.DataFrame([{k: o.get(k) for k in ["work_order_id", "severity", "status", "pred_RUL", "rul_lower",
                                                  "due_within_cycles", "suspected_component", "created_at"]}
                           for o in orders])
        wo["rank"] = wo["severity"].map(SEVERITY_ORDER)
        wo = wo.sort_values(["rank", "pred_RUL"], ascending=[False, True]).drop(columns="rank")
        status_filter = st.multiselect("Status", sorted(wo["status"].unique()), default=sorted(wo["status"].unique()))
        st.dataframe(wo[wo["status"].isin(status_filter)], hide_index=True, width="stretch", height=300)

        st.subheader("👤 Human review queue (exceptions only)")
        pending = [o for o in orders if o["status"] == "PENDING_HUMAN_REVIEW"]
        if not pending:
            st.success("Nothing waiting for review.")
        for o in pending:
            with st.expander(f"{o['work_order_id']}  |  {o['severity']}  |  pred RUL {o['pred_RUL']}"):
                st.write("**Why a human is needed:**", "; ".join(o["review_reasons"]))
                st.write("**Recommended action:**", o["recommended_action"])
                k1, k2 = st.columns(2)
                for col, label, status in [(k1, "✅ Approve", "APPROVED_BY_HUMAN"), (k2, "❌ Reject", "REJECTED_BY_HUMAN")]:
                    if col.button(label, key=f"{status}-{o['work_order_id']}"):
                        o["status"] = status
                        (WO_DIR / f"{o['work_order_id']}.json").write_text(json.dumps(o, indent=2))
                        st.rerun()

        chosen = st.selectbox("Open a work order", wo["work_order_id"].tolist())
        st.json(json.loads((WO_DIR / f"{chosen}.json").read_text()), expanded=False)

# ---------------- tab 4: model monitoring ----------------
with tab4:
    runs, champs = read_mlflow_history()
    if runs is None or runs.empty:
        st.warning("No MLflow runs found yet - run the MLOps notebook first.")
    else:
        st.subheader("Model registry: current champions")
        st.dataframe(pd.DataFrame([champs]), hide_index=True)
        cols = {"tags.mlflow.runName": "run", "params.subset": "subset", "params.train_engines": "train_engines",
                "metrics.gate_RMSE": "gate_RMSE", "metrics.gate_RMSE_RUL_le_30": "gate_RMSE_RUL<=30",
                "metrics.gate_interval_coverage_pct": "gate_coverage_%", "metrics.test_RMSE": "test_RMSE",
                "params.drift_max_psi": "drift_max_psi", "tags.promotion_decision": "decision",
                "start_time": "started"}
        hist_runs = runs[[k for k in cols if k in runs.columns]].rename(columns=cols)
        st.subheader("Experiment & retraining history (MLflow)")
        st.dataframe(hist_runs, hide_index=True, width="stretch")
        sub = hist_runs[hist_runs["subset"] == fd]
        if len(sub) and "gate_RMSE" in sub:
            st.plotly_chart(px.line(sub, x="run", y=["gate_RMSE", "test_RMSE"], markers=True,
                                    title=f"{fd}: model versions over time (lower is better)"),
                            width="stretch")

    st.subheader("Live prediction monitoring (current data)")
    k = st.columns(4)
    k[0].metric("Mean predicted RUL", f"{fleet['pred_RUL'].mean():.1f}")
    k[1].metric("Alarm rate", f"{100 * fleet['alarm'].mean():.1f}%")
    k[2].metric("Mean interval width", f"{(fleet['rul_upper'] - fleet['rul_lower']).mean():.1f} cycles")
    k[3].metric("Missing sensor values", f"{100 * raw.drop(columns=['unit', 'cycle']).isna().mean().mean():.2f}%")
    st.plotly_chart(px.histogram(fleet, x="pred_RUL", nbins=30, color="severity", color_discrete_map=SEV_COLORS,
                                 title="Distribution of predicted RUL"), width="stretch")

    for name, title in [("rul_test_results_v2.csv", "Offline RUL evaluation (NASA test set)"),
                        ("agent_evaluation_v2.csv", "Agent evaluation vs ground truth")]:
        path = REPORTS / name
        if path.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")
