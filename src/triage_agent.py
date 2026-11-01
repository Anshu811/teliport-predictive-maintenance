"""Rule-based Triage Agent (no LLM).

Reads model outputs for each engine, decides severity with transparent threshold rules,
flags exceptions for a human, and writes maintenance work orders (JSON + CSV log).
v2: uses the 90% lower bound of the RUL prediction for the safety-critical rule.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
NOT_LOCALISED = "Not localised (sensors inside healthy envelope)"

SENSOR_COMPONENT = {
    "s2": "Low-pressure compressor (LPC)", "s3": "High-pressure compressor (HPC)",
    "s4": "Low-pressure turbine (LPT)", "s7": "High-pressure compressor (HPC)",
    "s8": "Fan", "s9": "Engine core (HPC/HPT spool)", "s11": "High-pressure compressor (HPC)",
    "s12": "High-pressure compressor (HPC)", "s13": "Fan", "s14": "Engine core (HPC/HPT spool)",
    "s15": "Fan / bypass duct", "s17": "Bleed air system",
    "s20": "High-pressure turbine (HPT) cooling", "s21": "Low-pressure turbine (LPT) cooling",
}


@dataclass
class TriageConfig:
    critical_rul: float = 20      # <= 20 cycles left  -> CRITICAL (business definition)
    critical_margin: float = 2.0  # extra safety margin for the CRITICAL trigger, chosen on GATE engines
    high_rul: float = 40          # <= 40              -> HIGH
    medium_rul: float = 75        # <= 75              -> MEDIUM, else LOW (monitor only)
    due_fraction: float = 0.5     # schedule work at half the decision RUL (2x safety factor)
    due_basis: str = "lower"      # "point": due from decision RUL, "lower": due from the 90% lower bound
    disagreement_rul: float = 100 # alarm while RUL > 100 -> models disagree -> human review
    min_cycles: int = 20          # shorter history -> low confidence -> human review
    min_severity_for_wo: str = "MEDIUM"
    conservative_mode: str = "critical" # "critical": 90% lower bound used for the CRITICAL rule only
                                        # "all": lower bound for every rule, "none": point estimate only


@dataclass
class Decision:
    subset: str
    unit: int
    cycles_seen: int
    pred_RUL: float
    rul_lower: float
    rul_upper: float
    decision_RUL: float
    anomaly_score: float
    alarm: bool
    severity: str
    suspected_component: str
    top_sensors: list
    needs_human_review: bool
    review_reasons: list = field(default_factory=list)
    rules_fired: list = field(default_factory=list)
    model_version: str = "unknown"
    sensor_suspect: str = ""
    rul_backup: float | None = None


ACTIVE_STATUSES = {"AUTO_APPROVED", "PENDING_HUMAN_REVIEW", "APPROVED_BY_HUMAN"}


def parse_top_sensors(raw):
    """Accept a list of (sensor, name, pct) tuples, a list of dicts, or the same as a JSON/CSV string."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            try:
                raw = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                raw = []
    out = []
    for item in raw if isinstance(raw, (list, tuple)) else []:
        if isinstance(item, dict):
            item = (item.get("sensor"), item.get("name", item.get("sensor")), item.get("share_%", 0))
        if isinstance(item, (list, tuple)) and len(item) == 3:
            sensor, name, pct = item
            try:
                pct = float(pct)
            except (TypeError, ValueError):
                continue
            if sensor in SENSOR_COMPONENT and 0 <= pct <= 100:
                out.append((sensor, str(name), pct))
    return out


class ActiveOrderRegistry:
    """One active work order per engine. Re-scoring the same engine UPDATES its order instead of
    creating a duplicate: severity can only go up and the due cycle can only move earlier."""

    def __init__(self, path):
        self.path = Path(path)
        self.orders = json.loads(self.path.read_text()) if self.path.exists() else {}

    def register(self, wo):
        cur = self.orders.get(wo["asset"])
        if cur is None or cur.get("status") not in ACTIVE_STATUSES:
            wo = dict(wo, history=[{"cycle": wo["cycles_seen"], "severity": wo["severity"]}])
            self.orders[wo["asset"]] = wo
            return wo, True
        if SEVERITY_ORDER[wo["severity"]] > SEVERITY_ORDER[cur["severity"]]:
            for k in ("severity", "recommended_action", "rules_fired", "suspected_component", "top_sensors"):
                cur[k] = wo[k]
        if wo["status"] == "PENDING_HUMAN_REVIEW" and cur["status"] == "AUTO_APPROVED":
            cur["status"], cur["review_reasons"] = "PENDING_HUMAN_REVIEW", wo["review_reasons"]
        cur["due_by_cycle"] = min(cur["due_by_cycle"], wo["due_by_cycle"])
        cur["due_within_cycles"] = max(0, cur["due_by_cycle"] - wo["cycles_seen"])
        for k in ("cycles_seen", "pred_RUL", "rul_lower", "rul_upper", "decision_RUL", "anomaly_score",
                  "alarm", "model_version"):
            cur[k] = wo[k]
        cur["last_updated"] = wo["created_at"]
        entry = {"cycle": wo["cycles_seen"], "severity": wo["severity"]}
        history = cur.setdefault("history", [])
        if not history or history[-1] != entry:
            history.append(entry)
        return cur, False

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.orders, indent=2))
        tmp.replace(self.path)


class TriageAgent:
    def __init__(self, config: TriageConfig | None = None, out_dir=None):
        self.cfg = config or TriageConfig()
        self.out_dir = Path(out_dir) if out_dir else None

    # ---------- 1. decide ----------
    def assess(self, snap: dict) -> Decision:
        c, alarm = self.cfg, bool(snap["alarm"])
        point = float(snap["pred_RUL"])
        lower = float(snap.get("rul_lower", point))
        upper = float(snap.get("rul_upper", point))
        rules, reasons = [], []
        crit_limit = c.critical_rul + c.critical_margin

        def band(r):
            return ("CRITICAL" if r <= crit_limit else "HIGH" if r <= c.high_rul
                    else "MEDIUM" if r <= c.medium_rul else "LOW")

        if c.conservative_mode == "all" or (c.conservative_mode == "critical" and lower <= crit_limit):
            rul, name = lower, "conservative RUL (90% lower bound)"
        else:
            rul, name = point, "pred_RUL"
        sev = band(rul)
        limit = {"CRITICAL": crit_limit, "HIGH": c.high_rul, "MEDIUM": c.medium_rul}.get(sev)
        rule_id = {"CRITICAL": "R1", "HIGH": "R2", "MEDIUM": "R3", "LOW": "R4"}[sev]
        rules.append(f"{rule_id} {name} {rul:.0f} <= {limit:.0f}" if limit is not None
                     else f"R4 {name} {rul:.0f} > {c.medium_rul:.0f} (healthy)")
        if name.startswith("conservative") and band(point) != sev:
            rules.append(f"U1 point estimate {point:.0f} alone would give {band(point)}; "
                         "uncertainty pushes it to " + sev)

        if alarm and sev == "LOW":
            sev = "MEDIUM"; rules.append("E1 anomaly alarm active -> escalate LOW to MEDIUM (inspect)")

        suspect = snap.get("sensor_suspect") or ""
        suspect = suspect if isinstance(suspect, str) else ""
        backup = snap.get("rul_backup")
        backup = None if backup is None or pd.isna(backup) else float(backup)
        if suspect:
            reasons.append(f"Sensor {suspect} disagrees with the other sensors: verify the sensor before acting")
            if backup is not None:
                b_low = float(snap.get("rul_backup_lower", backup))
                b_rul = b_low if (c.conservative_mode == "all" or (c.conservative_mode == "critical" and b_low <= crit_limit)) else backup
                alt = band(b_rul)
                if SEVERITY_ORDER[alt] > SEVERITY_ORDER[sev]:
                    rules.append(f"S1 RUL without suspect sensor {suspect} is {backup:.0f}: escalate {sev} to {alt}")
                    sev, rul = alt, min(rul, b_rul)

        if alarm and point > c.disagreement_rul:
            reasons.append("Health monitor alarm but RUL model sees > "
                           f"{c.disagreement_rul:.0f} cycles: possible sensor fault or unseen failure mode")
        if rul <= c.high_rul and not alarm:
            reasons.append("RUL model predicts near-failure but no anomaly alarm: verify sensors/data")
        if int(snap["cycles_seen"]) < c.min_cycles:
            reasons.append(f"Only {int(snap['cycles_seen'])} cycles of history: low-confidence prediction")
        rules += [f"X{i+1} human review: {r}" for i, r in enumerate(reasons)]

        top = parse_top_sensors(snap.get("top_sensors") or [])
        weights = {}
        for s, _, pct in top:
            comp = SENSOR_COMPONENT.get(s, "Unknown")
            weights[comp] = weights.get(comp, 0) + pct
        # root cause only makes sense when the engine is actually outside its healthy envelope
        if weights and float(snap["anomaly_score"]) > 1.0:
            component = max(weights, key=weights.get)
        else:
            component = NOT_LOCALISED

        return Decision(subset=str(snap["subset"]), unit=int(snap["unit"]),
                        cycles_seen=int(snap["cycles_seen"]), pred_RUL=round(point, 1),
                        rul_lower=round(lower, 1), rul_upper=round(upper, 1), decision_RUL=round(rul, 1),
                        anomaly_score=round(float(snap["anomaly_score"]), 2), alarm=alarm,
                        severity=sev, suspected_component=component,
                        top_sensors=[{"sensor": s, "name": n, "share_%": p} for s, n, p in top],
                        needs_human_review=bool(reasons), review_reasons=reasons, rules_fired=rules,
                        model_version=str(snap.get("model_version", "unknown")),
                        sensor_suspect=suspect, rul_backup=backup)

    # ---------- 2. act ----------
    def recommended_action(self, d: Decision, due: int) -> str:
        comp = (d.suspected_component if d.suspected_component != NOT_LOCALISED
                else "the engine (full inspection, no single component stands out)")
        if d.severity == "CRITICAL":
            return f"Remove from service at the next stop; inspect and repair/replace {comp} before further operation."
        if d.severity == "HIGH":
            return f"Schedule maintenance on {comp} within {due} cycles; reserve parts and crew now."
        return f"Plan borescope inspection of {comp} within {due} cycles; increase monitoring frequency."

    def create_work_order(self, d: Decision, now: datetime | None = None,
                          write: bool = True) -> dict | None:
        if SEVERITY_ORDER[d.severity] < SEVERITY_ORDER[self.cfg.min_severity_for_wo]:
            return None
        now = now or datetime.now(timezone.utc)
        basis = d.rul_lower if self.cfg.due_basis == "lower" else d.decision_RUL
        due = 1 if d.severity == "CRITICAL" else max(1, int(basis * self.cfg.due_fraction))
        wo = {
            "work_order_id": f"WO-{d.subset}-U{d.unit:03d}-C{d.cycles_seen:03d}",
            "created_at": now.isoformat(timespec="seconds"),
            "asset": f"{d.subset}/engine-{d.unit:03d}",
            "severity": d.severity,
            "status": "PENDING_HUMAN_REVIEW" if d.needs_human_review else "AUTO_APPROVED",
            "due_within_cycles": due,
            "due_by_cycle": d.cycles_seen + due,
            "recommended_action": self.recommended_action(d, due),
            **{k: v for k, v in asdict(d).items() if k not in ("subset", "unit")},
        }
        if self.out_dir and write:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / f"{wo['work_order_id']}.json").write_text(json.dumps(wo, indent=2))
        return wo

    # ---------- 3. run over a fleet snapshot ----------
    def _write(self, wo):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        target = self.out_dir / f"{wo['work_order_id']}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(wo, indent=2))
        tmp.replace(target)

    def run(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        registry = ActiveOrderRegistry(self.out_dir / "active_registry.json") if self.out_dir else None
        rows, new_orders = [], []
        created = updated = 0
        for snap in snapshots.to_dict("records"):
            d = self.assess(snap)
            wo = self.create_work_order(d, write=False)
            if wo is not None and registry is not None:
                wo, fresh = registry.register(wo)
                self._write(wo)
                created, updated = created + int(fresh), updated + int(not fresh)
                if fresh:
                    new_orders.append(wo)
            rows.append({"subset": d.subset, "unit": d.unit, "cycles_seen": d.cycles_seen,
                         "pred_RUL": d.pred_RUL, "rul_lower": d.rul_lower, "rul_upper": d.rul_upper,
                         "anomaly_score": d.anomaly_score, "alarm": d.alarm,
                         "severity": d.severity, "suspected_component": d.suspected_component,
                         "needs_human_review": d.needs_human_review, "sensor_suspect": d.sensor_suspect,
                         "work_order_id": wo["work_order_id"] if wo else None,
                         "wo_status": wo["status"] if wo else "NO_ACTION",
                         "due_within_cycles": wo["due_within_cycles"] if wo else None})
        self.last_run_summary = {"new_orders": created, "updated_existing_orders": updated}
        if registry is not None:
            registry.save()
            if new_orders:
                log = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                     for k, v in o.items()} for o in new_orders])
                log_path = self.out_dir / "work_order_log.csv"
                log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        return pd.DataFrame(rows)
