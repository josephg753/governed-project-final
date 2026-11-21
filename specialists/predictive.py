"""
PREDICTIVE SPECIALIST: what will happen. The federated model as a tool, with its card facts attached.

Tools: model_card_facts, predict_record, rank_records_by_risk (template),
       member_model_comparison, high_risk_count, calibration_check, what_if (team).

Record-level tools (predict_record, rank_records_by_risk, what_if) are limited by policy to the home member.
The aggregate tools (high_risk_count, calibration_check, member_model_comparison) may be asked about any member in
scope: they return counts, rates and accuracies, never an id.
"""
import numpy as np

from agent import model as M
from agent import tools as T
from . import _data as D

MIN_CELL = 5   # counts from 1 to 4 are reported as "fewer than 5" (small-cell suppression)


def _m():
    return M.load_model()


def _probs(c, m):
    w = np.array(m["weights"])
    return 1 / (1 + np.exp(-(c.X_train @ w + m["bias"])))


def _latest_month(c):
    return max(int(r["month"]) for r in c.rows)


@T.tool("model_card_facts", "Provenance of the federated model: members, charter, gate decision, accuracy overall and per member, prohibited uses.",
        T.obj_schema({}), kind="read")
def model_card_facts():
    m = _m()
    out = {k: m[k] for k in ("model_name", "charter", "gate_decision", "neutral_test_accuracy", "per_member_accuracy", "intended_use", "prohibited_uses", "threshold_high_risk")}
    out["personalized_members"] = m.get("personalization", {}).get("adopted_by", [])
    return out


@T.tool("predict_record", "Probability of the outcome for ONE record of a member, with band and top drivers. Record-level: scope applies.",
        T.obj_schema({"member": ("string", "the member that holds the record"), "record_id": ("string", "the record's id, for example LG-00042")}), kind="model")
def predict_record(member, record_id):
    r = D.record(member, record_id)
    if r is None:
        return {"error": f"no record {record_id} at {member}"}
    m = _m()
    p = M.score(r, m)
    return {"member": member, "record_id": record_id, "month": int(r["month"]), "probability": round(p, 3), "band": M.band(p, m["threshold_high_risk"]),
            "threshold": m["threshold_high_risk"], "drivers": M.top_drivers(r, m), "model": m["model_name"],
            "model_neutral_test_accuracy": m["neutral_test_accuracy"],
            "model_accuracy_at_this_member": m["per_member_accuracy"].get(member)}


@T.tool("rank_records_by_risk", "The top_k highest-risk records of a member, optionally restricted to one month. Record-level: scope applies.",
        T.obj_schema({"member": ("string", "member name"), "top_k": ("integer", "how many records, default 5"), "month": ("integer", "1 to 12, optional")}, required=["member"]), kind="model")
def rank_records_by_risk(member, top_k=5, month=None):
    c = D.member(member)
    m = _m()
    probs = _probs(c, m)
    idx = [i for i in range(c.n) if month is None or int(c.rows[i]["month"]) == int(month)]
    idx.sort(key=lambda i: -probs[i])
    top = idx[:int(top_k)]
    return {"member": member, "month": month, "considered": len(idx),
            "records": [{"record_id": c.ids[i], "probability": round(float(probs[i]), 3), "band": M.band(float(probs[i]), m["threshold_high_risk"])} for i in top],
            "high_risk_share": round(float((probs[idx] >= m["threshold_high_risk"]).mean()), 3) if idx else None}


@T.tool("member_model_comparison",
        "For one member: accuracy on its neutral evaluation slice of its own local-only model, the federated model, the federated model "
        "fine-tuned locally under the charter's personalization clause, and the centralized reference. Says which the member should use. "
        "Reads the published model card; no rows are touched.",
        T.obj_schema({"member": ("string", "member name")}), kind="read")
def member_model_comparison(member):
    m = _m()
    b = m.get("member_baselines", {}).get(member)
    if b is None:
        return {"error": f"no baselines for {member} in the model file; rerun python run_federation.py"}
    pers = m.get("personalization", {})
    adopted = member in pers.get("adopted_by", [])
    options = {"local only": b["local_only"], "federated": b["fedavg"]}
    if "fedavg_finetuned" in b:
        options["federated + local fine-tune"] = b["fedavg_finetuned"]
    best = max(options, key=options.get)
    rows = m["training_rows"].get(member)
    if adopted:
        rec = (f"Use the federated model fine-tuned locally ({b['fedavg_finetuned']}): it beats the shared model by "
               f"{b['fedavg_finetuned'] - b['fedavg']:+.3f} and its own {rows}-row model by {b['fedavg_finetuned'] - b['local_only']:+.3f}. "
               "A fully separate model is not needed and would lose what the other members teach it.")
    elif b["fedavg"] >= b["local_only"]:
        rec = f"Use the federated model ({b['fedavg']}): it beats this member's own {rows}-row model ({b['local_only']}) by {b['fedavg'] - b['local_only']:+.3f}."
    else:
        rec = (f"The federated model ({b['fedavg']}) trails this member's own model ({b['local_only']}) by {b['local_only'] - b['fedavg']:.3f}, "
               "but fine-tuning does not clear the charter's adoption bar; keep the shared model and review at the next charter cycle.")
    return {"member": member, "training_rows": rows, "accuracy": options, "centralized_reference": b["centralized"],
            "best_option": best, "personalization_adopted": adopted,
            "charter_clause": {k: pers[k] for k in ("local_finetune_epochs", "adopt_when_finetune_gain_at_least") if k in pers},
            "recommendation": rec}


@T.tool("high_risk_count",
        "How many of a member's discharges score at or above the high-risk threshold, optionally in one month (default: the latest month "
        "in its data). Counts and shares only, no ids, so it may be asked about any member in aggregate scope. Counts of 1 to 4 are suppressed.",
        T.obj_schema({"member": ("string", "member name"), "month": ("integer", "1 to 12, optional; default the latest month")}, required=["member"]), kind="model")
def high_risk_count(member, month=None):
    c = D.member(member)
    m = _m()
    month = int(month) if month else _latest_month(c)
    probs = _probs(c, m)
    idx = np.array([int(r["month"]) == month for r in c.rows])
    n = int(idx.sum())
    hi = int((probs[idx] >= m["threshold_high_risk"]).sum())
    return {"member": member, "month": month, "discharges_in_month": n if n >= MIN_CELL else "fewer than 5",
            "high_risk": hi if hi == 0 or hi >= MIN_CELL else "fewer than 5",
            "high_risk_share": round(hi / n, 3) if n >= MIN_CELL else None,
            "threshold": m["threshold_high_risk"],
            "model_accuracy_at_this_member": m["per_member_accuracy"].get(member),
            "note": "Aggregate only. Naming the patients is record-level and allowed only at the home member."}


@T.tool("calibration_check",
        "Is the federated model calibrated for one member? Mean predicted probability vs the actual outcome rate on that member's slice of "
        "the neutral evaluation set, overall and in the high-risk band.",
        T.obj_schema({"member": ("string", "member name")}), kind="analyze")
def calibration_check(member):
    w = D.world()
    k = [c.name for c in w.clients].index(member)
    X, y = w.test_for(k)
    m = _m()
    p = 1 / (1 + np.exp(-(X @ np.array(m["weights"]) + m["bias"])))
    hi = p >= m["threshold_high_risk"]
    gap = float(p.mean() - y.mean())
    return {"member": member, "eval_rows": int(len(y)), "mean_predicted": round(float(p.mean()), 3), "actual_rate": round(float(y.mean()), 3),
            "calibration_gap": round(gap, 3),
            "high_band": {"rows": int(hi.sum()), "mean_predicted": round(float(p[hi].mean()), 3) if hi.any() else None,
                          "actual_rate": round(float(y[hi].mean()), 3) if hi.any() else None},
            "reading": ("well calibrated" if abs(gap) < 0.03 else
                        ("over-predicts" if gap > 0 else "under-predicts") + f" by {abs(gap):.3f}: treat probabilities at this member as "
                        + ("high" if gap > 0 else "low") + " and review the threshold before using them for staffing.")}


@T.tool("what_if",
        "Re-score ONE record with one feature changed (for example followup_scheduled from 0 to 1) to show how much that lever moves the "
        "risk. Record-level: scope applies. Does not change the record.",
        {"type": "object",
         "properties": {"member": {"type": "string", "description": "member that holds the record"},
                        "record_id": {"type": "string", "description": "record id"},
                        "feature": {"type": "string", "description": "feature to change"},
                        "value": {"type": "number", "description": "new value in human units"}},
         "required": ["member", "record_id", "feature", "value"]}, kind="model")
def what_if(member, record_id, feature, value):
    r = D.record(member, record_id)
    if r is None:
        return {"error": f"no record {record_id} at {member}"}
    if feature not in D.features():
        return {"error": f"{feature} is not a feature"}
    m = _m()
    before = M.score(r, m)
    r2 = dict(r)
    r2[feature] = value
    after = M.score(r2, m)
    return {"member": member, "record_id": record_id, "feature": feature, "from": r[feature], "to": value,
            "probability_before": round(before, 3), "probability_after": round(after, 3), "change": round(after - before, 3),
            "band_before": M.band(before, m["threshold_high_risk"]), "band_after": M.band(after, m["threshold_high_risk"]),
            "note": "Association in the model, not a causal effect."}
