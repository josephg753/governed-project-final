"""
DIAGNOSTIC SPECIALIST: why. Compares members and ranks drivers, still with aggregates only.

Tools: compare_members, rank_drivers (template),
       stratified_comparison, feature_effect_by_member, explain_change (team).
"""
import numpy as np

from agent import tools as T
from . import _data as D

AGE_BANDS = ([65, 80], ["<65", "65-79", "80+"])
COMORB_BANDS = ([2, 4], ["0-1", "2-3", "4+"])
MIN_STRATUM = 30


@T.tool("compare_members", "Compare two members: outcome rates and the features whose means differ most (in sd units).",
        T.obj_schema({"member_a": ("string", "first member"), "member_b": ("string", "second member")}), kind="analyze")
def compare_members(member_a, member_b):
    a, b = D.member(member_a), D.member(member_b)
    feats = D.features()
    diff = a.X_train.mean(axis=0) - b.X_train.mean(axis=0)      # standardized, so already in sd units
    order = np.argsort(-np.abs(diff))[:5]
    return {"member_a": {"name": member_a, "rows": a.n, "outcome_rate": round(float(a.y_train.mean()), 3)},
            "member_b": {"name": member_b, "rows": b.n, "outcome_rate": round(float(b.y_train.mean()), 3)},
            "largest_feature_differences_sd": [{"feature": feats[i], "a_minus_b_sd": round(float(diff[i]), 2)} for i in order],
            "reading": "A rate gap with small feature differences points to a base-rate (label) difference; large feature differences point to a different population."}


@T.tool("rank_drivers", "Within one member, which features move with the outcome (correlation with the label, sign and strength).",
        T.obj_schema({"member": ("string", "member name"), "top_k": ("integer", "how many features to return, default 5")}, required=["member"]), kind="analyze")
def rank_drivers(member, top_k=5):
    c = D.member(member)
    feats = D.features()
    y = c.y_train - c.y_train.mean()
    X = c.X_train - c.X_train.mean(axis=0)
    denom = (np.sqrt((X ** 2).sum(axis=0)) * np.sqrt((y ** 2).sum())) + 1e-9
    corr = (X * y[:, None]).sum(axis=0) / denom
    order = np.argsort(-np.abs(corr))[:int(top_k)]
    return {"member": member, "rows": c.n,
            "drivers": [{"feature": feats[i], "correlation_with_outcome": round(float(corr[i]), 3)} for i in order],
            "reading": "Compare the sign of a driver across members. A feature that is positive at every member but negative at one is a concept difference: the charter must decide what to do for that member."}


def _bands(c):
    age = np.array([int(r["age"]) for r in c.rows])
    com = np.array([int(r["comorbidity_index"]) for r in c.rows])
    return np.digitize(age, AGE_BANDS[0]), np.digitize(com, COMORB_BANDS[0])


@T.tool("stratified_comparison",
        "Compare two members' outcome rates WITHIN the same age band and comorbidity band (9 strata). Answers 'is the gap still "
        "there for similar patients?'. Strata with fewer than 30 rows at either member are skipped. Aggregate only.",
        T.obj_schema({"member_a": ("string", "first member"), "member_b": ("string", "second member")}), kind="analyze")
def stratified_comparison(member_a, member_b):
    A, B = D.member(member_a), D.member(member_b)
    (aa, ac), (ba, bc) = _bands(A), _bands(B)
    strata, diffs, wts = [], [], []
    for i, an in enumerate(AGE_BANDS[1]):
        for j, cn in enumerate(COMORB_BANDS[1]):
            ma, mb = (aa == i) & (ac == j), (ba == i) & (bc == j)
            if ma.sum() < MIN_STRATUM or mb.sum() < MIN_STRATUM:
                continue
            ra, rb = float(A.y_train[ma].mean()), float(B.y_train[mb].mean())
            strata.append({"age": an, "comorbidity": cn, "rate_a": round(ra, 3), "rate_b": round(rb, 3), "gap": round(ra - rb, 3)})
            diffs.append(ra - rb)
            wts.append(int(ma.sum() + mb.sum()))
    if not diffs:
        return {"error": "no stratum has at least 30 rows at both members"}
    within = float(np.average(diffs, weights=wts))
    raw = float(A.y_train.mean() - B.y_train.mean())
    higher = sum(d > 0 for d in diffs)
    if abs(within) >= 0.6 * abs(raw) and higher >= 0.75 * len(diffs):
        reading = (f"The gap survives matching on age and comorbidity ({member_a} higher in {higher} of {len(diffs)} strata), so it is not "
                   "explained by older or sicker patients. It is a base-rate difference: the same patient is readmitted more often at "
                   f"{member_a}. One shared intercept under-predicts there, which is why local fine-tuning helps it.")
    else:
        reading = "Most of the gap disappears within strata, so it is explained by who the patients are (case mix), not by the hospital."
    return {"member_a": member_a, "member_b": member_b, "raw_gap": round(raw, 3), "weighted_gap_within_strata": round(within, 3),
            "strata_a_higher": higher, "strata_compared": len(diffs), "strata": strata, "reading": reading}


@T.tool("feature_effect_by_member",
        "For one feature, its effect on the outcome at EVERY member in scope: for a yes/no feature the outcome rate with and without it; "
        "for a numeric feature the correlation. Flags any member whose sign is opposite to the majority (a concept difference). "
        "Optional min_age restricts to older patients. Aggregate only.",
        {"type": "object",
         "properties": {"feature": {"type": "string", "description": "a feature name from the data card, for example lives_alone"},
                        "members": {"type": "array", "items": {"type": "string"}, "description": "member names to include"},
                        "min_age": {"type": "integer", "description": "optional: only patients at or above this age"}},
         "required": ["feature", "members"]}, kind="analyze")
def feature_effect_by_member(feature, members, min_age=None):
    if feature not in D.features():
        return {"error": f"{feature} is not a feature; choose from {D.features()}"}
    lab = D.label()
    binary = next(f for f in D.world().schema["features"] if f["name"] == feature).get("type") == "binary"
    out = {}
    for name in members:
        c = D.member(name)
        rows = [r for r in c.rows if min_age is None or int(r["age"]) >= int(min_age)]
        y = np.array([int(r[lab]) for r in rows])
        x = np.array([float(r[feature]) for r in rows])
        if binary:
            on, off = x == 1, x == 0
            if on.sum() < 10 or off.sum() < 10:
                out[name] = {"rows": len(rows), "note": "fewer than 10 rows in a group; suppressed"}
                continue
            eff = float(y[on].mean() - y[off].mean())
            out[name] = {"rows": len(rows), "rate_with": round(float(y[on].mean()), 3), "rate_without": round(float(y[off].mean()), 3),
                         "effect": round(eff, 3), "rows_with": int(on.sum())}
        else:
            eff = float(np.corrcoef(x, y)[0, 1]) if len(rows) > 10 else 0.0
            out[name] = {"rows": len(rows), "correlation": round(eff, 3), "effect": round(eff, 3)}
    signs = [np.sign(v["effect"]) for v in out.values() if "effect" in v]
    majority = 1.0 if sum(s > 0 for s in signs) >= sum(s < 0 for s in signs) else -1.0
    reversed_at = [m for m, v in out.items() if "effect" in v and np.sign(v["effect"]) == -majority and abs(v["effect"]) >= 0.02]
    return {"feature": feature, "min_age": min_age, "kind": "binary" if binary else "numeric", "by_member": out,
            "majority_direction": "raises the outcome" if majority > 0 else "lowers the outcome", "reversed_at": reversed_at,
            "reading": (f"{feature} works the opposite way at {', '.join(reversed_at)}. A single federated weight is a compromise that "
                        "is wrong for that member; consider the charter's personalization clause." if reversed_at else
                        f"{feature} works the same direction at every member.")}


@T.tool("explain_change",
        "Why one member's outcome rate in a month differs from its other months: the rate gap and the features whose mean moved the "
        "most that month (in reference sd units), with each feature's direction of association. Aggregate only.",
        T.obj_schema({"member": ("string", "member name"), "month": ("integer", "1 to 12")}), kind="analyze")
def explain_change(member, month):
    c = D.member(member)
    month = int(month)
    mo = np.array([int(r["month"]) for r in c.rows])
    m = mo == month
    if m.sum() < 10:
        return {"error": f"only {int(m.sum())} rows in month {month} at {member}; too few to explain"}
    feats = D.features()
    shift = c.X_train[m].mean(axis=0) - c.X_train[~m].mean(axis=0)
    y = c.y_train - c.y_train.mean()
    X = c.X_train - c.X_train.mean(axis=0)
    corr = (X * y[:, None]).sum(axis=0) / ((np.sqrt((X ** 2).sum(axis=0)) * np.sqrt((y ** 2).sum())) + 1e-9)
    push = shift * np.sign(corr)          # a shift toward the risky direction of a feature pushes the rate up
    order = np.argsort(-np.abs(shift))[:4]
    seg_mix = {}
    for seg in D.world().segments:
        vals = sorted({r[seg] for r in c.rows})
        seg_mix[seg] = {v: {"share_this_month": round(float(np.mean([r[seg] == v for r, k in zip(c.rows, m) if k])), 3),
                            "share_other_months": round(float(np.mean([r[seg] == v for r, k in zip(c.rows, m) if not k])), 3)} for v in vals}
    return {"member": member, "month": month, "rows_in_month": int(m.sum()),
            "rate_this_month": round(float(c.y_train[m].mean()), 3), "rate_other_months": round(float(c.y_train[~m].mean()), 3),
            "feature_shifts_sd": [{"feature": feats[i], "shift_sd": round(float(shift[i]), 2),
                                   "pushes_rate": "up" if push[i] > 0 else "down"} for i in order],
            "segment_mix": seg_mix,
            "reading": "Shifts under about 0.15 sd in a month of ~200 discharges are within normal variation; a month whose rate moves "
                       "without a feature shift is noise or a cause the data does not record."}
