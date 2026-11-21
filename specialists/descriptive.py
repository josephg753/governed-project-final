"""
DESCRIPTIVE SPECIALIST: what happened. Aggregates only; a member's rows never leave its file.

Tools: summarize_member, segment_breakdown, trend_by_month (template),
       segment_table_across_members, month_over_month (team).
"""
from collections import Counter, defaultdict

from agent import tools as T
from . import _data as D


def _rates_by(rows, segment):
    lab = D.label()
    tot, pos = Counter(), Counter()
    for r in rows:
        tot[r[segment]] += 1
        pos[r[segment]] += int(r[lab])
    return {k: {"rows": tot[k], "outcome_rate": round(pos[k] / tot[k], 3)} for k in sorted(tot, key=lambda k: -pos[k] / tot[k])}


@T.tool("summarize_member", "Row count, outcome rate and segment mix for one member. Aggregate only.",
        T.obj_schema({"member": ("string", "member name exactly as in the data card")}), kind="read")
def summarize_member(member):
    c = D.member(member)
    out = {"member": member, "rows": c.n, "outcome_rate": round(float(c.y_train.mean()), 3), "segments": {}}
    for seg in D.world().segments:
        cnt = Counter(r[seg] for r in c.rows)
        out["segments"][seg] = {k: v for k, v in cnt.most_common()}
    return out


@T.tool("segment_breakdown", "Outcome rate by one segment column for one member (for example by ward or by payer).",
        T.obj_schema({"member": ("string", "member name"), "segment": ("string", "a segment column from the data card")}), kind="read")
def segment_breakdown(member, segment):
    c = D.member(member)
    if segment not in D.world().segments:
        return {"error": f"{segment} is not a segment column; choose from {D.world().segments}"}
    return {"member": member, "segment": segment, "rates": _rates_by(c.rows, segment)}


@T.tool("trend_by_month", "Outcome rate by month (1 to 12) for one member.",
        T.obj_schema({"member": ("string", "member name")}), kind="read")
def trend_by_month(member):
    c = D.member(member)
    lab = D.label()
    tot, pos = defaultdict(int), defaultdict(int)
    for r in c.rows:
        m = int(r["month"])
        tot[m] += 1
        pos[m] += int(r[lab])
    rates = {m: round(pos[m] / tot[m], 3) for m in sorted(tot)}
    hi = max(rates, key=rates.get)
    lo = min(rates, key=rates.get)
    return {"member": member, "rate_by_month": rates, "highest_month": hi, "lowest_month": lo}


@T.tool("segment_table_across_members",
        "For one segment column (ward or payer), the outcome rate of every segment value at each listed member, plus the "
        "segment that is highest at each member. Each member computes its own table; only the rates cross. Aggregate only.",
        {"type": "object",
         "properties": {"segment": {"type": "string", "description": "a segment column from the data card, for example ward or payer"},
                        "members": {"type": "array", "items": {"type": "string"}, "description": "member names to include"},
                        "months": {"type": "array", "items": {"type": "integer"}, "description": "optional months (1 to 12), for example a quarter [10, 11, 12]"}},
         "required": ["segment", "members"]}, kind="read")
def segment_table_across_members(segment, members, months=None):
    if segment not in D.world().segments:
        return {"error": f"{segment} is not a segment column; choose from {D.world().segments}"}
    keep = {int(m) for m in months} if months else None
    table, top = {}, {}
    for name in members:
        c = D.member(name)
        rows = [r for r in c.rows if keep is None or int(r["month"]) in keep]
        rates = _rates_by(rows, segment) if rows else {}
        # suppress segments with very few rows: a rate on 4 patients is noise and nearly identifying
        rates = {k: v for k, v in rates.items() if v["rows"] >= 10}
        table[c.name] = rates
        if rates:
            k0 = next(iter(rates))
            top[c.name] = {"segment_value": k0, "outcome_rate": rates[k0]["outcome_rate"], "rows": rates[k0]["rows"]}
    return {"segment": segment, "months": sorted(keep) if keep else "all", "by_member": table, "highest_at_each_member": top,
            "note": "Segment values with fewer than 10 rows at a member are suppressed."}


@T.tool("month_over_month", "One member's outcome rate in a given month compared with the month before and with its yearly average.",
        T.obj_schema({"member": ("string", "member name"), "month": ("integer", "1 to 12")}), kind="read")
def month_over_month(member, month):
    tr = trend_by_month(member)["rate_by_month"]
    month = int(month)
    if month not in tr:
        return {"error": f"no rows for month {month} at {member}"}
    prev = month - 1 if month > 1 else 12
    avg = round(sum(tr.values()) / len(tr), 3)
    c = D.member(member)
    n = sum(1 for r in c.rows if int(r["month"]) == month)
    return {"member": member, "month": month, "rows_in_month": n, "rate": tr[month], "previous_month": prev,
            "previous_rate": tr.get(prev), "change_vs_previous": round(tr[month] - tr.get(prev, tr[month]), 3),
            "year_average": avg, "change_vs_average": round(tr[month] - avg, 3)}
