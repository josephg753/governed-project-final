"""
STEP 2 EVIDENCE: how the members differ, which difference changes a conclusion, and what the privacy control costs.

    python experiments/member_differences.py

Every experiment trains with the same engine and seed as run_federation.py (fl/federation.py, seed 690), so the numbers
are reproducible. Training stays federated: each member trains on its own rows; the coordinator only sees updates and
evaluates on neutral_eval.csv, per member. The one exception is the centralized baseline, which pools rows on purpose
as the forbidden reference point.

Writes:
    artifacts/experiments/member_differences.json   every number, machine readable
    artifacts/experiments/member_differences.md     the tables quoted in docs/model_card.md

Experiments:
    E1  member profile           rows, base rate per member (training files and neutral eval slice)
    E2  concept difference       does living alone raise or lower risk at each member? (rate gap and local model weight)
    E3  base-rate difference     St. Brigid vs Lakeshore General within the same age and comorbidity strata
    E4  non-IID skew             FedAvg with and without Harbor: what Harbor's reversed feature does to the shared model
    E5  personalization          FedAvg, then 3 local epochs of fine-tuning at each member (the charter clause it produced)
    E6  privacy cost             secure aggregation and DP noise multipliers 0, 0.3, 1.0, 2.0: accuracy per member and the gate
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from fl import federation as F, model as M  # noqa: E402
from fl.consortium import load_world  # noqa: E402

OUT = ROOT / "artifacts" / "experiments"
SEED = 690


def charter():
    return json.load(open(ROOT / "governance" / "federation_policy.json"))


def base_cfg(pol, **kw):
    c = dict(rounds=pol["rounds"], E=pol["max_local_epochs"], aggregator=pol["aggregation"], clip_norm=pol["max_update_norm"],
             secagg=pol["secure_aggregation"], dp_noise=pol["dp_noise_multiplier"], seed=SEED, verbose=False)
    c.update(kw)
    return F.Config(**c)


def per_member(theta, w):
    return {c.name: round(M.accuracy(theta, *w.test_for(k)), 3) for k, c in enumerate(w.clients)}


def e1_profile(w):
    out = {}
    for k, c in enumerate(w.clients):
        _, yt = w.test_for(k)
        out[c.name] = {"rows": c.n, "base_rate": round(float(c.y_train.mean()), 3), "eval_rows": int(len(yt)),
                       "eval_base_rate": round(float(yt.mean()), 3)}
    return out


def e2_lives_alone(w, local):
    fi = w.features.index("lives_alone")
    out = {}
    for c in w.clients:
        a = np.array([int(r["lives_alone"]) for r in c.rows]) == 1
        age = np.array([int(r["age"]) for r in c.rows])
        el = age >= 75
        out[c.name] = {"rate_lives_alone": round(float(c.y_train[a].mean()), 3), "rate_not_alone": round(float(c.y_train[~a].mean()), 3),
                       "gap_alone_minus_not": round(float(c.y_train[a].mean() - c.y_train[~a].mean()), 3),
                       "elderly_75plus_gap": round(float(c.y_train[el & a].mean() - c.y_train[el & ~a].mean()), 3),
                       "local_model_weight_lives_alone": round(float(local[c.name][fi]), 3)}
    return out


def e3_strata(w, a="St. Brigid", b="Lakeshore General"):
    def bands(c):
        age = np.array([int(r["age"]) for r in c.rows])
        com = np.array([int(r["comorbidity_index"]) for r in c.rows])
        return np.digitize(age, [65, 80]), np.digitize(com, [2, 4])
    names = {0: "<65", 1: "65-79", 2: "80+"}, {0: "0-1", 1: "2-3", 2: "4+"}
    A, B = w.member(a), w.member(b)
    (aa, ac), (ba, bc) = bands(A), bands(B)
    rows, diffs, weights = [], [], []
    for i in range(3):
        for j in range(3):
            ma, mb = (aa == i) & (ac == j), (ba == i) & (bc == j)
            if ma.sum() < 30 or mb.sum() < 30:
                continue
            ra, rb = float(A.y_train[ma].mean()), float(B.y_train[mb].mean())
            rows.append({"age": names[0][i], "comorbidity": names[1][j], f"{a} rate": round(ra, 3), f"{b} rate": round(rb, 3),
                         "gap": round(ra - rb, 3), "rows_a": int(ma.sum()), "rows_b": int(mb.sum())})
            diffs.append(ra - rb)
            weights.append(ma.sum() + mb.sum())
    return {"strata": rows, "weighted_mean_gap_within_strata": round(float(np.average(diffs, weights=weights)), 3),
            "raw_gap": round(float(A.y_train.mean() - B.y_train.mean()), 3),
            "strata_where_a_is_higher": sum(d > 0 for d in diffs), "strata_compared": len(diffs)}


def e4_skew(w, pol):
    """FedAvg with all members vs without Harbor. The model the others share changes when Harbor's reversed feature is in it."""
    fi = w.features.index("lives_alone")
    th_all, _ = F.run(w, base_cfg(pol))
    harbor = next(k for k, c in enumerate(w.clients) if c.name.startswith("Harbor"))
    import copy
    w2 = copy.copy(w)
    w2.clients = [c for k, c in enumerate(w.clients) if k != harbor]
    keep = w.test_source != harbor
    w2.X_test, w2.y_test = w.X_test[keep], w.y_test[keep]
    w2.test_source = np.array([s if s < harbor else s - 1 for s in w.test_source[keep]])
    th_wo, _ = F.run(w2, base_cfg(pol))
    acc_all = per_member(th_all, w)
    acc_wo = per_member(th_wo, w)   # Harbor scored by a model it did not help train
    return {"lives_alone_weight_with_harbor": round(float(th_all[fi]), 3), "lives_alone_weight_without_harbor": round(float(th_wo[fi]), 3),
            "accuracy_with_harbor": acc_all, "accuracy_without_harbor_in_training": acc_wo}


def e5_personalization(w, pol, theta, local, central):
    ep = pol.get("personalization", {}).get("local_finetune_epochs", 3)
    out = {}
    for k, c in enumerate(w.clients):
        Xt, yt = w.test_for(k)
        ft = M.local_train(theta, c.X_train, c.y_train, ep, 64, 0.1, rng=np.random.default_rng(SEED))
        out[c.name] = {"local_only": round(M.accuracy(local[c.name], Xt, yt), 3), "fedavg": round(M.accuracy(theta, Xt, yt), 3),
                       "fedavg_finetuned": round(M.accuracy(ft, Xt, yt), 3), "centralized": round(M.accuracy(central, Xt, yt), 3)}
        out[c.name]["finetune_gain_over_fedavg"] = round(out[c.name]["fedavg_finetuned"] - out[c.name]["fedavg"], 3)
        out[c.name]["fedavg_gain_over_local"] = round(out[c.name]["fedavg"] - out[c.name]["local_only"], 3)
    return {"local_finetune_epochs": ep, "per_member": out}


def e6_privacy(w, pol):
    runs = [("no privacy control (plain FedAvg)", False, 0.0, None), ("secure aggregation, no DP", True, 0.0, pol["max_update_norm"]),
            ("secure aggregation + DP 0.3 (charter)", True, 0.3, pol["max_update_norm"]),
            ("secure aggregation + DP 1.0", True, 1.0, pol["max_update_norm"]), ("secure aggregation + DP 2.0", True, 2.0, pol["max_update_norm"])]
    out = []
    for label, sa, dp, clip in runs:
        th, h = F.run(w, base_cfg(pol, secagg=sa, dp_noise=dp, clip_norm=clip))
        pc = {k: round(v, 3) for k, v in h[-1]["per_client"].items()}
        gap = round(max(pc.values()) - min(pc.values()), 3)
        ok = (h[-1]["global_acc"] >= pol["min_global_accuracy"] and min(pc.values()) >= pol["min_member_accuracy"]
              and gap <= pol["max_member_accuracy_gap"])
        out.append({"setting": label, "secure_aggregation": sa, "dp_noise_multiplier": dp, "global_accuracy": round(h[-1]["global_acc"], 4),
                    "per_member": pc, "gap": gap, "would_pass_accuracy_checks": bool(ok)})
    ref = out[0]
    for r in out:
        r["cost_vs_no_privacy"] = {m: round(r["per_member"][m] - ref["per_member"][m], 3) for m in r["per_member"]}
        r["global_cost"] = round(r["global_accuracy"] - ref["global_accuracy"], 4)
    return out


def table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def main():
    w = load_world(config.CONSORTIUM)
    pol = charter()
    print(f"Member-difference experiments on {w.title}, charter v{pol['policy_version']}, seed {SEED}")
    theta, _ = F.run(w, base_cfg(pol))
    local, central = F.local_only(w), F.centralized(w)
    res = {"consortium": w.name, "charter_version": pol["policy_version"], "seed": SEED,
           "E1_profile": e1_profile(w), "E2_lives_alone": e2_lives_alone(w, local), "E3_strata": e3_strata(w),
           "E4_skew": e4_skew(w, pol), "E5_personalization": e5_personalization(w, pol, theta, local, central),
           "E6_privacy": e6_privacy(w, pol)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "member_differences.json").write_text(json.dumps(res, indent=2))

    md = [f"# Member differences and privacy cost ({w.title})", "",
          f"Generated by `python experiments/member_differences.py` (charter v{pol['policy_version']}, seed {SEED}). Accuracy is on each member's slice of neutral_eval.csv (400 rows each).", ""]
    md += ["## E1 Member profile", table(["Member", "Training rows", "Base rate", "Eval base rate"],
           [[m, v["rows"], v["base_rate"], v["eval_base_rate"]] for m, v in res["E1_profile"].items()])]
    md += ["## E2 Concept difference: living alone", table(["Member", "Rate if lives alone", "Rate if not", "Gap", "Gap, age 75+", "Local model weight"],
           [[m, v["rate_lives_alone"], v["rate_not_alone"], v["gap_alone_minus_not"], v["elderly_75plus_gap"], v["local_model_weight_lives_alone"]] for m, v in res["E2_lives_alone"].items()])]
    e3 = res["E3_strata"]
    md += ["## E3 Base-rate difference: St. Brigid vs Lakeshore General at the same age and comorbidity",
           table(["Age", "Comorbidity", "St. Brigid", "Lakeshore General", "Gap", "Rows SB / LG"],
                 [[r["age"], r["comorbidity"], r["St. Brigid rate"], r["Lakeshore General rate"], r["gap"], f"{r['rows_a']} / {r['rows_b']}"] for r in e3["strata"]]),
           f"Raw gap {e3['raw_gap']}; weighted gap within strata {e3['weighted_mean_gap_within_strata']}; St. Brigid higher in {e3['strata_where_a_is_higher']} of {e3['strata_compared']} strata.", ""]
    e4 = res["E4_skew"]
    md += ["## E4 Non-IID skew: train with and without Harbor",
           f"lives_alone weight in the shared model: {e4['lives_alone_weight_with_harbor']} with Harbor, {e4['lives_alone_weight_without_harbor']} without.", "",
           table(["Member", "Accuracy, Harbor in training", "Accuracy, Harbor left out"],
                 [[m, e4["accuracy_with_harbor"][m], e4["accuracy_without_harbor_in_training"][m]] for m in e4["accuracy_with_harbor"]])]
    e5 = res["E5_personalization"]
    md += [f"## E5 Baselines and personalization ({e5['local_finetune_epochs']} local fine-tuning epochs after FedAvg)",
           table(["Member", "Local only", "FedAvg", "FedAvg + fine-tune", "Centralized", "FedAvg gain over local", "Fine-tune gain"],
                 [[m, v["local_only"], v["fedavg"], v["fedavg_finetuned"], v["centralized"], f"{v['fedavg_gain_over_local']:+.3f}", f"{v['finetune_gain_over_fedavg']:+.3f}"] for m, v in e5["per_member"].items()])]
    md += ["## E6 Privacy control and its cost",
           table(["Setting", "Global"] + [m.split(" ")[0] for m in res["E1_profile"]] + ["Gap", "Passes accuracy checks"],
                 [[r["setting"], r["global_accuracy"]] + [f"{r['per_member'][m]} ({r['cost_vs_no_privacy'][m]:+.3f})" for m in res["E1_profile"]] + [r["gap"], "yes" if r["would_pass_accuracy_checks"] else "NO"] for r in res["E6_privacy"]]),
           "Each cell: accuracy (change vs no privacy control)."]
    (OUT / "member_differences.md").write_text("\n".join(md) + "\n")
    print((OUT / "member_differences.md").read_text())


if __name__ == "__main__":
    main()
