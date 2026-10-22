"""
summarize.py -- merge all certify.py result files (*.json with keys args/results) of this folder into one table.
For every job (q, k, extra): HiGHS value, exact check of the HiGHS representative, best certified lower bound
(exhaustive search; maximum over runs), status.  Writes summary.json and summary.out (markdown table).
Usage: python -I summarize.py
"""
import glob
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    jobs, levels, files = {}, {}, []
    for fn in sorted(glob.glob(os.path.join(HERE, "*.json"))):
        name = os.path.basename(fn)
        if name.startswith(("summary", "z3")):
            continue
        with open(fn) as f:
            J = json.load(f)
        if not isinstance(J, dict) or "results" not in J:
            continue
        files.append(name)
        for R in J["results"]:
            L = R["level"]
            lv = levels.setdefault(L["d"], {"pins": True, "lifts_same": True, "sym": True, "runs": 0})
            lv["runs"] += 1
            lv["pins"] &= bool(L["pins_class"])
            lv["lifts_same"] &= bool(L.get("lifts_same_classes"))
            lv["sym"] &= bool(L["sym_level_ok"])
            lv.update(n=L["n"], rank_L=L["rank_L"], rank_S=L["rank_S"], index=L["index_L_S"], k=L["k"],
                      snf=L["snf_nonunit"])
            for r in R["jobs"]:
                j = r["job"]
                key = (j["q"], j["k"], tuple(j["extra_q"]))
                e = jobs.setdefault(key, {"q": j["q"], "k": j["k"], "d": j["d"], "extra": j["extra_q"],
                                          "srcs": j["srcs"], "highs": j["highs"], "rep_ok": False, "rep_len": None,
                                          "lb": 0, "runs": [], "genuine": 0, "selftest": None, "count": None})
                rc = r["rep_check"]
                e["rep_ok"] |= bool(rc.get("ok"))
                e["rep_len"] = rc.get("length")
                if r["certified_lower_bound"] is not None:
                    e["lb"] = max(e["lb"], r["certified_lower_bound"])
                e["genuine"] += r["search"]["genuine"]
                e["runs"].append({"file": name, "B": r["search"]["B"], "w1": r["search"]["w1"],
                                  "w2_max": r["search"]["w2_max"], "sec": r["search"]["sec"],
                                  "symmetry": r["target"]["symmetry"], "orbits": r["target"]["orbit_sizes"]})
                if r.get("selftest"):
                    e["selftest"] = r["selftest"]
                if r.get("count"):
                    e["count"] = r["count"]
    rows = []
    for key in sorted(jobs, key=lambda k: (jobs[k]["d"], k)):
        e = jobs[key]
        pins = levels[e["d"]]["pins"] and levels[e["d"]]["sym"]
        if not pins:
            st = "NOT CERTIFIED: level checks failed"
        elif e["genuine"]:
            st = "DISCREPANCY"
        elif e["rep_ok"] and e["rep_len"] == e["highs"] and e["lb"] == e["highs"]:
            st = "CERTIFIED"
        else:
            st = "LOWER BOUND ONLY"
        e["status"] = st
        best = max(e["runs"], key=lambda x: x["B"])
        rows.append("| %d | %d | %d | %s | %s | %d | %s %s | %d | B=%d, %.3g+%.3g, %.0f s | %s%s |" % (
            e["d"], e["q"], e["k"], ",".join(map(str, e["extra"])) or "-", ",".join(e["srcs"]), e["highs"],
            e["rep_len"], "ok" if e["rep_ok"] else "FAIL", e["lb"], best["B"], best["w1"], best["w2_max"],
            best["sec"], st, " (selftest at B=HiGHS: rep orbit found)" if (e["selftest"] or {}).get(
                "highs_rep_orbit_found") else ""))
    head = ["| d | q | k | extra | srcs | HiGHS | rep len, exact check | certified LB | search (B, |W1|+|W2|max, time) "
            "| status |", "|---|---|---|---|---|---|---|---|---|---|"]
    lv_lines = ["level d=%d: n=%d rank L=%d rank S=%d [L:S]=%s k=%d SNF non-unit=%s pins=%s lifts.class_functionals "
                "same classes=%s symmetry ok=%s" % (d, v["n"], v["rank_L"], v["rank_S"], v["index"], v["k"],
                                                     v["snf"], v["pins"], v["lifts_same"], v["sym"])
                for d, v in sorted(levels.items())]
    nC = sum(1 for e in jobs.values() if e["status"] == "CERTIFIED")
    tail = "%d jobs, %d levels: %d CERTIFIED, %d LOWER BOUND ONLY, %d other; all levels pin the class: %s; " \
           "all lower bounds >= 7 where HiGHS >= 7: %s; files: %s" % (
               len(jobs), len(levels), nC, sum(1 for e in jobs.values() if e["status"] == "LOWER BOUND ONLY"),
               sum(1 for e in jobs.values() if e["status"] not in ("CERTIFIED", "LOWER BOUND ONLY")),
               all(v["pins"] and v["sym"] and v["lifts_same"] for v in levels.values()),
               all(e["lb"] >= 7 for e in jobs.values() if e["highs"] >= 7), files)
    text = "\n".join(head + rows + [""] + lv_lines + ["", tail]) + "\n"
    with open(os.path.join(HERE, "summary.out"), "w", newline="\n") as f:
        f.write(text)
    with open(os.path.join(HERE, "summary.json"), "w", newline="\n") as f:
        json.dump({"jobs": [dict(e, key=list(map(str, k))) for k, e in sorted(jobs.items())],
                   "levels": levels, "files": files}, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
