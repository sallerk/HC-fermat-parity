"""
certify.py -- driver for cert_core (exact certification of the lifted lengths of NOTE.txt Section 7).

Jobs are read from the frozen lifts.json (sections Q, S, T: q, k, d, extra classes, HiGHS value, representative).
For each job: exact level checks (cert_core.setup_level), exact re-check of the HiGHS representative, exhaustive
exclusion of all class elements of length <= B (B = HiGHS value - 1 if within the work caps, else the largest
feasible B), optional self-test (search at B = HiGHS value must find elements, among them a unit translate of the
HiGHS representative), optional count of all elements of a given length.

Usage: python -I certify.py OUTNAME [--src Q,S,T] [--q 15,21] [--dmin D] [--dmax D] [--procs P] [--selftest]
                                     [--count D:B] [--Bfix B] [--plan] [--w1cap X] [--w2cap Y] [--deadline SEC]
Writes OUTNAME.json and OUTNAME.out next to this script.
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import cert_core as CC  # noqa: E402


def load_jobs():
    with open(CC.LIFTS_JSON) as f:
        J = json.load(f)
    res = J["results"]
    jobs = {}

    def add(src, row, extra_info=None):
        q, k = row["q"], row["k"]
        l, p = CC.lp_of(q)
        extra = [list(CC.lp_of(x)) for x in row.get("extra", [])]
        key = (q, k, tuple(tuple(e) for e in extra))
        j = jobs.get(key)
        if j is None:
            j = dict(q=q, l=l, p=p, k=k, d=row["d"], extra=extra, extra_q=list(row.get("extra", [])),
                     highs=row["least"], rep=row["rep"], lifts_verified=row.get("verified"), srcs=[])
            jobs[key] = j
        elif j["highs"] != row["least"]:
            j.setdefault("conflicts", []).append([src, row["least"]])
        j["srcs"].append(src)
        if extra_info:
            j.update(extra_info)

    for row in res["Q"]["table"]:
        add("Q", row, {"orbit_program": row.get("orbit_program")})
    for row in res["S"]:
        add("S", row)
    for row in res["T"]:
        add("T", row, {"mmrv": row.get("mmrv")})
    return list(jobs.values())


def choose_B(sizes, v, w1cap, w2cap, Bfix=None):
    if Bfix is not None:
        return min(Bfix, v - 1)
    for B in range(v - 1, 0, -1):
        w1, w2 = CC.plan_cost(sizes, B)
        if (w1 <= w1cap and w2 <= w2cap) or B <= 5:
            return B
    return 0


def run_job(lev, job, opts):
    t0 = time.time()
    d, target = CC.LF.lifted(job["l"], job["p"], job["k"], tuple(tuple(e) for e in job["extra"]))
    assert d == lev["d"] == job["d"]
    tg = CC.setup_target(lev, target)
    rc = CC.check_rep(lev, tg, job["rep"])
    v = job["highs"]
    sizes = tg["info"]["orbit_sizes"]
    B = choose_B(sizes, v, opts["w1cap"], opts["w2cap"], opts.get("Bfix"))
    res, _ = CC.search(lev, tg, B)
    out = {"job": {k: job[k] for k in job if k != "rep"}, "rep": job["rep"], "target": tg["info"],
           "rep_check": {k: x for k, x in rc.items() if k != "vec"}, "search": res}
    if opts["selftest"] and rc["ok"]:
        w1, w2 = CC.plan_cost(sizes, v)
        if w1 <= opts["st_w1cap"] and w2 <= opts["st_w2cap"]:
            st, coll = CC.search(lev, tg, v, collect=True, collect_cap=500000)
            imgs = CC.orbit_images(lev, rc["vec"])
            out["selftest"] = {"B": v, "genuine_matches": st["genuine"], "min_norm": st["min_norm"],
                               "distinct_normalized": len(coll), "highs_rep_orbit_found": bool(imgs & coll),
                               "sec": st["sec"]}
    if d in opts["count"]:
        Bc = opts["count"][d]
        st, coll = CC.search(lev, tg, Bc, collect=True, collect_cap=5_000_000)
        allsol = set()
        for row in coll:
            allsol |= CC.orbit_images(lev, np.frombuffer(row, np.int8).astype(np.int64))
        bynorm = {}
        for row in allsol:
            nn = int(np.abs(np.frombuffer(row, np.int8).astype(np.int64)).sum())
            bynorm[nn] = bynorm.get(nn, 0) + 1
        out["count"] = {"B": Bc, "distinct_normalized": len(coll), "all_solutions_by_norm": bynorm,
                        "collect_cap_hit": len(coll) >= 5_000_000, "sec": st["sec"]}
    pins = lev["info"]["pins_class"] and tg["info"]["target_hodge"] and tg["info"]["target_nontrivial"]
    if not pins:
        status = "NOT CERTIFIED: exact class checks failed"
    elif res["genuine"]:
        status = "DISCREPANCY: class element of norm %d <= %d found" % (res["min_norm"], B)
    elif not rc["ok"] or rc["length"] != v:
        status = "LOWER BOUND ONLY (>= %d); HiGHS representative not verified" % (B + 1)
    elif B >= v - 1:
        status = "CERTIFIED"
    else:
        status = "LOWER BOUND ONLY (>= %d)" % (B + 1)
    out["status"] = status
    out["certified_lower_bound"] = (B + 1) if (pins and not res["genuine"]) else None
    out["sec"] = round(time.time() - t0, 1)
    return out


def run_task(task):
    d, jobs, opts = task
    t0 = time.time()
    lev = CC.setup_level(d)
    outs = [run_job(lev, job, opts) for job in jobs]
    return {"d": d, "level": lev["info"], "jobs": outs, "sec": round(time.time() - t0, 1)}


def conv(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def job_line(r):
    j, s, rc = r["job"], r["search"], r["rep_check"]
    return ("  q=%d k=%d d=%d extra=%s HiGHS=%d | rep ok=%s len=%s | B=%d w1=%d w2max=%d keymatch=%d genuine=%d "
            "%.1fs | %s%s" % (j["q"], j["k"], j["d"], j["extra_q"], j["highs"], rc.get("ok"), rc.get("length"),
                               s["B"], s["w1"], s["w2_max"], s["key_matches"], s["genuine"], s["sec"], r["status"],
                               (" | selftest " + json.dumps(r["selftest"])) if "selftest" in r else "")
            + ((" | count " + json.dumps(r["count"])) if "count" in r else ""))


def level_line(L):
    return ("level d=%d n=%d rank_L=%d rank_S=%d [L:S]=%s SNF-nonunit=%s k=%d pins=%s lifts_same=%s lifts_rows=%s "
            "std=%s sym=%s maxL=%d setup %.1fs" % (L["d"], L["n"], L["rank_L"], L["rank_S"], L["index_L_S"],
                                                  L["snf_nonunit"], L["k"], L["pins_class"],
                                                  L.get("lifts_same_classes"), L.get("lifts_rows_equal"),
                                                  L["standard_tuples_equal_v2_exotic"], L["sym_level_ok"],
                                                  L["max_entry_L_basis"], L["setup_sec"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--src", default="Q,S,T")
    ap.add_argument("--q", default="")
    ap.add_argument("--dmin", type=int, default=0)
    ap.add_argument("--dmax", type=int, default=10 ** 9)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--w1cap", type=float, default=2.5e8)
    ap.add_argument("--w2cap", type=float, default=3e7)
    ap.add_argument("--st_w1cap", type=float, default=3e7)
    ap.add_argument("--st_w2cap", type=float, default=5e6)
    ap.add_argument("--Bfix", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--count", default="")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--noextra", action="store_true")
    ap.add_argument("--extra", default="")
    ap.add_argument("--deadline", type=float, default=560.0)
    a = ap.parse_args()
    T0 = time.time()
    srcs = set(a.src.split(","))
    qs = {int(x) for x in a.q.split(",") if x}
    jobs = [j for j in load_jobs() if set(j["srcs"]) & srcs and (not qs or j["q"] in qs)
            and a.dmin <= j["d"] <= a.dmax and not (a.noextra and j["extra"])
            and (not a.extra or j["extra_q"] == [int(x) for x in a.extra.split(",")])]
    opts = {"w1cap": a.w1cap, "w2cap": a.w2cap, "st_w1cap": a.st_w1cap, "st_w2cap": a.st_w2cap, "Bfix": a.Bfix,
            "selftest": a.selftest,
            "count": {int(x.split(":")[0]): int(x.split(":")[1]) for x in a.count.split(",") if x}}
    by_d = {}
    for j in jobs:
        sizes = CC.gcd_orbit_sizes(j["d"])
        B = choose_B(sizes, j["highs"], a.w1cap, a.w2cap, a.Bfix)
        w1, w2 = CC.plan_cost(sizes, B)
        j["_plan"] = (B, w1, w2)
        by_d.setdefault(j["d"], []).append(j)
    lines = []
    if a.plan:
        for d in sorted(by_d):
            for j in by_d[d]:
                B, w1, w2 = j["_plan"]
                lines.append("plan d=%d q=%d k=%d extra=%s HiGHS=%d srcs=%s -> B=%d (%s) w1=%.3g w2=%.3g" % (
                    d, j["q"], j["k"], j["extra_q"], j["highs"], ",".join(j["srcs"]), B,
                    "exact" if B >= j["highs"] - 1 else "lower bound", w1, w2))
        print("\n".join(lines))
        print("%d jobs, %d levels" % (len(jobs), len(by_d)))
        return
    tasks = sorted(((d, [{k: x for k, x in j.items() if k != "_plan"} for j in js], opts) for d, js in by_d.items()),
                   key=lambda t: -sum(j["_plan"][1] + 3 * j["_plan"][2] for j in by_d[t[0]]) - 2000 * t[0])
    print("%d jobs in %d levels, %d processes" % (len(jobs), len(tasks), a.procs), flush=True)
    done, unfinished = [], []
    with multiprocessing.Pool(a.procs) as pool:
        it = pool.imap_unordered(run_task, tasks)
        for _ in range(len(tasks)):
            remaining = a.deadline - (time.time() - T0)
            try:
                r = it.next(timeout=max(1.0, remaining))
            except multiprocessing.TimeoutError:
                pool.terminate()
                break
            done.append(r)
            ls = [level_line(r["level"])] + [job_line(x) for x in r["jobs"]]
            lines += ls
            print("\n".join(ls), flush=True)
    got = {r["d"] for r in done}
    unfinished = sorted(d for d in by_d if d not in got)
    tail = "finished %d of %d levels in %.1f s; unfinished levels: %s" % (len(done), len(tasks), time.time() - T0,
                                                                         unfinished)
    print(tail, flush=True)
    lines.append(tail)
    with open(os.path.join(HERE, a.out + ".json"), "w", newline="\n") as f:
        json.dump({"args": vars(a), "results": sorted(done, key=lambda r: r["d"]), "unfinished": unfinished,
                   "seconds": round(time.time() - T0, 1)}, f, indent=1, default=conv)
    with open(os.path.join(HERE, a.out + ".out"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
