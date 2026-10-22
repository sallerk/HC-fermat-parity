"""
z3check.py -- independent cross-check of the degree-2q values of Theorem 18 with z3 (exact linear integer
arithmetic), using the level data of cert_core (Hodge rows, class functionals F, target xi_q lifted to 2q).
For each q: "Hodge, F(alpha) = F(xi_q) mod 2, sum |alpha_u| <= B" must be unsat for B = value - 1 and sat for
B = value; a sat model is re-checked with exact integer arithmetic.
Usage: python -I z3check.py OUTNAME TIMEOUT_SEC q1,q2,...
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import z3  # noqa: E402

import cert_core as CC  # noqa: E402

VALUES = {15: 3, 21: 3, 33: 4, 35: 6, 39: 3, 51: 7, 55: 5, 57: 6, 65: 10, 69: 9}


def solve(lev, tg, B, timeout):
    n = lev["n"]
    a = [z3.Int("a%d" % i) for i in range(n)]
    pp = [z3.Int("p%d" % i) for i in range(n)]
    mm = [z3.Int("m%d" % i) for i in range(n)]
    s = z3.SolverFor("QF_LIA")
    s.set("timeout", int(timeout * 1000))
    for i in range(n):
        s.add(pp[i] >= 0, mm[i] >= 0, a[i] == pp[i] - mm[i])
    s.add(z3.Sum(pp + mm) <= B)
    for row in lev["R"]:
        s.add(z3.Sum([c * a[i] for i, c in enumerate(row) if c]) == 0)
    for j in range(lev["k"]):
        sj = z3.Int("s%d" % j)
        s.add(z3.Sum([a[i] for i in range(n) if lev["F_np"][j, i]]) == 2 * sj + int(tg["cvec"][j]))
    t0 = time.time()
    r = s.check()
    out = {"B": B, "result": str(r), "sec": round(time.time() - t0, 2)}
    if r == z3.sat:
        mdl = s.model()
        v = np.array([mdl.eval(a[i], model_completion=True).as_long() for i in range(n)], dtype=np.int64)
        out["model_norm"] = int(np.abs(v).sum())
        out["model_hodge"] = not (lev["R_np"] @ v).any()
        out["model_in_class"] = bool((((lev["F_np"] @ v) % 2) == tg["cvec"]).all())
    return out


def main():
    name, timeout, qs = sys.argv[1], float(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
    res, lines = [], []
    for q in qs:
        l, p = CC.lp_of(q)
        d, target = CC.LF.lifted(l, p, 2, ())
        lev = CC.setup_level(d, compare_lifts=False)
        tg = CC.setup_target(lev, target)
        v = VALUES[q]
        if len(sys.argv) > 4:                       # only "no element of length <= B" (unsat expected)
            Bq = int(sys.argv[4])
            r1 = solve(lev, tg, Bq, timeout)
            line = "q=%d d=%d: B=%d %s (%.1fs) -> %s" % (q, d, Bq, r1["result"], r1["sec"],
                                                        "no element of length <= %d" % Bq
                                                        if r1["result"] == "unsat" else "NOT confirmed")
            print(line, flush=True)
            lines.append(line)
            res.append({"q": q, "d": d, "B": Bq, "pins": lev["info"]["pins_class"], "below": r1})
            continue
        r1 = solve(lev, tg, v - 1, timeout)
        r2 = solve(lev, tg, v, timeout)
        ok = r1["result"] == "unsat" and r2["result"] == "sat" and r2.get("model_hodge") and \
            r2.get("model_in_class") and r2.get("model_norm", 99) <= v
        res.append({"q": q, "d": d, "value": v, "pins": lev["info"]["pins_class"], "below": r1, "at": r2,
                    "confirms_value": bool(ok)})
        line = "q=%d d=%d value=%d: B=%d %s (%.1fs); B=%d %s (%.1fs, model norm %s hodge %s class %s) -> %s" % (
            q, d, v, v - 1, r1["result"], r1["sec"], v, r2["result"], r2["sec"], r2.get("model_norm"),
            r2.get("model_hodge"), r2.get("model_in_class"), "confirmed" if ok else "NOT confirmed")
        print(line, flush=True)
        lines.append(line)
    with open(os.path.join(HERE, name + ".json"), "w", newline="\n") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(HERE, name + ".out"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
