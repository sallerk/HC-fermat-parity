"""
compare_outputs.py -- compare the result files of a rerun with the reference outputs.

Usage:  python -I compare_outputs.py REFERENCE_DIR RERUN_DIR
Every .json and .out file in REFERENCE_DIR is compared with the file of the same name in RERUN_DIR; files whose
names end in "_stdout.out" are skipped.  Some parts of the results depend on the run, not on the mathematics, and
are normalized in both files before the comparison:
  - run times: in .json files the keys "seconds", "sec" and the keys ending in "_sec" are removed; in .out files a
    number followed by "s" (as in "134.3s" or "135.4 s") and the values of such keys are replaced by "<t>";
  - order: certify.py runs the levels in parallel worker processes, so the order of its results depends on which
    worker finishes first; the lines of .out files are compared as sorted lists, and in .json files every list of
    records that have a key "d" (one record per level) is sorted;
  - solver tuples: the integer-programming solver may return a different tuple of the same optimal length, so a
    list under the key "rep" is replaced by its length (the scripts check every tuple exactly and record the result
    of that check, which is compared);
  - options: keys of "args" that occur in only one of the two files are ignored (result files written by earlier
    versions of certify.py lack its options --extra and --noextra, whose default values give the earlier behaviour);
  - time limits: a z3 query (files z3_*) that reached its time limit ("unknown") in one of the two runs is not
    compared, and neither is the verdict that depends on it; such files are reported as "equal up to z3 time limits".
Each file is reported as "equal", "equal up to z3 time limits" or "DIFFERENT".  Expected last line: "ALL EQUAL".
"""

import json
import os
import re
import sys

TIME = re.compile(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)? ?s\b")
TIME_KEY_TEXT = re.compile(r'"(seconds|sec|\w+_sec)": -?\d+(?:\.\d+)?(?:e[+-]?\d+)?')
Z3_RESULT = re.compile(r"B=(\d+) (unsat|sat|unknown)\b")


def is_time_key(k):
    return k in ("seconds", "sec") or k.endswith("_sec")


def norm(o):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if is_time_key(k):
                continue
            out[k] = ("length", len(v)) if k == "rep" and isinstance(v, list) else norm(v)
        return out
    if isinstance(o, list):
        lst = [norm(v) for v in o]
        if lst and all(isinstance(x, dict) and "d" in x for x in lst):
            lst.sort(key=lambda x: json.dumps(x, sort_keys=True))
        return lst
    return o


def mask_z3_records(A, B):
    """For z3 result lists: drop the results of queries that reached the time limit in one run, and the verdict.
    The records of the two runs are paired by (q, d)."""
    masked = 0
    if not (isinstance(A, list) and isinstance(B, list)):
        return masked
    key_of = lambda r: (r.get("q"), r.get("d")) if isinstance(r, dict) else None
    other = {key_of(b): b for b in B}
    for a in A:
        b = other.get(key_of(a))
        if not (isinstance(a, dict) and isinstance(b, dict)):
            continue
        for key in ("below", "at"):
            qa, qb = a.get(key), b.get(key)
            if isinstance(qa, dict) and isinstance(qb, dict) and qa.get("result") != qb.get("result") \
                    and "unknown" in (qa.get("result"), qb.get("result")):
                qa.pop("result", None)
                qb.pop("result", None)
                for r in (a, b):
                    r.pop("confirms_value", None)
                masked += 1
    return masked


def out_lines(path):
    with open(path, encoding="utf-8") as f:
        return [TIME_KEY_TEXT.sub(lambda m: '"%s": <t>' % m.group(1), TIME.sub("<t>", line.rstrip())) for line in f]


def mask_z3_lines(la, lb):
    """Pair the lines of two z3 .out files by the text before ":" and mask the results that timed out in one run."""
    masked = 0
    ka = {x.split(":", 1)[0]: x for x in la}
    kb = {x.split(":", 1)[0]: x for x in lb}
    for k in set(ka) & set(kb):
        x, y = ka[k], kb[k]
        rx, ry = Z3_RESULT.findall(x), Z3_RESULT.findall(y)
        if x == y or len(rx) != len(ry):
            continue
        if all(u == v or (u[0] == v[0] and "unknown" in (u[1], v[1])) for u, v in zip(rx, ry)):
            differ = {u[0] for u, v in zip(rx, ry) if u != v}
            mask = lambda m: "B=%s <result>" % m.group(1) if m.group(1) in differ else m.group(0)
            ka[k] = Z3_RESULT.sub(mask, x).split(" -> ")[0]
            kb[k] = Z3_RESULT.sub(mask, y).split(" -> ")[0]
            masked += 1
    return sorted(ka.values()), sorted(kb.values()), masked


def main():
    ref, new = sys.argv[1], sys.argv[2]
    allequal = True
    for name in sorted(os.listdir(ref)):
        if not name.endswith((".json", ".out")) or name.endswith("_stdout.out"):
            continue
        a = os.path.join(ref, name)
        b = os.path.join(new, name)
        if not os.path.exists(b):
            print("%-28s missing in rerun" % name)
            allequal = False
            continue
        masked = 0
        if name.endswith(".json"):
            with open(a, encoding="utf-8") as f:
                A = json.load(f)
            with open(b, encoding="utf-8") as f:
                B = json.load(f)
            if isinstance(A, dict) and isinstance(B, dict) and isinstance(A.get("args"), dict) \
                    and isinstance(B.get("args"), dict):
                common = set(A["args"]) & set(B["args"])
                A["args"] = {k: v for k, v in A["args"].items() if k in common}
                B["args"] = {k: v for k, v in B["args"].items() if k in common}
            if name.startswith("z3_"):
                masked = mask_z3_records(A, B)
            A, B = norm(A), norm(B)
        else:
            A, B = sorted(out_lines(a)), sorted(out_lines(b))
            if name.startswith("z3_"):
                A, B, masked = mask_z3_lines(A, B)
        eq = A == B
        print("%-28s %s" % (name, "DIFFERENT" if not eq else
                            "equal up to z3 time limits" if masked else "equal"))
        allequal &= eq
    print("ALL EQUAL" if allequal else "DIFFERENCES FOUND")


if __name__ == "__main__":
    main()
