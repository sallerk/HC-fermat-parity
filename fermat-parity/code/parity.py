"""
parity.py -- checks for ../NOTE.txt: Aoki's exceptional elements xi_q, q = l p (l in {4} + odd primes, p an odd
prime), and the parity of the number of their entries of a given order.

 (A) For every q = l p <= 400 with l < p odd primes: xi_q built from [MMRV] Def. 4.8 (half systems of powers of
     the least primitive root; d-parts from the recursion with the tau maps of [MMRV] (24), with the sign that
     makes xi_q Hodge, see NOTE 1) is a Hodge tuple; its order-l part is -sum_{h in H_l cap u H_l} (h)_l with
     u = -p^(-1) mod l (NOTE Lemma 2); the number of its entries of order l is odd iff (p / l) = -1 (Theorem 3(i));
     and its length is even, which is quadratic reciprocity for (l, p) (Remark 4).  For q = 4p <= 400: the order-4
     part of xi_q is +-(1)_4 if p = 3 mod 4 and 0 if p = 1 mod 4.
 (B) On the lattice L_q (B_q / D_q) for every q = l p and 4p <= 400 (97 degrees): the count of entries of order l
     is nonzero on B_q iff chi_l(p) = -1, and that of order p iff (l0 / p) = -1 (l0 = l, or 2 if l = 4).
 (C) For the same degrees and each count with symbol -1: the count is even on every generator of the base span
     (lifted standard, Lefschetz, curve pairs) except for q in {12, 15, 20, 21, 28}.
 (D) B_q / (base span) by Smith normal form for the 97 degrees l p, 4p <= 400 (l in {3, 4, 5, 7, 11, 13, 17,
     19}): Z/2 except for q in {12, 15, 20, 21, 28, 39}, where it is 0; and a count functional detects it iff one
     of the two symbols is -1.
Usage:  python -I parity.py OUT.json      Expected last line "ALL CHECKS PASS: True".  About 20 minutes (3 cores).
"""

import itertools
import json
import multiprocessing
import os
import sys
import time
from math import gcd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "cyclic-covers", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "fermat-frontier", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ts-powers", "code"))
import fermat_tools as FT  # noqa: E402
from v2_exotic import hodge4, lattice_L  # noqa: E402

KILLED = [12, 15, 20, 21, 28]          # xi_q in the base span although a symbol is -1
ZERO = [12, 15, 20, 21, 28, 39]        # B_q = base span
LS = (3, 4, 5, 7, 11, 13, 17, 19)


def isprime(n):
    return n > 1 and all(n % r for r in range(2, int(n ** 0.5) + 1))


def leg(a, l):
    return 1 if pow(a % l, (l - 1) // 2, l) == 1 else -1


def chi(a, l):
    return (1 if a % 4 == 1 else -1) if l == 4 else leg(a, l)


def degrees(bound=400):
    out = []
    for l in LS:
        for p in range(3, bound):
            if isprime(p) and p != l and (l == 4 or p > l) and l * p <= bound:
                out.append((l, p))
    return sorted(out, key=lambda x: x[0] * x[1])


def prim_root(p):
    return next(g for g in range(2, p) if all(pow(g, (p - 1) // r, p) != 1 for r in range(2, p)
                                              if (p - 1) % r == 0 and isprime(r)))


def half(l):
    """Aoki's half system: (1)_4 for l = 4, else 1, mu, ..., mu^((l-3)/2) for the least primitive root mu."""
    if l == 4:
        return [1]
    mu = prim_root(l)
    return [pow(mu, j, l) for j in range((l - 1) // 2)]


def add(v, x, m, c):
    """add c (x)_m to v, an element of R_m / D_m with m odd or 4 (stored on the half system 2x < m)."""
    x %= m
    if 2 * x < m:
        v[x] = v.get(x, 0) + c
    else:
        v[m - x] = v.get(m - x, 0) - c


def crt(a, m, b, n):
    return next(c for c in range(m * n) if c % m == a % m and c % n == b % n)


def xi_odd(l, p):
    """xi_q for q = l p, l, p odd: (primitive part on R_q, order-l part on R_l, order-p part on R_p)."""
    q = l * p
    prim = {}
    for h in half(l):
        for k in half(p):
            add(prim, crt(h, l, k, p), q, 1)
    parts = {}
    for d, m in ((p, l), (l, p)):
        u = (-pow(d, -1, m)) % m
        s = {}
        for c, n in prim.items():
            add(s, c % m, m, n)
            add(s, u * c % m, m, n)
        assert all(v % (d - 1) == 0 for v in s.values())
        parts[d] = {x: -(v // (d - 1)) for x, v in s.items() if v}
    return prim, parts[p], parts[l]


def as_tuple(q, pieces):
    """pieces: list of (multiplier e, modulus m, element of R_m / D_m); returns a multiset of residues mod q."""
    t = []
    for e, m, v in pieces:
        for x, c in v.items():
            t += [(e * (x if c > 0 else m - x)) % q] * abs(c)
    return tuple(sorted(t))


def lemma2(l, p):
    """-sum over h in H_l cap u H_l of (h)_l, u = -p^-1 mod l."""
    H = half(l)
    u = (-pow(p, -1, l)) % l
    uH = {(u * h) % l for h in H}
    v = {}
    for h in H:
        if h in uH:
            add(v, h, l, -1)
    return v


def check_A():
    rows, ok = [], True
    for l, p in degrees():
        if l == 4:
            continue
        q = l * p
        prim, part_l, part_p = xi_odd(l, p)
        t = as_tuple(q, [(1, q, prim), (p, l, part_l), (l, p, part_p)])
        hodge = FT.is_hodge(t, q)
        lem = part_l == lemma2(l, p) and part_p == lemma2(p, l)
        odd_l = sum(1 for x in t if q // gcd(x, q) == l) % 2 == 1
        odd_p = sum(1 for x in t if q // gcd(x, q) == p) % 2 == 1
        rule = odd_l == (leg(p, l) == -1) and odd_p == (leg(l, p) == -1)
        qr = (((l - 1) // 2) * ((p - 1) // 2) - odd_l - odd_p) % 2 == 0
        ok &= hodge and lem and rule and qr and len(t) % 2 == 0
        rows.append({"q": q, "l": l, "p": p, "length": len(t), "hodge": hodge, "lemma2": lem,
                     "odd_order_l": odd_l, "odd_order_p": odd_p, "rule": rule})
    for l, p in degrees():
        if l != 4:
            continue
        # order-4 part: (1/phi(p)) (1 + [-p^-1]) applied to the primitive part (1)_4 x H_p, up to sign
        s = {}
        for k in half(p):
            add(s, 1, 4, 1)
            add(s, (-pow(p, -1, 4)) % 4, 4, 1)
        assert all(v % (p - 1) == 0 for v in s.values())
        part4 = {x: v // (p - 1) for x, v in s.items() if v}
        good = (part4 == {1: 1}) if p % 4 == 3 else (part4 == {})
        ok &= good
        rows.append({"q": 4 * p, "order4_part": part4, "as_predicted": good})
    print("    (A) xi_q Hodge, Lemma 2, Theorem 3(i), reciprocity, order-4 parts: %s" % ok, flush=True)
    return rows, ok


def lattice_job(lp):
    l, p = lp
    q = l * p
    Kp, Lb = lattice_L(q)

    def nonzero(order):
        return any(sum(x for u, x in zip(Kp, e) if q // gcd(u, q) == order) % 2 for e in Lb)
    return q, nonzero(l), chi(p, l) == -1, nonzero(p), leg(2 if l == 4 else l, p) == -1


def base_job(lp):
    l, p = lp
    q = l * p
    gens = FT.standard_generators(q)[0] + [FT.evec(b, q) for b in hodge4(q)] + FT.curve_pair_generators(q)
    Kp = [u for u in range(1, q) if 2 * u < q]
    out = []
    for order, sym in ((l, chi(p, l)), (p, leg(2 if l == 4 else l, p))):
        if sym == -1:
            idx = [i for i, u in enumerate(Kp) if q // gcd(u, q) == order]
            out.append((order, all(sum(v[i] for i in idx) % 2 == 0 for v in gens)))
    return q, out


def snf_job(lp):
    import obstruction as OB
    l, p = lp
    q = l * p
    divisors, Kp, Lb, phis, H, n = OB.obstruction(q)
    cf = OB.count_functionals(q, Kp, Lb, phis) if divisors == [2] else []
    sym = chi(p, l) == -1 or leg(2 if l == 4 else l, p) == -1
    return q, divisors, bool(cf), sym


def check_B(pool):
    rows, ok = [], True
    for q, nl, pl, np_, pp in pool.map(lattice_job, degrees()):
        good = nl == pl and np_ == pp
        ok &= good
        rows.append({"q": q, "order_l_nonzero": nl, "order_p_nonzero": np_, "as_predicted": good})
    print("    (B) counts nonzero on B_q exactly as the residue symbols predict, %d degrees: %s" % (len(rows), ok),
          flush=True)
    return rows, ok


def check_C(pool):
    rows, ok = [], True
    for q, out in pool.map(base_job, degrees()):
        for order, van in out:
            ok &= van == (q not in KILLED)
            rows.append({"q": q, "order": order, "even_on_base_generators": van})
    print("    (C) counts with symbol -1 even on the base generators except for %s: %s" % (KILLED, ok), flush=True)
    return rows, ok


def check_D(pool):
    rows, ok = [], True
    for q, divisors, count, sym in pool.map(snf_job, degrees()):
        good = (divisors == [] if q in ZERO else divisors == [2] and count == sym)
        ok &= good
        rows.append({"q": q, "quotient": divisors, "count_functional": count, "symbol_minus_one": sym,
                     "as_predicted": good})
    print("    (D) B_q / base = Z/2 (0 for %s), detected by a count iff a symbol is -1, %d degrees: %s" % (
        ZERO, len(rows), ok), flush=True)
    return rows, ok


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "parity.json"
    res, allok, secs = {}, True, {}
    with multiprocessing.Pool(3) as pool:
        for key, fn in (("A", check_A), ("B", lambda: check_B(pool)), ("C", lambda: check_C(pool)),
                        ("D", lambda: check_D(pool))):
            t0 = time.time()
            r, ok = fn()
            secs[key] = round(time.time() - t0, 1)
            res[key] = r
            allok &= ok
            print("(%s) %s" % (key, ok), flush=True)
    with open(out, "w", newline="\n") as f:
        json.dump({"results": res, "seconds": secs, "_all_checks_pass": allok}, f, indent=1, sort_keys=True,
                  default=str)
    print("ALL CHECKS PASS:", allok)


if __name__ == "__main__":
    main()
