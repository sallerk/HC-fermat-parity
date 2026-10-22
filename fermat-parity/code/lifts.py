"""
lifts.py -- checks for ../NOTE.txt, Section 7 (lifted lengths of Aoki's elements xi_q).

l_{2,d}(xi_q) = least length of an element alpha of R_d with xi_q - alpha in D_d + S_d ([MMRV] (29)), d a
multiple of q; entries d/2 are not counted ((d/2) lies in D_d).  It is computed exactly by an integer program
(scipy / HiGHS): alpha Hodge, and f(alpha) = f(xi_q) mod 2 for a basis of the GF(2)-functionals f that kill the
standard tuples of degree d (lifted ones included) and are independent on the Hodge lattice; these pin down the
class, since B_d / (D_d + S_d) is 2-torsion ([MMRV] Thm. 4.3).
 (Q) Theorem 18 (degree 2q, q = l p odd): nu(y + 1/2) = (2y) - (y) maps the lattice basis of B_{2q} to Hodge
     elements of degree q, and non-Hodge elements to non-Hodge ones (2q <= 210); B_{2q} / (D + S) = Z/2 by Smith
     normal form (2q <= 222); l_{2,2q}(xi_q) by two independent integer programs (the direct one, and the one on
     (A, B) and the orbit data of Theorem 18) for all odd q = l p <= 150; it is <= 6 exactly for 15, 21, 33, 35,
     39, 55, 57, and >= (l - 1)(p - 1)/8.
 (R) Theorem 19 (one more prime): the explicit witness families at levels r q (r prime not dividing q; r >= 5 for q
     odd) and 8p are valid (even, kill every standard tuple of the level, 1 on xi_q), cover every point at most
     once on average, and give the stated bound; it equals the integer program in every case listed.
 (S) l_{2,kq}(xi_q) for q = 44, 51, 52 and k <= 12, for the other q in Q up to 69 and k q <= 420, and the least
     lengths in the cosets xi_q + (known algebraic xi_q') + D_d + S_d at 10 levels.
 (T) The 20 entries of [MMRV] Table 3 equal the exact l_{2,d}(xi_q) at their degree d.
Every representative found is checked to be a Hodge tuple in the class of xi_q.
Usage:  python -I lifts.py OUT.json      Expected last line "ALL CHECKS PASS: True".  About 12 minutes (3 cores).
"""

import itertools
import json
import multiprocessing
import os
import random
import sys
import time
from fractions import Fraction
from math import gcd

import flint
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("cyclic-covers", "fermat-frontier", "ts-powers"):
    sys.path.insert(0, os.path.join(HERE, "..", "..", sub, "code"))
sys.path.insert(0, HERE)
import fermat_tools as FT  # noqa: E402
from v2_exotic import integer_kernel, standard  # noqa: E402
import parity as PA  # noqa: E402
import structure_lp as SL  # noqa: E402

MMRV_TABLE3 = {33: (66, 4), 35: (70, 6), 39: (78, 3), 44: (132, 7), 51: (102, 7), 52: (156, 7), 55: (110, 5),
               57: (114, 6), 65: (130, 10), 68: (204, 8), 69: (138, 9), 76: (228, 11), 77: (154, 12), 85: (170, 12),
               87: (174, 10), 91: (182, 17), 92: (276, 12), 93: (186, 10), 95: (190, 10), 111: (222, 10)}
TWO_PRIME_Q = [(4, 3), (3, 5), (4, 5), (3, 7), (4, 7), (3, 11), (5, 7), (3, 13), (4, 11), (3, 17), (4, 13), (5, 11),
               (3, 19), (5, 13), (4, 17), (3, 23)]


def units(n):
    return [t for t in range(1, n) if gcd(t, n) == 1]


def crt(a, m, b, n):
    return next(c for c in range(m * n) if c % m == a % m and c % n == b % n)


def xi_tuple(l, p):
    """a tuple of degree q = l p (l = 4 or an odd prime) in the class of xi_q mod D_q + S_q."""
    if l == 4:
        return SL.tuple_4p(p)[0], 4 * p
    q = l * p
    prim, part_l, part_p = PA.xi_odd(l, p)
    return PA.as_tuple(q, [(1, q, prim), (p, l, part_l), (l, p, part_p)]), q


# ---------------------------------------------------------------------------------------------------------------
# the exact integer program

def half(d):
    return [u for u in range(1, d) if 2 * u < d]


def vec(t, d, idx):
    v = [0] * len(idx)
    for a in t:
        a %= d
        if a == 0 or 2 * a == d:
            continue
        if 2 * a < d:
            v[idx[a]] += 1
        else:
            v[idx[d - a]] -= 1
    return v


def hodge_rows(d, Kp):
    w = lambda k: (d - 2 * (k % d)) if (k % d) and 2 * (k % d) != d else 0
    return [[w(t * u) for u in Kp] for t in units(d) if 2 * t < d]


def class_functionals(d):
    """GF(2) functionals on Z^{half(d)} that kill the standard tuples of degree d (lifted ones included),
    independent on the Hodge lattice L_d."""
    Kp = half(d)
    idx = {u: i for i, u in enumerate(Kp)}
    S = [vec(tuple((d // e) * x for x in s), d, idx) for e in range(3, d + 1) if d % e == 0 for s in standard(e)]
    rows = hodge_rows(d, Kp)
    Lb = integer_kernel(rows, len(Kp))
    if not S:
        return Kp, idx, rows, [], Lb
    M = flint.nmod_mat([[x % 2 for x in s] for s in S], 2)
    N, nul = M.nullspace()
    F = [[int(N[i, j]) for i in range(len(Kp))] for j in range(nul)]
    keep, rk = [], 0
    for j in range(len(F)):
        m = flint.nmod_mat([[sum(F[c][i] * e[i] for i in range(len(Kp))) % 2 for e in Lb] for c in keep + [j]], 2)
        if m.rank() > rk:
            keep, rk = keep + [j], rk + 1
    return Kp, idx, rows, [F[j] for j in keep], Lb


def least(d, target, time_limit=3000):
    """(least length, a representative, number of class functionals, solver status, verified)."""
    Kp, idx, rows, F, Lb = class_functionals(d)
    n, k = len(Kp), len(F)
    xi = vec(target, d, idx)
    A, lo, hi = [], [], []
    for r in rows:
        A.append(list(r) + [-x for x in r] + [0] * k)
        lo.append(0)
        hi.append(0)
    for j, f in enumerate(F):
        rhs = sum(f[i] * xi[i] for i in range(n))
        A.append(list(f) + [-x for x in f] + [-2 if jj == j else 0 for jj in range(k)])
        lo.append(rhs)
        hi.append(rhs)
    cons = LinearConstraint(np.array(A, dtype=float), np.array(lo, dtype=float), np.array(hi, dtype=float))
    lb = np.concatenate([np.zeros(2 * n), -1e6 * np.ones(k)])
    ub = np.concatenate([np.full(2 * n, 60.0), 1e6 * np.ones(k)])
    res = milp(c=np.concatenate([np.ones(2 * n), np.zeros(k)]), constraints=[cons], integrality=np.ones(2 * n + k),
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None, None, k, res.status, False
    x = np.round(res.x).astype(int)
    t = []
    for i, u in enumerate(Kp):
        t += [u] * int(x[i]) + [d - u] * int(x[n + i])
    v = vec(t, d, idx)
    tt = tuple(t) + ((d // 2,) if (sum(t) % d) else ())
    ok = FT.is_hodge(tt, d) if t else True
    ok &= all(sum(f[i] * (v[i] - xi[i]) for i in range(n)) % 2 == 0 for f in F)
    return len(t), tuple(sorted(t)), k, res.status, ok


def lifted(l, p, k, extra=()):
    t, q = xi_tuple(l, p)
    d = k * q
    target = tuple(k * a for a in t)
    for (l2, p2) in extra:
        t2, q2 = xi_tuple(l2, p2)
        target += tuple((d // q2) * a for a in t2)
    return d, target


def least_job(args):
    l, p, k, extra = args
    d, target = lifted(l, p, k, extra)
    t0 = time.time()
    L, rep, nf, st, ok = least(d, target)
    return {"q": l * p, "k": k, "d": d, "extra": [a * b for a, b in extra], "least": L, "rep": rep,
            "functionals": nf, "status": int(st), "verified": bool(ok and st == 0),
            "sec": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------------------------
# (Q) degree 2q

def nu(t, q):
    """the map nu of Theorem 18 on a tuple of degree 2q: returns a dict residue mod q -> coefficient."""
    out = {}
    for a in t:
        a %= 2 * q
        if a == 0 or a == q:
            continue
        if a % 2 == 0:
            out[(a // 2) % q] = out.get((a // 2) % q, 0) + 1
        else:
            y = ((a - q) // 2) % q          # a/(2q) = y/q + 1/2
            out[(2 * y) % q] = out.get((2 * y) % q, 0) + 1
            out[y] = out.get(y, 0) - 1
    return out


def hodge_combo(c, q):
    """is the integer combination c (residue -> coefficient) of degree q Hodge?"""
    return all(sum(n * (((t * a) % q) * 2 - q) for a, n in c.items() if a % q) == 0 for t in units(q))


def check_nu(q):
    d = 2 * q
    Kp = half(d)
    Lb = integer_kernel(hodge_rows(d, Kp), len(Kp))
    ok = True
    for e in Lb:
        t = []
        for u, x in zip(Kp, e):
            t += [u if x > 0 else d - u] * abs(x)
        ok &= hodge_combo(nu(t, q), q)
    rng = random.Random(q)
    for _ in range(20):
        t = [rng.randrange(1, d) for _ in range(6)]
        beta_hodge = all(sum(2 * ((s * a) % d) - d for a in t) == 0 for s in units(d))
        ok &= beta_hodge == hodge_combo(nu(t, q), q)
    return ok


def snf_job(d):
    import composite as CO
    import obstruction as OB
    CO.base_span = lambda D: FT.span(D, lef=False, cp=False)
    return d, OB.obstruction(d)[0]


def orbit_program(l, p, time_limit=3000):
    """second computation of l_{2,2q}: minimise the number of entries of orders q, 2q, p, 2p, l, 2l subject to
    F1 + F2 o 2^-1 - F2 = A + B (primitive), = A(l .) - A (order p), = B(p .) - B (order l), with A, B odd and in
    1/2 + Z (Theorems 12, 18; F1, F2 the odd parts of the two multiplicities)."""
    q = l * p
    i2 = {m: pow(2, -1, m) for m in (q, p, l)}
    cols = []                          # variables: entries (class, residue, z) as nonnegative integers
    for m in (q, p, l):
        for u in units(m):
            for z in (0, 1):
                cols.append((m, u, z))
    nA, nB = (p - 1) // 2, (l - 1) // 2   # 2A(y) = 2 a_y + 1 for y in the half 1..(p-1)/2, odd extension
    col = {c: i for i, c in enumerate(cols)}
    nv = len(cols) + nA + nB

    def A2(y, row, coef):              # add coef * 2A(y) to row (2A(y) = 2 a_y + 1, odd)
        s = 1
        if 2 * y > p:
            y, s = p - y, -1
        row[len(cols) + y - 1] += 2 * coef * s
        return coef * s

    def B2(x, row, coef):
        s = 1
        if 2 * x > l:
            x, s = l - x, -1
        row[len(cols) + nA + x - 1] += 2 * coef * s
        return coef * s

    def lhs(m, u, row):                # 2 * (F1 + F2 o 2^-1 - F2)(u) on class m
        for z, uu, c in ((0, u, 2), (0, (-u) % m, -2), (1, (u * i2[m]) % m, 2), (1, (-u * i2[m]) % m, -2),
                         (1, u, -2), (1, (-u) % m, 2)):
            row[col[(m, uu, z)]] += c
    A, rhs = [], []
    for x in range(1, l):
        for y in range(1, p):
            row = [0] * nv
            lhs(q, crt(x, l, y, p), row)
            const = A2(y, row, -1) + B2(x, row, -1)
            A.append(row)
            rhs.append(-const)
    for y in range(1, p):
        row = [0] * nv
        lhs(p, y, row)
        const = A2((l * y) % p, row, -1) + A2(y, row, 1)
        A.append(row)
        rhs.append(-const)
    for x in range(1, l):
        row = [0] * nv
        lhs(l, x, row)
        const = B2((p * x) % l, row, -1) + B2(x, row, 1)
        A.append(row)
        rhs.append(-const)
    c = np.concatenate([np.ones(len(cols)), np.zeros(nA + nB)])
    lb = np.concatenate([np.zeros(len(cols)), -50 * np.ones(nA + nB)])
    ub = np.concatenate([np.full(len(cols), 60.0), 50 * np.ones(nA + nB)])
    res = milp(c=c, constraints=[LinearConstraint(np.array(A, float), np.array(rhs, float), np.array(rhs, float))],
               integrality=np.ones(nv), bounds=Bounds(lb, ub), options={"time_limit": time_limit, "disp": False})
    return (int(round(res.fun)) if res.x is not None else None), int(res.status)


def q_job(lp):
    l, p = lp
    r = least_job((l, p, 2, ()))
    r["orbit_program"], r["orbit_status"] = orbit_program(l, p)
    r["lower_bound"] = (l - 1) * (p - 1) / 8
    return r


def check_Q(pool):
    rows, ok = [], True
    odd = sorted([(l, p) for l in (3, 5, 7, 11) for p in range(l + 2, 60) if SL.isprime(l) and SL.isprime(p)
                  and l * p <= 150], key=lambda x: x[0] * x[1])
    nu_ok = all(pool.map(check_nu, [l * p for l, p in odd if l * p <= 105]))
    ok &= nu_ok
    snf = dict(pool.map(snf_job, [2 * l * p for l, p in odd if 2 * l * p <= 222]))
    ok &= all(v == [2] for v in snf.values())
    small = set()
    for r in pool.map(q_job, odd):
        good = r["verified"] and r["least"] == r["orbit_program"] and r["least"] >= r["lower_bound"]
        ok &= good
        if r["least"] <= 6:
            small.add(r["q"])
        rows.append(r)
    ok &= small == {15, 21, 33, 35, 39, 55, 57}
    print("    (Q) nu on bases: %s; B_2q / (D + S) = Z/2 for %d degrees; l_{2,2q} for %d odd q <= 150, two "
          "programs agree; <= 6 exactly for %s: %s" % (nu_ok, len(snf), len(rows), sorted(small), ok), flush=True)
    return {"nu": nu_ok, "snf": snf, "table": rows}, ok


# ---------------------------------------------------------------------------------------------------------------
# (R) one more prime: witness families

def orbits(m, gens):
    seen, out = set(), []
    for r0 in units(m):
        if r0 in seen:
            continue
        orb, stack = {r0}, [r0]
        while stack:
            x = stack.pop()
            for g in gens:
                y = g * x % m
                if y not in orb:
                    orb.add(y)
                    stack.append(y)
        seen |= orb
        out.append(orb)
    return out


def family_odd_prime(l, p, r):
    """witnesses at level r q, r an odd prime not dividing q = l p (l = 4 or an odd prime): the coset witness
    (1_K on order q; Psi at r-part w0, -Psi at -w0 on order r q) and the r-stable orbit and order-4 witnesses."""
    q, d = l * p, l * p * r
    K = {crt(sx, l, sy, p) for sx in (1, -1) for sy in (1, -1)}
    s, t = pow(r, -1, l), pow(r, -1, p)
    Psi = {}
    for (a, b) in ((1, 1), (1, -1), (s, t), (-s, t), (s, 1), (-s, -1)):
        u = crt(a, l, b, p)
        Psi[u] = Psi.get(u, 0) ^ 1
    Psi = {u for u, v in Psi.items() if v}
    ir, iq = pow(r, -1, q), pow(q, -1, r)
    split = lambda a: ((a * ir) % q, (a * iq) % r)

    def coset(a):
        U, Z = split(a % d)
        if gcd(U, q) != 1:
            return 0
        if Z == 0:
            return int(U in K)
        return int((Z == 1 and U in Psi) or (Z == r - 1 and (q - U) % q in Psi))
    fam = [coset]
    l0 = 2 if l == 4 else l
    if SL.minus_one_in(l0, p) and all(all((r * c) % p in C for c in C) for C in orbits(p, [l0 % p])):
        C = orbits(p, [l0 % p])[0]
        fam.append(lambda a, C=C: int(split(a % d)[1] == 0 and split(a % d)[0] % l == 0
                                      and (split(a % d)[0] % p) in C))
    if l != 4 and SL.minus_one_in(p, l) and all(all((r * c) % l in C for c in C) for C in orbits(l, [p % l])):
        C = orbits(l, [p % l])[0]
        fam.append(lambda a, C=C: int(split(a % d)[1] == 0 and split(a % d)[0] % p == 0
                                      and (split(a % d)[0] % l) in C))
    if l == 4 and p % 4 == 3:
        fam.append(lambda a: int(split(a % d)[1] == 0 and split(a % d)[0] % p == 0 and split(a % d)[0] % 2 == 1))
    return d, fam


def family_8p(p):
    """witnesses at level 8p for q = 4p."""
    d = 8 * p
    i8, ip = pow(p, -1, 8), pow(8, -1, p)
    parts = lambda a: ((a * i8) % 8, (a * ip) % p)

    def coset(a):
        X, Y = parts(a % d)
        if Y == 0:
            return 0
        if X % 4 == 2:
            return int(Y in (1, p - 1))
        return int(X % 2 == 1 and X in (1, 7) and (2 * Y) % p in (1, p - 1))
    fam = [coset]
    if SL.minus_one_in(2, p):
        C = orbits(p, [2])[0]
        fam.append(lambda a, C=C: int(parts(a % d)[0] == 0 and parts(a % d)[1] in C))
    if p % 4 == 3:
        def z4(a):
            X, Y = parts(a % d)
            if Y == 0:
                return int(X in (2, 6, 1, 7))
            return int(p % 8 == 3 and (X, Y) in {(1, 1), (5, 1), (3, p - 1), (7, p - 1)})
        fam.append(z4)
    return d, fam


def witness_job(args):
    l, p, r = args
    q = l * p
    d, fam = family_8p(p) if (l == 4 and r == 2) else family_odd_prime(l, p, r)
    xt, _ = xi_tuple(l, p)
    lift = [(d // q) * a for a in xt]
    stds = [tuple((d // e) * x % d for x in s) for e in range(3, d + 1) if d % e == 0 for s in standard(e)]
    G = units(d)
    ginv = [pow(g, -1, d) for g in G]
    valid, bound = True, 0
    cov = [Fraction(0)] * d
    for f in fam:
        vals = [f(a) if a else 0 for a in range(d)]
        valid &= all(vals[a] == vals[d - a] for a in range(1, d)) and (d % 2 or vals[d // 2] == 0)
        valid &= sum(vals[a % d] for a in lift) % 2 == 1
        valid &= all(sum(vals[a] for a in s) % 2 == 0 for s in stds)
        old = frozenset(a for a in range(d) if vals[a] and a % (d // q) == 0)
        N = len({frozenset((g * a) % d for a in old) for g in G})
        bound += N
        for x in range(1, d):
            cov[x] += Fraction(N * sum(1 for gi in ginv if vals[(gi * x) % d]), len(G))
    mx = max(cov)
    L, rep, nf, st, ver = least(d, tuple(lift))
    return {"q": q, "r": r, "d": d, "valid": valid, "max_coverage": str(mx), "bound": bound, "least": L,
            "verified": ver, "tight": L == bound}


def check_R(pool):
    cases = [(3, 5, 7), (3, 7, 5), (3, 11, 5), (3, 11, 7), (3, 13, 5), (3, 13, 7), (3, 17, 5), (3, 17, 7),
             (3, 19, 5), (5, 7, 11), (5, 7, 13), (5, 11, 7), (4, 3, 5), (4, 3, 7), (4, 5, 3), (4, 7, 3), (4, 11, 3),
             (4, 11, 5), (4, 11, 7), (4, 13, 3), (4, 13, 5), (4, 17, 3), (4, 19, 3), (4, 29, 3), (4, 3, 2),
             (4, 5, 2), (4, 7, 2), (4, 11, 2), (4, 13, 2), (4, 17, 2), (4, 19, 2), (4, 23, 2), (4, 43, 2)]
    rows, ok = [], True
    for r in pool.map(witness_job, cases):
        ok &= r["valid"] and Fraction(r["max_coverage"]) <= 1 and r["verified"] and r["least"] >= r["bound"]
        rows.append(r)
    tight = sum(r["tight"] for r in rows)
    print("    (R) witness families valid, coverage <= 1, bound <= least, for %d levels (tight in %d): %s"
          % (len(rows), tight, ok), flush=True)
    return rows, ok


# ---------------------------------------------------------------------------------------------------------------
# (S), (T)

def check_S(pool):
    jobs = [(l, p, k, ()) for (l, p) in ((4, 11), (3, 17), (4, 13)) for k in range(1, 13)]
    jobs += [(l, p, k, ()) for (l, p) in TWO_PRIME_Q if l * p <= 69 and l * p not in (44, 51, 52)
             for k in range(1, 7) if k * l * p <= 420]
    mixed = [((4, 11), [(4, 3), (3, 11)], 3), ((4, 13), [(4, 3), (3, 13)], 3), ((3, 17), [(4, 3)], 4),
             ((4, 11), [(4, 5), (5, 11)], 5), ((3, 17), [(3, 5)], 5), ((4, 13), [(4, 5)], 5),
             ((4, 11), [(4, 7)], 7), ((3, 17), [(3, 7)], 7), ((4, 13), [(4, 7)], 7), ((3, 17), [(3, 5)], 10)]
    for (l, p), others, k in mixed:
        for n in range(1, len(others) + 1):
            for sub in itertools.combinations(others, n):
                jobs.append((l, p, k, tuple(sub)))
    rows = list(pool.imap(least_job, jobs))
    ok = all(r["verified"] for r in rows)
    open3 = {q: sorted({r["least"] for r in rows if r["q"] == q and not r["extra"]}) for q in (44, 51, 52)}
    ok &= open3 == {44: [7], 51: [7, 10], 52: [7]}
    ok &= all(r["least"] >= 7 for r in rows if r["q"] in (44, 51, 52))
    by51 = {r["k"]: r["least"] for r in rows if r["q"] == 51 and not r["extra"]}
    ok &= all(v == (7 if k % 2 == 0 else 10) for k, v in by51.items())
    mixed_ok = all(r["least"] >= min(s["least"] for s in rows if s["q"] == r["q"] and s["d"] == r["d"]
                                    and not s["extra"]) for r in rows if r["extra"])
    ok &= mixed_ok
    print("    (S) %d integer programs; 44, 52: 7 at every k <= 12; 51: 7 for even k, 10 for odd k; adding known "
          "algebraic xi_q' never helps: %s" % (len(rows), ok), flush=True)
    return rows, ok


def check_T(pool):
    jobs = []
    for q, (d, _) in sorted(MMRV_TABLE3.items()):
        l = 4 if q % 4 == 0 else next(r for r in range(3, q) if q % r == 0)
        jobs.append((l, q // l, d // q, ()))
    rows, ok = [], True
    for r in pool.imap(least_job, jobs):
        good = r["verified"] and r["least"] == MMRV_TABLE3[r["q"]][1]
        ok &= good
        r["mmrv"] = MMRV_TABLE3[r["q"]][1]
        rows.append(r)
    print("    (T) all %d entries of [MMRV] Table 3 are the exact l_{2,d}: %s" % (len(rows), ok), flush=True)
    return rows, ok


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "lifts.json"
    res, allok, secs = {}, True, {}
    with multiprocessing.Pool(3) as pool:
        for key, fn in (("Q", check_Q), ("R", check_R), ("S", check_S), ("T", check_T)):
            t0 = time.time()
            r, ok = fn(pool)
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
