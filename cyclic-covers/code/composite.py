"""
composite.py -- checks for ../NOTE.txt, Section 7: cyclic covers y^nu = prod_j (x - z_j)^{a_j} of composite degree
nu with all a_j units mod nu.

 (F) For each (nu, N) listed in CASES and every exponent class (up to permutation and scaling by units): the
     monodromy certificate (M_nu) -- for every Galois orbit of eigenspaces (representative chi = g | nu)
     End = scalars, fourth moment 2 (3 for the hyperelliptic eigenspace chi = nu/2 if n >= 4), pseudo-reflections,
     an indefinite member; Goursat separation by full-twist eigenvalues, or, for the pairs those do not separate,
     absolute irreducibility of H_chi (x) H_chi' and H_chi (x) H_chi'^dual (all mod l) -- and the Fermat status of
     the determinant lattice L(nu, a): "span" if every gamma_e of a basis lies in the base span (linear, lifted
     standard, Lefschetz, curve pairs) at degree nu or 2 nu, "equivalent" if some gamma_e does not and the base
     span plus it contains B_nu.
 (G) Lemma 1 over F_31 for nu = 15, a = (1, 1, 1, 1, 11), all chi including the non-units.
 (H) The obstruction: for nu in {35, 51, 55, 57, 65, 69}, B_nu / (base span) = Z/2 (Smith normal form, odd nu, via
     B_nu / D_nu = L_nu), and whether the obstruction phi is the restriction of a Galois-invariant parity functional
     Phi_c(e) = sum_chi c(gcd(chi, nu)) e_chi mod 2 (yes for 35, 51, 57, 65, 69; no for 55).  Parity law samples:
     Fermat status of all (or 200 random) exponent classes with N = 3, ..., 7.
Usage:  python -I composite.py OUT.json      Expected last line "ALL CHECKS PASS: True".  About 75 minutes.
"""

import itertools
import json
import os
import random
import sys
import time
from math import gcd

import flint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "fermat-frontier", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ts-powers", "code"))
import cyclic as C  # noqa: E402
import fermat_tools as FT  # noqa: E402
from v2_exotic import lattice_L  # noqa: E402

CASES = [(15, 5), (21, 5), (33, 5), (39, 5), (35, 5), (51, 5), (55, 5), (57, 5),
         (20, 6), (28, 6), (51, 6), (44, 6), (52, 6), (57, 6)]
EXPECT = {(15, 5): "span", (21, 5): "span", (33, 5): "span", (39, 5): "span", (35, 5): "equivalent",
          (51, 5): "equivalent", (55, 5): "equivalent", (57, 5): "equivalent", (20, 6): "span", (28, 6): "span",
          (51, 6): "span", (44, 6): "span", (52, 6): "span", (57, 6): "span"}


def units(nu):
    return [u for u in range(1, nu) if gcd(u, nu) == 1]


def classes(nu, N):
    U = units(nu)
    seen, out = set(), []
    for a in itertools.combinations_with_replacement(U, N):
        if sum(a) % nu:
            continue
        c = min(tuple(sorted((u * x) % nu for x in a)) for u in U)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def prime_1_mod(nu):
    q = 2 * nu + 1
    while not (q % nu == 1 and all(q % s for s in range(2, int(q ** 0.5) + 1))):
        q += 1
    return q


def primitive_zeta(nu, l):
    fac = [r for r in range(2, l) if (l - 1) % r == 0 and all(r % s for s in range(2, r))]
    g = next(g for g in range(2, l) if all(pow(g, (l - 1) // r, l) != 1 for r in fac))
    return pow(g, (l - 1) // nu, l)


def monodromy(nu, a, chi, l, zeta):
    N = len(a)
    t = [pow(zeta, (chi * x) % nu, l) for x in a]
    ker = C.nullspace([[(tj - 1) % l for tj in t]], l, N)
    vinf = C.fox_row([(j, 1) for j in range(N)], t, l, N)
    basis = []
    for v in [vinf] + ker:
        if C.rref(basis + [v], l, N)[0] > len(basis):
            basis.append(v)
    mats, lams = [], []
    for i, j in itertools.combinations(range(N), 2):
        imgs = C.pure_twist(N, i, j)
        J = [C.fox_row(imgs[k], t, l, N) for k in range(N)]
        Cm = []
        for r in basis:
            img = [sum(r[k] * J[k][m] for k in range(N)) % l for m in range(N)]
            Mt = [[basis[k][m] for k in range(N - 1)] + [img[m]] for m in range(N)]
            rk, R, piv = C.rref(Mt, l, N)
            assert (N - 1) not in piv
            sol = [0] * (N - 1)
            for rr, pc in enumerate(piv):
                sol[pc] = R[rr][N - 1]
            Cm.append(sol)
        mats.append([row[1:] for row in Cm[1:]])
        lams.append(t[i] * t[j] % l)
    return mats, lams


def commutant_upto(gens, l, n, target):
    e = None
    for k in range(min(3, len(gens)), len(gens) + 1):
        e = C.commutant_dim_flint(gens[:k], l, n)
        if e == target:
            break
    return e


def certify(nu, a):
    n = len(a) - 2
    l = prime_1_mod(nu)
    zeta = primitive_zeta(nu, l)
    cache = {}

    def mats(c):
        if c not in cache:
            cache[c] = monodromy(nu, a, c, l, zeta)
        return cache[c]
    for g in (g for g in range(1, nu) if nu % g == 0):
        orbit = sorted({(g * u) % nu for u in range(1, nu // g) if gcd(u, nu // g) == 1})
        if not any(1 <= sum((c * x) % nu for x in a) // nu - 1 <= n - 1 for c in orbit):
            return False
        Ms, lams = mats(g)
        if C.commutant_dim_flint(Ms, l, n) != 1:
            return False
        target = 3 if 2 * g == nu and n >= 4 else 2      # Sp_2 = SL_2 has fourth moment 2
        if commutant_upto([C.kron(M, M, l) for M in Ms], l, n * n, target) != target:
            return False
        for M, lam in zip(Ms, lams):
            rk = flint.nmod_mat([[(M[r][c] - (r == c)) % l for c in range(n)] for r in range(n)], l).rank()
            ev = (sum(M[r][r] for r in range(n)) - (n - 1)) % l
            if not ((rk <= 1) if lam == 1 else (rk == 1 and ev in (lam, pow(lam, l - 2, l)))):
                return False
    S = {(a[i] + a[j]) % nu for i, j in itertools.combinations(range(len(a)), 2)}
    K = [c for c in range(1, nu) if 2 * c < nu]
    for c1, c2 in itertools.combinations(K, 2):
        if any({(c1 * s) % nu, (-c1 * s) % nu} != {(c2 * s) % nu, (-c2 * s) % nu} for s in S):
            continue
        for c2b in (c2, (-c2) % nu):
            gens = [C.kron(A, B, l) for A, B in zip(mats(c1)[0], mats(c2b)[0])]
            if commutant_upto(gens, l, n * n, 1) != 1:
                return False
    return True


def w(nu, a, c):
    return len(a) - 2 * sum((c * x) % nu for x in a) // nu


def det_lattice(nu, a):
    K = [c for c in range(1, nu) if 2 * c < nu]
    rows = [[w(nu, a, (t * c) % nu) for c in K] for t in units(nu)]
    return K, C.integer_kernel(rows, len(K))


def gamma(nu, a, K, e):
    tup = []
    for ei, c in zip(e, K):
        for _ in range(abs(ei)):
            tup += [((1 if ei > 0 else -1) * c * x) % nu for x in a]
    return tup


SPANS = {}


def base_span(D):
    if D not in SPANS:
        SPANS[D] = FT.span(D)
    return SPANS[D]


def fermat_status(nu, a):
    K, L = det_lattice(nu, a)
    if not L:
        return "no exotic", 0
    outside = []
    for e in L:
        g = gamma(nu, a, K, e)
        assert C.is_hodge(tuple(g), nu)
        H, n = base_span(nu)
        if FT.in_span(H, FT.evec(g, nu), n):
            continue
        H2, n2 = base_span(2 * nu)
        if FT.in_span(H2, FT.evec([2 * x for x in g], 2 * nu), n2):
            continue
        outside.append(e)
    if not outside:
        return "span", len(L)
    H, n = base_span(nu)
    Hx = FT.hnf_rows(H + [FT.evec(gamma(nu, a, K, outside[0]), nu)], n)
    return ("equivalent" if FT.fermat_condition(nu, nu, Hx, n) else "outside"), len(L)


def check_F():
    rows, ok = [], True
    for nu, N in CASES:
        t0 = time.time()
        stats = {}
        for a in classes(nu, N):
            key = "%s/%s" % (certify(nu, a), fermat_status(nu, a)[0])
            stats[key] = stats.get(key, 0) + 1
        good = stats == {"True/%s" % EXPECT[(nu, N)]: sum(stats.values())}
        ok &= good
        rows.append({"nu": nu, "N": N, "stats": stats, "as_expected": good, "seconds": round(time.time() - t0)})
        print("    (F) nu=%d N=%d %s %s" % (nu, N, stats, good), flush=True)
    return rows, ok


def check_G():
    p, a, q = 15, (1, 1, 1, 1, 11), 31
    rng = random.Random(5)
    tw = {k: set() for k in range(1, p)}
    un = {k: set() for k in range(1, p)}
    for _ in range(4):
        z = rng.sample(range(q), len(a))
        det, gauss, c, chi = C.frob_data(p, a, q, z)
        for k in range(1, p):
            G = 1
            for aj in a:
                G *= gauss[(k * aj) % p]
            base = det[k] * q / G
            un[k].add(C.rou(base, 2 * p))
            tw[k].add(C.rou(base / chi(c, k), 2 * p))
    const = all(len(s) == 1 and None not in s for s in tw.values())
    varies = any(len(s) > 1 for s in un.values())
    return {"twisted_constant": const, "untwisted_varies": varies}, const and varies


def obstruction_data(nu):
    Kp, Lb = lattice_L(nu)
    H, n = base_span(nu)
    Lvecs = []
    for e in Lb:
        v = [0] * n
        for u, x in zip(Kp, e):
            v[u - 1] += x
        Lvecs.append(v)
    r, cols = len(Lvecs), []
    for j in range(n):
        if flint.fmpq_mat([[v[k] for k in cols + [j]] for v in Lvecs]).rank() == len(cols) + 1:
            cols.append(j)
        if len(cols) == r:
            break
    A = flint.fmpq_mat([[v[k] for k in cols] for v in Lvecs]).transpose()
    coords = []
    for h in H:
        sol = A.solve(flint.fmpq_mat([[h[k]] for k in cols]))
        coords.append([int(sol[i, 0].p) for i in range(r)])
    snf = flint.fmpz_mat(coords).snf()
    divisors = [int(snf[i, i]) for i in range(min(snf.nrows(), snf.ncols())) if snf[i, i] not in (0, 1)]
    phis = [0 if FT.in_span(H, v, n) else 1 for v in Lvecs]
    divs = sorted({gcd(u, nu) for u in Kp})
    sols = [dict(zip(divs, c)) for c in itertools.product((0, 1), repeat=len(divs))
            if all(sum(dict(zip(divs, c))[gcd(u, nu)] * x for u, x in zip(Kp, e)) % 2 == ph
                   for e, ph in zip(Lb, phis))]
    return divisors, sols


def check_H():
    rows, ok = [], True
    rng = random.Random(4)
    for nu in (35, 51, 55, 57, 65, 69):
        divisors, sols = obstruction_data(nu)
        parity = bool(sols)
        ok &= divisors == [2] and parity == (nu != 55)
        samples = {}
        for N in (3, 4, 5, 6, 7):
            if N <= 6:
                cl = classes(nu, N)
            else:
                U, cl = units(nu), set()
                while len(cl) < 40:
                    a = [rng.choice(U) for _ in range(N - 1)]
                    if (-sum(a)) % nu in U:
                        cl.add(min(tuple(sorted((u * x) % nu for x in a + [(-sum(a)) % nu])) for u in U))
                cl = sorted(cl)
            if len(cl) > 200:
                cl = rng.sample(cl, 200)
            st = {}
            for a in cl:
                s = fermat_status(nu, a)[0]
                st[s] = st.get(s, 0) + 1
            samples[N] = st
            if parity:
                ok &= st == {("span" if N % 2 == 0 else "equivalent"): len(cl)}
        rows.append({"nu": nu, "quotient": divisors, "parity_functionals": sols, "samples": samples})
        print("    (H) nu=%d B/base = Z/%s, parity functional: %s, samples %s" % (
            nu, divisors, parity, samples), flush=True)
    return rows, ok


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "composite.json"
    res, allok, secs = {}, True, {}
    for key, fn in (("G", check_G), ("H", check_H), ("F", check_F)):
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
