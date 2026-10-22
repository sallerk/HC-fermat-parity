"""
cyclic.py -- checks for ../NOTE.txt: the Hodge conjecture for powers of Jacobians of very general cyclic covers
y^p = prod_j (x - z_j)^{a_j} of the projective line of prime degree p, with arbitrary exponents.

 (A) Determinant lattice.  For p in {3, 5, 7, 11, 13} and N in {5, 6} branch points, every exponent class
     (a_j in (Z/p)^x, sum a_j = 0 mod p, up to permutation and scaling by units): the lattice L(p, a) of exponent
     vectors e (indexed by chi in (Z/p)^x / +-1) with sum_chi e_chi w(t chi) = 0 for all t, w(chi) = N - 2 sum_j
     <chi a_j / p>, has rank equal to the number of odd characters psi mod p with sum_j psi(a_j) = 0.  For a
     Z-basis of L (and all e with entries in {-1, 0, 1}) the joined Fermat tuple of the determinant class is a
     Hodge tuple of degree p and is decomposable (a union of pairs b, -b), i.e. its class is a linear-cycle class.
 (B) Monodromy certificate.  For p in {5, 7, 11, 13}, N in {5, 6}, every exponent class (210 classes): the pure
     braid group acting on the eigenspace H_chi (Fox calculus on the free group, rank-one local system
     x_j -> zeta^{chi a_j}, V = ker(d) / <boundary loop>, computed in F_l, l = 1 mod p) has End(V) = scalars and
     dim End(V (x) V) = 2 (the fourth moment), every full twist A_ij acts as a pseudo-reflection with eigenvalue
     (t_i t_j)^{+-1} (or unipotently if a_i + a_j = 0), and some eigenspace has indefinite signature.  Reduction
     mod l can only enlarge commutants, so these are certificates for the complex monodromy.
 (C) Rigidity over finite fields.  For five exponent vectors and random z_j in F_q (q = 1 mod p), with
     c = prod_j G'(z_j)^{a_j}: det(Frob | H_chi) * q / prod_j g(chi^{a_j}) / chi(c) is the same root of unity for
     all z (floating point, tolerance 1e-5), while without the twist chi(c) it varies.
 (D) The certificate of (B), with ranks computed by python-flint (nmod_mat), for N = 5 and every prime
     17 <= p <= 43, N = 6 and p in {17, 19}, N = 7 and p in {5, 7, 11}, N = 8 and p in {5, 7} (the base case and
     direct evidence for the induction of NOTE Proposition 3').
 (E) Lie algebras: for non-proportional delta_1, delta_2 (random, and orthogonal) in C^n with a hermitian form of
     each signature, n = 3..6, sl(delta_1^perp) + 0 and sl(delta_2^perp) + 0 generate sl_n (numpy, rank tolerance
     1e-8).
Usage:  python -I cyclic.py OUT.json         Expected last line "ALL CHECKS PASS: True".  About 25 minutes.
"""

import cmath
import itertools
import json
import os
import random
import sys
import time

import flint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ts-powers", "code"))
from v2_exotic import integer_kernel, evec, is_hodge  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# (A) the determinant lattice

def w(p, a, c):
    return len(a) - 2 * sum((c * x) % p for x in a) // p


def det_lattice(p, a):
    K = [c for c in range(1, p) if 2 * c < p]
    rows = [[w(p, a, (t * c) % p) for c in K] for t in range(1, p)]
    return K, integer_kernel(rows, len(K))


def odd_zero_sums(p, a):
    g = next(g for g in range(2, p) if all(pow(g, k, p) != 1 for k in range(1, p - 1))) if p > 2 else 1
    dlog = {pow(g, k, p): k for k in range(p - 1)}
    return sum(1 for m in range(1, p - 1, 2)
               if abs(sum(cmath.exp(2j * cmath.pi * m * dlog[x % p] / (p - 1)) for x in a)) < 1e-9)


def canon(p, a):
    return min(tuple(sorted((u * x) % p for x in a)) for u in range(1, p))


def classes(p, N):
    seen = []
    for a in itertools.combinations_with_replacement(range(1, p), N):
        if sum(a) % p == 0:
            c = canon(p, a)
            if c not in seen:
                seen.append(c)
    return seen


def joined_tuple(p, a, K, e):
    tup = []
    for ei, c in zip(e, K):
        for _ in range(abs(ei)):
            tup += [((1 if ei > 0 else -1) * c * x) % p for x in a]
    return tup


def check_A():
    rows, ok, exotic = [], True, []
    for p in (3, 5, 7, 11, 13):
        for N in (5, 6):
            for a in classes(p, N):
                K, L = det_lattice(p, a)
                z = odd_zero_sums(p, a)
                ok &= len(L) == z
                if L:
                    es = list(L) + [e for e in itertools.product((-1, 0, 1), repeat=len(K)) if any(e) and
                                    all(sum(ei * w(p, a, (t * c) % p) for ei, c in zip(e, K)) == 0
                                        for t in range(1, p))]
                    dec = all(is_hodge(tuple(joined_tuple(p, a, K, e)), p) and
                              not any(evec(joined_tuple(p, a, K, e), p)) for e in es)
                    ok &= dec
                    exotic.append({"p": p, "a": a, "rank_L": len(L), "basis": L, "decomposable": dec})
            rows.append({"p": p, "N": N, "classes": len(classes(p, N)),
                         "with_exotic": sum(1 for r in exotic if r["p"] == p and len(r["a"]) == N)})
    return {"counts": rows, "exotic": exotic}, ok


# ---------------------------------------------------------------------------------------------------------------
# (B) monodromy via Fox calculus, mod l

def artin(i, sign):
    if sign == 1:
        return {i: [(i, 1), (i + 1, 1), (i, -1)], i + 1: [(i, 1)]}
    return {i: [(i + 1, 1)], i + 1: [(i + 1, -1), (i, 1), (i + 1, 1)]}


def freereduce(wd):
    st = []
    for x in wd:
        if st and st[-1][0] == x[0] and st[-1][1] == -x[1]:
            st.pop()
        else:
            st.append(x)
    return st


def substitute(sub, word):
    out = []
    for g, e in word:
        img = sub.get(g, [(g, 1)])
        out += img if e == 1 else [(h, -f) for h, f in reversed(img)]
    return freereduce(out)


def pure_twist(N, i, j):
    """images of the free generators under the full twist A_ij = s_{j-1}..s_{i+1} s_i^2 s_{i+1}^-1..s_{j-1}^-1."""
    letters = [(k, 1) for k in range(j - 1, i, -1)] + [(i, 1), (i, 1)] + [(k, -1) for k in range(i + 1, j)]
    imgs = {k: [(k, 1)] for k in range(N)}
    for (k, s) in letters:
        sub = artin(k, s)
        imgs = {m: substitute(sub, wd) for m, wd in imgs.items()}
    return imgs


def fox_row(word, t, l, N):
    row, pref = [0] * N, 1
    for g, e in word:
        if e == 1:
            row[g] = (row[g] + pref) % l
            pref = pref * t[g] % l
        else:
            pref = pref * pow(t[g], l - 2, l) % l
            row[g] = (row[g] - pref) % l
    return row


def rref(M, l, ncols):
    M = [r[:] for r in M]
    rk, piv = 0, []
    for c in range(ncols):
        pr = next((r for r in range(rk, len(M)) if M[r][c] % l), None)
        if pr is None:
            continue
        M[rk], M[pr] = M[pr], M[rk]
        inv = pow(M[rk][c], l - 2, l)
        M[rk] = [x * inv % l for x in M[rk]]
        for r in range(len(M)):
            if r != rk and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % l for x, y in zip(M[r], M[rk])]
        piv.append(c)
        rk += 1
    return rk, M, piv


def nullspace(M, l, ncols):
    rk, R, piv = rref(M, l, ncols)
    basis = []
    for f in (c for c in range(ncols) if c not in piv):
        v = [0] * ncols
        v[f] = 1
        for r, pc in enumerate(piv):
            v[pc] = (-R[r][f]) % l
        basis.append(v)
    return basis


def eigenspace_monodromy(p, a, chi, l):
    """matrices of the full twists on V = ker(d)/<v_inf> (dimension N - 2) over F_l, and t_i t_j."""
    N = len(a)
    g = next(x for x in range(2, l) if pow(x, (l - 1) // p, l) != 1)
    zeta = pow(g, (l - 1) // p, l)
    t = [pow(zeta, (chi * x) % p, l) for x in a]
    ker = nullspace([[(tj - 1) % l for tj in t]], l, N)
    vinf = fox_row([(j, 1) for j in range(N)], t, l, N)
    basis = []
    for v in [vinf] + ker:
        if rref(basis + [v], l, N)[0] > len(basis):
            basis.append(v)
    mats, lams = [], []
    for i, j in itertools.combinations(range(N), 2):
        imgs = pure_twist(N, i, j)
        J = [fox_row(imgs[k], t, l, N) for k in range(N)]
        C = []
        for r in basis:
            img = [sum(r[k] * J[k][m] for k in range(N)) % l for m in range(N)]
            Mt = [[basis[k][m] for k in range(N - 1)] + [img[m]] for m in range(N)]
            rk, R, piv = rref(Mt, l, N)
            assert (N - 1) not in piv
            sol = [0] * (N - 1)
            for rr, pc in enumerate(piv):
                sol[pc] = R[rr][N - 1]
            C.append(sol)
        mats.append([row[1:] for row in C[1:]])
        lams.append(t[i] * t[j] % l)
    return mats, lams


def commutant_dim(mats, l, n):
    eqs = []
    for M in mats:
        for r in range(n):
            for c in range(n):
                row = [0] * (n * n)
                for k in range(n):
                    row[r * n + k] = (row[r * n + k] + M[k][c]) % l
                    row[k * n + c] = (row[k * n + c] - M[r][k]) % l
                eqs.append(row)
    return n * n - rref(eqs, l, n * n)[0]


def kron(A, B, l):
    return [[A[i][j] * B[k][m] % l for j in range(len(A)) for m in range(len(B))]
            for i in range(len(A)) for k in range(len(B))]


def prime_1_mod(p):
    q = 2 * p + 1
    while not (q % p == 1 and all(q % s for s in range(2, int(q ** 0.5) + 1))):
        q += 1
    return q


def certify(p, a):
    l = prime_1_mod(p)
    n = len(a) - 2
    mats, lams = eigenspace_monodromy(p, a, 1, l)
    e1 = commutant_dim(mats, l, n)
    e2 = commutant_dim([kron(M, M, l) for M in mats], l, n * n)
    refl = True
    for M, lam in zip(mats, lams):
        rk = rref([[(M[r][c] - (r == c)) % l for c in range(n)] for r in range(n)], l, n)[0]
        ev = (sum(M[r][r] for r in range(n)) - (n - 1)) % l
        refl &= (rk <= 1) if lam == 1 else (rk == 1 and ev in (lam, pow(lam, l - 2, l)))
    indef = any(1 <= sum((c * x) % p for x in a) // p - 1 <= n - 1 for c in range(1, p))
    return {"p": p, "a": a, "l": l, "End_V": e1, "End_VV": e2, "reflections": refl, "indefinite": indef,
            "certified": e1 == 1 and e2 == 2 and refl and indef}


def check_B():
    rows, ok = [], True
    for p in (5, 7, 11, 13):
        for N in (5, 6):
            for a in classes(p, N):
                r = certify(p, a)
                rows.append(r)
                ok &= r["certified"]
    return rows, ok


# ---------------------------------------------------------------------------------------------------------------
# (C) Frobenius

def prim_root(q):
    fac = [r for r in range(2, q) if (q - 1) % r == 0 and all(r % s for s in range(2, r))]
    return next(g for g in range(2, q) if all(pow(g, (q - 1) // r, q) != 1 for r in fac))


def frob_data(p, a, q, z):
    n = len(a) - 2
    g0 = prim_root(q)
    dlog, x = {}, 1
    for e in range(q - 1):
        dlog[x] = e
        x = x * g0 % q
    zeta = cmath.exp(2j * cmath.pi / p)
    chi = lambda v, k: 0 if v % q == 0 else zeta ** ((k * (dlog[v % q] % p)) % p)  # noqa: E731
    P = flint.nmod_poly([1], q)
    for zj, aj in zip(z, a):
        P = P * flint.nmod_poly([-zj % q, 1], q) ** aj
    co = [int(c) for c in P.coeffs()]
    tot = {k: [] for k in range(1, p)}
    for m in range(1, n + 1):
        F = flint.fq_default_ctx(q, m)
        cs = [F(c) for c in co]
        acc = {k: 1 + 0j for k in range(1, p)}        # infinity is unramified: p rational points, chi(1) = 1
        for coeffs in itertools.product(range(q), repeat=m):
            xx = F(list(coeffs))
            v = cs[-1]
            for c in reversed(cs[:-1]):
                v = v * xx + c
            if v.is_zero():
                continue
            ll = dlog[(int(v.norm()) if m > 1 else int(v)) % q] % p
            for k in range(1, p):
                acc[k] += zeta ** ((k * ll) % p)
        for k in range(1, p):
            tot[k].append(acc[k])
    det = {}
    for k in range(1, p):
        ps = [-s for s in tot[k]]
        e = [1]
        for r in range(1, n + 1):
            e.append(sum((-1) ** (i - 1) * e[r - i] * ps[i - 1] for i in range(1, r + 1)) / r)
        det[k] = e[n]
    gauss = {k: sum(chi(s, k) * cmath.exp(2j * cmath.pi * s / q) for s in range(1, q)) for k in range(1, p)}
    c = 1
    for j, (zj, aj) in enumerate(zip(z, a)):
        gp = 1
        for k, zk in enumerate(z):
            if k != j:
                gp = gp * (zj - zk) % q
        c = c * pow(gp, aj, q) % q
    return det, gauss, c, chi


def rou(zv, n, tol=1e-5):
    return next((j for j in range(n) if abs(zv - cmath.exp(2j * cmath.pi * j / n)) < tol), None)


def check_C():
    rows, ok = [], True
    for p, a, q, trials in ((7, (1, 1, 2, 4, 6), 29, 5), (11, (1, 3, 4, 5, 9), 23, 4), (5, (1, 1, 2, 3, 3), 11, 5),
                            (5, (1, 1, 2, 3, 4, 4), 11, 4), (13, (1, 1, 3, 9, 12), 53, 3)):
        rng = random.Random(1000 * p + q)
        tw = {k: set() for k in range(1, p)}
        un = {k: set() for k in range(1, p)}
        for _ in range(trials):
            z = rng.sample(range(q), len(a))
            det, gauss, c, chi = frob_data(p, a, q, z)
            for k in range(1, p):
                G = 1
                for aj in a:
                    G *= gauss[(k * aj) % p]
                base = det[k] * q / G
                un[k].add(rou(base, 2 * p))
                tw[k].add(rou(base / chi(c, k), 2 * p))
        const = all(len(s) == 1 and None not in s for s in tw.values())
        varies = any(len(s) > 1 for s in un.values())
        rows.append({"p": p, "a": a, "q": q, "trials": trials, "twisted_constant": const, "untwisted_varies": varies})
        ok &= const and varies
        print("    (C) p=%d a=%s q=%d: twisted ratio constant %s, untwisted varies %s" % (p, a, q, const, varies),
              flush=True)
    return rows, ok


def commutant_dim_flint(mats, l, n):
    rows = []
    for M in mats:
        for r in range(n):
            for c in range(n):
                row = [0] * (n * n)
                for k in range(n):
                    row[r * n + k] = (row[r * n + k] + M[k][c]) % l
                    row[k * n + c] = (row[k * n + c] - M[r][k]) % l
                rows.append(row)
    return n * n - flint.nmod_mat(rows, l).rank()


def certify_flint(p, a):
    """as certify(), but the commutants are computed for growing subsets of the generators (the commutant of a
    subset contains that of the group, so dimensions 1 and 2 for a subset are certificates)."""
    l = prime_1_mod(p)
    n = len(a) - 2
    mats, lams = eigenspace_monodromy(p, a, 1, l)
    e1 = commutant_dim_flint(mats, l, n)
    e2 = None
    for k in range(3, len(mats) + 1):
        e2 = commutant_dim_flint([kron(M, M, l) for M in mats[:k]], l, n * n)
        if e2 == 2:
            break
    refl = True
    for M, lam in zip(mats, lams):
        rk = flint.nmod_mat([[(M[r][c] - (r == c)) % l for c in range(n)] for r in range(n)], l).rank()
        ev = (sum(M[r][r] for r in range(n)) - (n - 1)) % l
        refl &= (rk <= 1) if lam == 1 else (rk == 1 and ev in (lam, pow(lam, l - 2, l)))
    indef = any(1 <= sum((c * x) % p for x in a) // p - 1 <= n - 1 for c in range(1, p))
    return e1 == 1 and e2 == 2 and refl and indef


def check_D():
    rows, ok = [], True
    cases = [(p, 5) for p in (17, 19, 23, 29, 31, 37, 41, 43)] + [(17, 6), (19, 6), (5, 7), (7, 7), (11, 7),
                                                                  (5, 8), (7, 8)]
    for p, N in cases:
        t0 = time.time()
        cl = classes(p, N)
        bad = [a for a in cl if not certify_flint(p, a)]
        rows.append({"p": p, "N": N, "classes": len(cl), "failures": bad, "seconds": round(time.time() - t0, 1)})
        ok &= not bad
        print("    (D) p=%d N=%d: %d classes, %d certified" % (p, N, len(cl), len(cl) - len(bad)), flush=True)
    return rows, ok


def check_E():
    import numpy as np
    rng = np.random.default_rng(20261017)

    def subalg(delta, h):
        n = len(delta)
        f = h @ delta.conj()
        perp = np.linalg.svd(f.reshape(1, -1))[2][1:].conj()
        B = np.column_stack([delta] + list(perp))
        Binv = np.linalg.inv(B)
        out = []
        for i in range(1, n):
            for j in range(1, n):
                if i == j == 1:
                    continue
                E = np.zeros((n, n), complex)
                E[i, j] = 1
                if i == j:
                    E[1, 1] = -1
                out.append(B @ E @ Binv)
        return out

    def gen_dim(gens):
        basis = []

        def add(X):
            M = np.array([b.ravel() for b in basis] + [X.ravel()])
            if np.linalg.matrix_rank(M, tol=1e-8) > len(basis):
                basis.append(X)
                return True
            return False
        for g in gens:
            add(g)
        changed = True
        while changed:
            changed = False
            cur = list(basis)
            for A in cur:
                for Bm in cur:
                    changed |= add(A @ Bm - Bm @ A)
        return len(basis)
    rows, ok = [], True
    for n in (3, 4, 5, 6):
        for sig in range(1, n):
            h = np.diag([1.0] * sig + [-1.0] * (n - sig))
            d1 = rng.normal(size=n) + 1j * rng.normal(size=n)
            d2 = rng.normal(size=n) + 1j * rng.normal(size=n)
            d2o = d2 - (d2 @ h @ d1.conj()) / (d1 @ h @ d1.conj()) * d1
            for name, dd in (("generic", d2), ("orthogonal", d2o)):
                dim = gen_dim(subalg(d1, h) + subalg(dd, h))
                rows.append({"n": n, "sig": sig, "case": name, "dim": dim})
                ok &= dim == n * n - 1
    return rows, ok


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "cyclic.json"
    res, allok, secs = {}, True, {}
    for key, fn in (("A", check_A), ("B", check_B), ("C", check_C), ("D", check_D), ("E", check_E)):
        t0 = time.time()
        r, ok = fn()
        secs[key] = round(time.time() - t0, 1)
        res[key] = r
        allok &= ok
        print("(%s) %s" % (key, ok), flush=True)
        if key == "A":
            print("    classes / with exotic determinant classes:",
                  [(x["p"], x["N"], x["classes"], x["with_exotic"]) for x in r["counts"]])
            print("    examples:", [(x["p"], x["a"], x["rank_L"]) for x in r["exotic"] if len(x["a"]) == 5])
        if key == "B":
            print("    certified exponent classes: %d of %d" % (sum(x["certified"] for x in r), len(r)))
    with open(out, "w", newline="\n") as f:
        json.dump({"results": res, "seconds": secs, "_all_checks_pass": allok}, f, indent=1, sort_keys=True,
                  default=list)
    print("ALL CHECKS PASS:", allok)


if __name__ == "__main__":
    main()
