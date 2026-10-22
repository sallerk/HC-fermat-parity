"""
v2_exotic.py -- v = 2, every r: the exotic (determinant) Hodge classes; NOTE.txt Section 8, Theorem 8.10.

Setting.  G a squarefree binary form of degree d, Y_G = {w^d + G(u, v) = 0} (a plane curve), N = d - 2,
H^1(Y_G) (x) C = H_1 + ... + H_{d-1} (H_k: the pullback by w -> zeta w acts by zeta^k), dim H_k = N,
X^(r)_G = {G(x^1) + ... + G(x^r) = 0}.  K+ = {1 <= u < d/2}; L = {e in Z^{K+} : sum_u e_u w(t u) = 0 for all units
t}, w(k) = d - 2 (k mod d) (w(d/2) = 0), is the lattice of determinant characters trivial on the Hodge torus.
For e in L, gamma_e is the tuple with |e_u| entries sign(e_u) u.  Fermat tuples are multisets of nonzero residues
mod D; R is the free abelian group on them (Aoki's formal module), modulo the linear pairs (a, -a).

Checks.
 (A) Rigidity (Lemma 8.10.1) over finite fields: Lambda^N H_k(Y_G) is the Fermat line V(k, ..., k) (d entries)
     twisted by the discriminant.  For d = 4, 5, 6, three primes p = 1 mod d each, and eight random squarefree G
     (four split, four not): the Frobenius power sums on H_k from brute-force character sums over F_{p^m},
     m <= N; Newton's identities give det(F | H_k); then
        det(F | H_k) * p / ( g(chi^k)^d * chi^k(disc G) * (disc G / p) )
     is the same 2d-th root of unity for all eight G (g = Gauss sum over F_p, (./p) the Legendre symbol, i.e. the
     sign of the Frobenius permutation of the roots).  Without the twist the value depends on G, and the trace of
     F on H_k is not constant.  Also a direct point count of Y_G over F_p.  Floating point (complex), tolerance
     1e-6; only integers (exponents of roots of unity, counts) are stored.
 (B) Hodge-number bookkeeping.  (B1) Lambda^N H_u has the type (d-1-u, u-1) of V(u, ..., u) on the Fermat variety
     of dimension N (d <= 60).  (B2) For the domination Y_G^r x F^{r-2} --> X^(r)_G (F the Fermat variety of
     dimension r - 2): the piece of X^(r)_G on which block i scales by zeta^{c_i} (all c_i != 0, sum 0) has the
     Hodge numbers of (x)_i H_{-c_i} (x) V(c_1, ..., c_r); computed for G = u^d + v^d from Fermat characters of
     the 2r-variable Fermat variety, against the generic (h^{1,0}, h^{0,1})(H_k) = (d-1-k, k-1); r = 3, 4, d <= 9.
     (With V(-c) instead the numbers differ: also checked.)
 (C) The lattice L for 5 <= d <= 200: rank = sum_{e | d, 3 <= e < d} phi(e)/2; for a Z-basis, sum_u u e_u =
     (d/2) sum_u e_u as integers (so N sum_u u e_u = 0 mod d always, i.e. L' = L); gamma_e (sum_u e_u even) resp.
     gamma_e + (d/2) (d even, sum_u e_u odd) is a Hodge tuple.
 (D) Spans of known algebraic Fermat classes, for 5 <= d <= 150 with L != 0 (d != 6): A(D) = the span, in Z^{K+_D}
     (+ Z for the residue D/2), of the linear pair (D/2, D/2), of all Hodge tuples of length 4 of degree D
     (Lefschetz (1,1) on Fermat surfaces) and of Aoki's standard tuples of every degree D' | D (lifted;
     [MMRV, Def. 3.7, Prop. 3.7]; each is checked to be a Hodge tuple).  With L embedded by u -> (D/d) u:
       X-condition  2L in A(d)   (Theorem 8.10(a): all X^(r)_G; predicted for every d by Kubert-Aoki);
       J-condition  dL in A(D)   (Theorem 8.10(b): all powers of Y_G); D = d, and D = 2d for odd d <= 120 that fail
                                at D = d.  For the d that still fail: 2L in A(d) (the obstruction is 2-torsion)
                                and the 2-rank s of L / (L cap (1/d) A(d)).

Usage:  python -I v2_exotic.py OUT.json        Expected last line "ALL CHECKS PASS: True".  ~3 minutes.
"""

import cmath
import itertools
import json
import math
import random
import sys
import time

import flint
import numpy as np


# ---------------------------------------------------------------------------------------------------------------
# (A) rigidity over finite fields

def prim_root(p):
    fac = [q for q in range(2, p) if (p - 1) % q == 0 and all(q % s for s in range(2, q))]
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in fac):
            return g


def legendre(a, p):
    a %= p
    return 0 if a == 0 else (1 if pow(a, (p - 1) // 2, p) == 1 else -1)


def frobenius_data(d, p, co):
    """co: monic squarefree G(x, 1) over F_p, co[i] = coefficient of x^i.  Returns det and trace of F on H_k
    (complex), the discriminant, and the brute-force and predicted numbers of F_p-points of Y_G."""
    N = d - 2
    g0 = prim_root(p)
    dlog = {}
    x = 1
    for a in range(p - 1):
        dlog[x] = a
        x = x * g0 % p
    zeta = cmath.exp(2j * cmath.pi / d)
    chi = lambda a, k: 0 if a % p == 0 else zeta ** ((k * dlog[a % p]) % d)
    tot = {k: [] for k in range(1, d)}
    for m in range(1, N + 1):
        F = flint.fq_default_ctx(p, m)
        cs = [F(c) for c in co]
        acc = {k: chi((-1) ** m, k) for k in range(1, d)}        # the point (1 : 0): -G(1, 0) = -1, norm (-1)^m
        for coeffs in itertools.product(range(p), repeat=m):
            xx = F(list(coeffs))
            a = cs[d]
            for c in reversed(cs[:d]):
                a = a * xx + c
            a = -a
            if a.is_zero():
                continue
            nrm = int(a.norm()) if m > 1 else int(a)
            l = dlog[nrm % p]
            for k in range(1, d):
                acc[k] += zeta ** ((k * l) % d)
        for k in range(1, d):
            tot[k].append(acc[k])
    det, tr = {}, {}
    for k in range(1, d):
        ps = [-t for t in tot[k]]                                # power sums of the eigenvalues of F on H_k
        e = [1]
        for n in range(1, N + 1):
            e.append(sum((-1) ** (i - 1) * e[n - i] * ps[i - 1] for i in range(1, n + 1)) / n)
        det[k], tr[k] = e[N], ps[0]
    disc = int(flint.nmod_poly(co, p).discriminant())
    Gx = lambda xx: sum(c * pow(xx, i, p) for i, c in enumerate(co)) % p
    count = sum(1 for xx in range(p) for w in range(p) if (pow(w, d, p) + Gx(xx)) % p == 0)
    count += sum(1 for w in range(p) if (pow(w, d, p) + 1) % p == 0)
    predicted = p + 1 + sum(tot[k][0] for k in range(1, d))
    gauss = {k: sum(chi(t, k) * cmath.exp(2j * cmath.pi * t / p) for t in range(1, p)) for k in range(1, d)}
    return det, tr, disc, count, predicted, gauss, chi


def root_of_unity_exponent(z, n, tol=1e-6):
    for j in range(n):
        if abs(z - cmath.exp(2j * cmath.pi * j / n)) < tol:
            return j
    return None


def check_A():
    rng = random.Random(20261017)
    rows, ok = [], True
    for d, primes in [(4, [5, 13, 17]), (5, [11, 31, 41]), (6, [7, 13, 19])]:
        for p in primes:
            polys = []
            for _ in range(4):                                    # split: four random sets of d roots
                roots = rng.sample(range(p), d)
                P = flint.nmod_poly([1], p)
                for z in roots:
                    P = P * flint.nmod_poly([-z % p, 1], p)
                polys.append([int(c) for c in P.coeffs()])
            while len(polys) < 8:                                  # not split
                co = [rng.randrange(p) for _ in range(d)] + [1]
                P = flint.nmod_poly(co, p)
                if int(P.discriminant()) % p == 0:
                    continue
                if all(f[0].degree() == 1 for f in P.factor()[1]):
                    continue
                polys.append(co)
            twisted = {k: set() for k in range(1, d)}
            untwisted = {k: set() for k in range(1, d)}
            traces = {k: set() for k in range(1, d)}
            counts_ok = True
            types = []
            for co in polys:
                det, tr, disc, count, predicted, gauss, chi = frobenius_data(d, p, co)
                counts_ok &= abs(predicted - count) < 1e-6
                types.append(sorted(f[0].degree() for f in flint.nmod_poly(co, p).factor()[1]))
                for k in range(1, d):
                    base = det[k] * p / gauss[k] ** d
                    twisted[k].add(root_of_unity_exponent(base / (chi(disc, k) * legendre(disc, p)), 2 * d))
                    untwisted[k].add(root_of_unity_exponent(base, 2 * d))
                    traces[k].add((round(tr[k].real, 4), round(tr[k].imag, 4)))
            const = all(len(s) == 1 and None not in s for s in twisted.values())
            varies = any(len(s) > 1 for s in untwisted.values())
            row = {"d": d, "p": p, "factor_types": types, "point_counts_ok": counts_ok,
                   "twisted_ratio_exponents_mod_2d": {str(k): sorted(s) for k, s in twisted.items()},
                   "untwisted_values": {str(k): len(s) for k, s in untwisted.items()},
                   "distinct_traces": {str(k): len(s) for k, s in traces.items()},
                   "constant": const, "untwisted_varies": varies}
            rows.append(row)
            ok &= counts_ok and const
    ok &= any(r["untwisted_varies"] for r in rows)
    return rows, ok


# ---------------------------------------------------------------------------------------------------------------
# (B) Hodge-number bookkeeping

def check_B():
    ok = True
    for d in range(3, 61):
        N = d - 2
        for u in range(1, d):
            # holomorphic forms u^i w^j du / w^(d-1), i + j <= d - 3, have character zeta^(j+1): count them
            h10 = sum(1 for j in range(d - 2) for i in range(d - 2 - j) if (j + 1) % d == u)
            h01 = N - h10
            q_fermat = d * u // d - 1                             # q = sum <u/d> - 1 over d entries
            ok &= (h10, h01) == (d - 1 - u, u - 1) and (N - q_fermat, q_fermat) == (h10, h01)
    b1 = ok
    rows = []
    for d in range(3, 10):
        for r in (3, 4):
            for c in itertools.product(range(1, d), repeat=r):
                if sum(c) % d:
                    continue
                # X side: G = u^d + v^d, characters (a_i, b_i) with a_i + b_i = c_i mod d, all nonzero
                xpoly = [0] * (2 * r)
                per_block = []
                for ci in c:
                    per_block.append([(a + b) // d for a in range(1, d) for b in range(1, d) if (a + b) % d == ci])
                for combo in itertools.product(*per_block):
                    q = sum(ci + d * s for ci, s in zip(c, combo)) // d - 1
                    xpoly[q] += 1
                # A side: (x) H_{-c_i} (x) V_F(c) resp. V_F(-c)
                def apoly(sign):
                    poly = [1]
                    for ci in c:
                        k = (-ci) % d
                        f = [d - 1 - k, k - 1]
                        new = [0] * (len(poly) + 1)
                        for i, x in enumerate(poly):
                            for j, y in enumerate(f):
                                new[i + j] += x * y
                        poly = new
                    qf = sum((sign * ci) % d for ci in c) // d - 1
                    out = [0] * (2 * r)
                    for i, x in enumerate(poly):
                        out[i + qf] += x
                    return out
                good, bad = apoly(1) == xpoly, apoly(-1) == xpoly
                rows.append((d, r, good, bad))
    b2 = all(g for _, _, g, _ in rows)
    b2_wrong_sign_fails = any(not b for _, _, _, b in rows)
    return {"B1_types": b1, "B2_pieces": len(rows), "B2_ok": b2, "B2_opposite_sign_fails_somewhere":
            b2_wrong_sign_fails}, b1 and b2 and b2_wrong_sign_fails


# ---------------------------------------------------------------------------------------------------------------
# lattices

def units(d):
    return [t for t in range(1, d) if math.gcd(t, d) == 1]


def euler_phi(n):
    return len(units(n)) if n > 1 else 1


def integer_kernel(rows, n):
    """Z-basis of {e in Z^n : rows . e = 0}, via the Hermite normal form of [rows^T | I]."""
    m = len(rows)
    A = flint.fmpz_mat([[rows[i][j] for i in range(m)] + [1 if k == j else 0 for k in range(n)] for j in range(n)])
    H = A.hnf()
    out = []
    for i in range(n):
        row = [int(H[i, j]) for j in range(m + n)]
        if all(x == 0 for x in row[:m]) and any(row[m:]):
            out.append(row[m:])
    return out


def hnf_rows(vecs, n):
    if not vecs:
        return []
    H = flint.fmpz_mat(vecs).hnf()
    return [r for r in ([int(H[i, j]) for j in range(n)] for i in range(H.nrows())) if any(r)]


def in_span(H, v, n):
    return hnf_rows(H + [v], n) == H


def lattice_L(d):
    Kp = [u for u in range(1, d) if 2 * u < d]
    w = lambda k: (d - 2 * (k % d)) if (k % d) and 2 * (k % d) != d else 0
    rows = [[w(t * u) for u in Kp] for t in units(d)]
    return Kp, integer_kernel(rows, len(Kp))


def is_hodge(beta, D):
    L = len(beta)
    return L % 2 == 0 and sum(beta) % D == 0 and all(x % D for x in beta) and \
        all(sum((t * b) % D for b in beta) == (L // 2) * D for t in units(D))


def check_C():
    rows, ok = [], True
    for d in range(5, 201):
        if d == 6:
            continue
        Kp, Lb = lattice_L(d)
        rk = sum(euler_phi(e) // 2 for e in range(3, d) if d % e == 0)
        good = len(Lb) == rk
        for e in Lb:
            S, s1 = sum(u * x for u, x in zip(Kp, e)), sum(e)
            good &= 2 * S == d * s1
            gam = sum(([u if x > 0 else d - u] * abs(x) for u, x in zip(Kp, e)), [])
            if s1 % 2 == 0:
                good &= is_hodge(gam, d)
            else:
                good &= d % 2 == 0 and is_hodge(gam + [d // 2], d)
        rows.append({"d": d, "rank_L": len(Lb), "ok": good})
        ok &= good
    return rows, ok


# ---------------------------------------------------------------------------------------------------------------
# (D) spans of known algebraic Fermat classes

def hodge4(D):
    U = np.array(units(D), dtype=np.int64)
    out = []
    for a in range(1, D):
        b = np.arange(a, D, dtype=np.int64)
        B, C = np.meshgrid(b, b, indexing="ij")
        mask = C >= B
        B, C = B[mask], C[mask]
        E = (-(a + B + C)) % D
        keep = (E >= C) & (E != 0)
        B, C, E = B[keep], C[keep], E[keep]
        if len(B) == 0:
            continue
        good = np.ones(len(B), dtype=bool)
        for t in U:
            good &= ((t * a) % D + (t * B) % D + (t * C) % D + (t * E) % D) == 2 * D
            if not good.any():
                break
        out += [(a, int(x), int(y), int(z)) for x, y, z in zip(B[good], C[good], E[good])]
    return out


def is_prime(n):
    return n >= 2 and all(n % q for q in range(2, int(n ** 0.5) + 1))


def standard(D):
    """Aoki's standard tuples s_{p,i} of degree D = p k, k >= 3, p prime, p not dividing i [MMRV, Def. 3.7]
    (those with a zero entry, k | i, are left out)."""
    out = []
    for p in range(2, D + 1):
        if D % p or not is_prime(p) or D // p < 3:
            continue
        k = D // p
        for i in range(1, D):
            if i % p == 0:
                continue
            if p == 2:
                out.append((i, k + i, (D - 2 * i) % D, k))
            else:
                out.append(tuple((j * k + i) % D for j in range(p)) + ((D - p * i) % D,))
    return [s for s in out if all(x % D for x in s)]          # k | i gives zero entries: not a character


def evec(beta, D):
    Kp = [u for u in range(1, D) if 2 * u < D]
    idx = {u: i for i, u in enumerate(Kp)}
    v = [0] * (len(Kp) + (1 if D % 2 == 0 else 0))
    for b in beta:
        b %= D
        if 2 * b == D:
            v[-1] += 1
        elif 2 * b < D:
            v[idx[b]] += 1
        else:
            v[idx[D - b]] -= 1
    return v


SPANS = {}


def span_A(D):
    if D in SPANS:
        return SPANS[D]
    nD = len([u for u in range(1, D) if 2 * u < D]) + (1 if D % 2 == 0 else 0)
    gens = [evec(b, D) for b in hodge4(D)]
    std_ok, nstd = True, 0
    for Dp in range(3, D + 1):
        if D % Dp:
            continue
        for b in standard(Dp):
            std_ok &= is_hodge(b, Dp)
            gens.append(evec(tuple((D // Dp) * x for x in b), D))
            nstd += 1
    if D % 2 == 0:
        g = [0] * nD
        g[-1] = 2
        gens.append(g)
    SPANS[D] = (hnf_rows(gens, nD), nD, std_ok, nstd)
    return SPANS[D]


def lift(Kp, v, m, k, nD):
    out = [0] * nD
    for u, x in zip(Kp, v):
        out[k * u - 1] += m * x
    return out


def two_rank(Kp, Lb, d, H, nD):
    """s with [L : L cap (1/d) A] = 2^s (given 2 L in A)."""
    n = len(Lb)
    # integer kernel of [d*Lb ; -H] in the combined coordinates: x . (d Lb) = y . H
    M = [lift(Kp, v, d, 1, nD) for v in Lb] + [[-x for x in h] for h in H]
    ker = integer_kernel([list(col) for col in zip(*M)], len(M))
    sub = hnf_rows([[sum(k[i] * Lb[i][j] for i in range(n)) for j in range(len(Kp))] for k in ker], len(Kp))
    gram = lambda B: flint.fmpz_mat(B) * flint.fmpz_mat(B).transpose()
    ratio = int(gram(sub).det()) // int(gram(Lb).det())
    idx = math.isqrt(ratio)
    return round(math.log2(idx)) if idx * idx == ratio and idx & (idx - 1) == 0 else None


def check_D():
    rows, ok = [], True
    for d in range(5, 151):
        if d == 6:
            continue
        Kp, Lb = lattice_L(d)
        if not Lb:
            continue
        H, nD, std_ok, nstd = span_A(d)
        ok &= std_ok
        X = all(in_span(H, lift(Kp, v, 2, 1, nD), nD) for v in Lb)
        J = all(in_span(H, lift(Kp, v, d, 1, nD), nD) for v in Lb)
        row = {"d": d, "rank_L": len(Lb), "rank_A": len(H), "n_standard": nstd, "X": X, "J_at_d": J, "J_at_2d": None,
               "two_rank": None}
        if not J and d % 2 == 1 and d <= 120:
            H2, nD2, std_ok2, _ = span_A(2 * d)
            ok &= std_ok2
            row["J_at_2d"] = all(in_span(H2, lift(Kp, v, d, 2, nD2), nD2) for v in Lb)
        if not J and not row["J_at_2d"]:
            row["two_rank"] = two_rank(Kp, Lb, d, H, nD)
            ok &= row["two_rank"] is not None
        ok &= X and (J or d % 2 == 1)                             # X always; J for every even d
        rows.append(row)
    return rows, ok


# ---------------------------------------------------------------------------------------------------------------

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "v2_exotic.json"
    allok, seconds = True, {}
    t = time.time()
    ra, oka = check_A()
    seconds["A"] = round(time.time() - t, 1)
    allok &= oka
    print("(A) rigidity over F_p: det(F | H_k) p / (g(chi^k)^d chi^k(disc) (disc/p)) is one 2d-th root of unity")
    print("    for all eight G (four split, four not); without the twist it varies:")
    for r in ra:
        print("    d=%d p=%2d  factor types %s" % (r["d"], r["p"], r["factor_types"]))
        print("         exponents (mod 2d) %s  untwisted #values %s  #traces %s  counts %s  constant %s" % (
            r["twisted_ratio_exponents_mod_2d"], r["untwisted_values"], r["distinct_traces"], r["point_counts_ok"],
            r["constant"]))
    t = time.time()
    rb, okb = check_B()
    seconds["B"] = round(time.time() - t, 1)
    allok &= okb
    print("(B) Hodge types: Lambda^N H_u ~ V(u,...,u) for d <= 60: %s; pieces of X^(r) (r = 3, 4, d <= 9) ~"
          % rb["B1_types"])
    print("    (x) H_{-c_i} (x) V(c): %d pieces, %s; with V(-c) some piece differs: %s" % (
        rb["B2_pieces"], rb["B2_ok"], rb["B2_opposite_sign_fails_somewhere"]))
    t = time.time()
    rc, okc = check_C()
    seconds["C"] = round(time.time() - t, 1)
    allok &= okc
    print("(C) L for 5 <= d <= 200: rank formula, sum u e_u = (d/2) sum e_u, gamma_e (+ d/2) Hodge: %s" % okc)
    t = time.time()
    rd, okd = check_D()
    seconds["D"] = round(time.time() - t, 1)
    allok &= okd
    xf = [r["d"] for r in rd if not r["X"]]
    jd = [r["d"] for r in rd if r["J_at_d"]]
    j2 = [r["d"] for r in rd if r["J_at_2d"]]
    jf = [(r["d"], r["two_rank"]) for r in rd if not r["J_at_d"] and not r["J_at_2d"]]
    print("(D) spans of linear, Lefschetz (length 4) and lifted standard tuples, 5 <= d <= 150, L != 0:")
    print("    X-condition 2L in A(d) fails for: %s" % xf)
    print("    J-condition dL in A(d) holds for %d degrees; holds only at D = 2d for: %s" % (len(jd), j2))
    print("    J-condition fails (odd d; 2L in A, 2-rank s of the obstruction): %s" % jf)
    print("    odd d with the J-condition: %s" % [d for d in jd + j2 if d % 2])
    with open(out, "w", newline="\n") as f:
        json.dump({"A": ra, "B": rb, "C": rc, "D": rd, "seconds": seconds, "_all_checks_pass": allok}, f,
                  indent=1, sort_keys=True)
    print("ALL CHECKS PASS:", allok)


if __name__ == "__main__":
    main()
