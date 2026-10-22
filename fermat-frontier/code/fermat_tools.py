"""
fermat_tools.py -- spans of known algebraic classes in Aoki's formal module, curve-pair (CM isogeny) classes,
Markman-type classes on small CM factors, all Hodge classes of the powers of small CM factors, Weil classes for
CM fields K' (with twisted copies), the rigidity relations at CM binary forms, and the Galois (Kummer) characters of
Fermat Hodge classes.  Library for frontier.py; see ../NOTE.txt.

Conventions.  A tuple of degree D is a multiset of nonzero residues mod D; [tuple] is its element of the free group
R on (Q/Z) - {0}.  We work in R_D modulo the linear pairs [a] + [-a]: an element is a vector in Z^{K+_D} (+ one
coordinate for the residue D/2 when D is even), via evec() of ts-powers/code/v2_exotic.py.
"""

import cmath
import itertools
import math
import os
import sys

import numpy as np
import flint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ts-powers", "code"))
from v2_exotic import (evec, standard, hnf_rows, in_span, units, lattice_L, hodge4, is_hodge,  # noqa: E402
                       integer_kernel)


# ---------------------------------------------------------------------------------------------------------------
# spans

def nvec(D):
    return len([u for u in range(1, D) if 2 * u < D]) + (1 if D % 2 == 0 else 0)


def triples(D):
    """Fermat-curve characters (a, b, c), a <= b <= c, nonzero, a + b + c = 0 mod D."""
    out = []
    for a in range(1, D):
        for b in range(a, D):
            c = (-a - b) % D
            if c and c >= b:
                out.append((a, b, c))
    return out


def cm_type(t, D, U):
    """the Hodge type function s -> q(s t) in {0, 1} of the line V(t) of H^1 of the Fermat curve."""
    return tuple(((s * t[0]) % D + (s * t[1]) % D + (s * t[2]) % D) // D - 1 for s in U)


def curve_pair_generators(D):
    """[t'] - [t] for triples with identical type functions: V(t)^dual (x) V(t') is a Hodge class in H^1 (x) H^1 of
    two Fermat curves, i.e. a homomorphism between CM factors of the Jacobian, hence algebraic."""
    U = units(D)
    groups = {}
    for t in triples(D):
        groups.setdefault(cm_type(t, D, U), []).append(t)
    gens = []
    for ts in groups.values():
        v0 = evec(ts[0], D)
        for t in ts[1:]:
            gens.append([a - b for a, b in zip(evec(t, D), v0)])
    return gens


def standard_generators(D):
    gens, ok = [], True
    for Dp in range(3, D + 1):
        if D % Dp:
            continue
        for b in standard(Dp):
            ok &= is_hodge(b, Dp)
            gens.append(evec(tuple((D // Dp) * x for x in b), D))
    return gens, ok


def span(D, std=True, lef=True, cp=True, extra=(), extra_vecs=()):
    n = nvec(D)
    gens = []
    if D % 2 == 0:
        g = [0] * n
        g[-1] = 2
        gens.append(g)                                       # the linear pair (D/2, D/2)
    if std:
        gens += standard_generators(D)[0]
    if lef:
        gens += [evec(b, D) for b in hodge4(D)]
    if cp:
        gens += curve_pair_generators(D)
    for tup in extra:
        gens.append(evec(tup, D))
    gens += list(extra_vecs)
    return hnf_rows(gens, n), n


def fermat_condition(d, D, H, n):
    """B_d contained in the span (lifted to degree D)?  B_d / D_d = L (via e -> (e, sum e mod 2)), NOTE Sec. 1."""
    Kp, Lb = lattice_L(d)
    k = D // d
    ok = True
    for e in Lb:
        v = [0] * n
        for u, x in zip(Kp, e):
            v[k * u - 1] += x
        if d % 2 == 0 and sum(e) % 2:
            v[-1] += 1
        ok &= in_span(H, v, n)
    return ok


def j_condition(d, D, H, n):
    """d L contained in the span (HC for all powers of J(Y_G), ts-powers Theorem 8.10(b), (c))."""
    Kp, Lb = lattice_L(d)
    k = D // d
    for e in Lb:
        v = [0] * n
        for u, x in zip(Kp, e):
            v[k * u - 1] += d * x
        if not in_span(H, v, n):
            return False
    return True


def obstruction_representatives(d, H, n):
    """gamma_e for the L-basis vectors e (d odd) not in the span."""
    Kp, Lb = lattice_L(d)
    out = []
    for e in Lb:
        v = [0] * n
        for u, x in zip(Kp, e):
            v[u - 1] += x
        if not in_span(H, v, n):
            out.append(sorted(sum(([u if x > 0 else d - u] * abs(x) for u, x in zip(Kp, e)), [])))
    return out


# ---------------------------------------------------------------------------------------------------------------
# CM factors of Fermat curve Jacobians; Markman-type classes

def line_data(D):
    """for each distinct type function (= eigenline up to isogeny): a triple, the dimension of the simple CM factor
    (number of cosets of the stabilizer, halved), and the factor's isogeny class (orbit of the type function)."""
    U = units(D)
    idx = {s: i for i, s in enumerate(U)}
    mult = [[idx[(s * x) % D] for x in U] for s in U]
    lines = {}
    for t in triples(D):
        lines.setdefault(cm_type(t, D, U), t)
    info = {}
    for tau, t in lines.items():
        H = [h for h in range(len(U)) if all(tau[mult[h][x]] == tau[x] for x in range(len(U)))]
        delta = len(U) // (2 * len(H))
        orbit = min(tuple(tau[mult[s][x]] for x in range(len(U))) for s in range(len(U)))
        info[tau] = (t, delta, orbit, [U[h] for h in H])
    return U, info


def m4_generators(D, maxdim=4):
    """Sum of four distinct eigenlines lying in one abelian variety of dimension <= 4 (a product of CM factors),
    with sum of type functions = 2: a Hodge class on an abelian fourfold, algebraic by Markman's theorem
    (arXiv:2502.03415, Cor. 1.6.1); transported to the Fermat curves by the factor homomorphisms."""
    U, info = line_data(D)
    small = {tau: v for tau, v in info.items() if v[1] <= maxdim}
    classes = {}
    for tau, (t, delta, orb, H) in small.items():
        classes.setdefault(orb, (delta, []))[1].append(tau)
    cls = list(classes.items())
    n = len(U)
    gens = []
    for r in range(1, 5):
        for combo in itertools.combinations(range(len(cls)), r):
            if sum(cls[c][1][0] for c in combo) > maxdim:
                continue
            lines = [tau for c in combo for tau in cls[c][1][1]]
            for quad in itertools.combinations(lines, 4):
                if any(not any(tau in cls[c][1][1] for tau in quad) for c in combo):
                    continue
                if all(sum(tau[x] for tau in quad) == 2 for x in range(n)):
                    gens.append([x for tau in quad for x in small[tau][0]])
    return gens


def cm_factor_classes(D):
    """the isogeny classes of simple CM factors of J(C_D): [dimension, [(type function, triple) for every line]]."""
    U, info = line_data(D)
    cl = {}
    for t in triples(D):                          # every line of J(C_D); equal tau = isogenous factors
        tau = cm_type(t, D, U)
        cl.setdefault(info[tau][2], [info[tau][1], []])[1].append((tau, t))
    return U, sorted(cl.values(), key=lambda c: (c[0], c[1][0][1]))


def product_hodge_vectors(D, U, cl, combo):
    """all Hodge classes of all powers of the product of the classes in combo, as vectors in R_D / (linear pairs):
    the integer kernel of n -> (sum_t n_t (2 tau_t(s) - 1))_s over the lines t, mapped to sum n_t [t]."""
    n = nvec(D)
    lines = [lt for i in combo for lt in cl[i][1]]
    rows = [[2 * tau[x] - 1 for tau, t in lines] for x in range(len(U))]
    gens = []
    for k in integer_kernel(rows, len(lines)):
        v = [0] * n
        for coef, (tau, t) in zip(k, lines):
            if coef:
                v = [a + coef * b for a, b in zip(v, evec(t, D))]
        if any(v):
            gens.append(v)
    return gens


def cm_power_generators(D, maxdim=5, single=(6,)):
    """All Hodge classes of all powers of B = a product of pairwise non-isogenous simple CM factors of the
    Fermat-curve Jacobian with dim B <= maxdim, and of single simple factors of a dimension in `single`.  HC for
    these powers is HC-cm-sixfolds, Theorem 5.2 and Corollary E (reduced dimension <= 5, simple sixfolds), both
    assuming Markman's theorem (arXiv:2502.03415).  Contains the curve pairs and m4_generators().  Only maximal
    products are used (a sub-product's classes are among those of the product)."""
    U, cl = cm_factor_classes(D)
    combos = []

    def rec(start, budget, cur):
        if cur:
            combos.append(frozenset(cur))
        for i in range(start, len(cl)):
            if cl[i][0] <= budget:
                rec(i + 1, budget - cl[i][0], cur + [i])
    rec(0, maxdim, [])
    maximal = [c for c in combos if not any(c < e for e in combos)]
    maximal += [frozenset([i]) for i in range(len(cl)) if cl[i][0] in single and cl[i][0] > maxdim]
    gens = []
    for combo in maximal:
        gens += product_hodge_vectors(D, U, cl, sorted(combo))
    return gens


def product_generators(D, reps):
    """all Hodge classes of all powers of the product of the simple CM factors containing the lines reps."""
    U, cl = cm_factor_classes(D)
    taus = {cm_type(tuple(sorted(r)), D, U) for r in reps}
    combo = [i for i, c in enumerate(cl) if any(tau in taus for tau, t in c[1])]
    return product_hodge_vectors(D, U, cl, combo), [cl[i][0] for i in combo]


def unit_subgroups(D):
    """all subgroups of (Z/D)^x, as frozensets."""
    G = units(D)

    def gen(gs):
        S = {1}
        while True:
            T = S | {(x * g) % D for x in S for g in gs}
            if T == S:
                return frozenset(S)
            S = T
    subs, frontier = {frozenset([1])}, [frozenset([1])]
    while frontier:
        new = []
        for S in frontier:
            for g in G:
                if g not in S:
                    T = gen(list(S) + [g])
                    if T not in subs:
                        subs.add(T)
                        new.append(T)
        frontier = new
    return subs


def weil_field_generators(D, degrees):
    """Weil classes for the CM fields K' = Q(zeta_D)^M (-1 not in M, [K':Q] in degrees) on all products of
    Fermat-curve CM factors, with multiplicities and Galois-twisted copies (the units of HC-cm-dim7-8: G-orbits on
    lines(A_i) x Hom(K', C)).  The unit (A_i, gamma) has fibre
    F_tau = {lines x of A_i : x = tau gamma^-1 mod H_i M}; a Weil class at tau is sum_u c_u sum_{x in F_tau(u)} [x],
    balanced at every tau; all Galois conjugates are included.  Returns {degree: [vectors]}."""
    G = units(D)
    U, info = line_data(D)
    n = nvec(D)
    reps = {}
    for tau, (t, delta, orb, H) in info.items():
        reps.setdefault(orb, (t, frozenset(H)))
    out = {}
    for M in unit_subgroups(D):
        deg = len(G) // len(M)
        if (D - 1) in M or deg not in degrees:
            continue
        unit_f, unit_sig = [], []
        for orb, (t, H) in reps.items():
            HM = frozenset((h * m) % D for h in H for m in M)
            cos = []
            for g in G:
                c = min((g * x) % D for x in HM)
                if c not in cos:
                    cos.append(c)
            for gam in cos:
                ginv = pow(gam, -1, D)

                def fibre(g):
                    ls = {}
                    for hm in HM:
                        tt = tuple(sorted((g * ginv * hm * a) % D for a in t))
                        ls[cm_type(tt, D, U)] = tt
                    return list(ls.values())
                unit_f.append(fibre(1))
                unit_sig.append([sum(2 * cm_type(tt, D, U)[0] - 1 for tt in fibre(g)) for g in G])
        rows = [[sg[k] for sg in unit_sig] for k in range(len(G))]
        for c in integer_kernel(rows, len(unit_f)):
            for g in G:
                v = [0] * n
                for ci, F in zip(c, unit_f):
                    if ci:
                        for tt in F:
                            v = [a + ci * b for a, b in zip(v, evec(tuple((g * x) % D for x in tt), D))]
                if any(v):
                    out.setdefault(deg, []).append(v)
    return out


def odd_quadratic_characters(D):
    U = units(D)
    gens, sub = [], {1}
    for g in U:
        if g in sub:
            continue
        gens.append(g)
        new, frontier = set(sub), list(sub)
        while frontier:
            x = frontier.pop()
            for h in gens:
                y = x * h % D
                if y not in new:
                    new.add(y)
                    frontier.append(y)
        sub = new
    out = []
    for signs in itertools.product([1, -1], repeat=len(gens)):
        chi, frontier, ok = {1: 1}, [1], True
        while frontier and ok:
            x = frontier.pop()
            for h, s in zip(gens, signs):
                y, v = x * h % D, chi[x] * s
                if y in chi:
                    ok &= chi[y] == v
                else:
                    chi[y] = v
                    frontier.append(y)
        if ok and len(chi) == len(U) and chi[D - 1] == -1 and chi not in out:
            out.append(chi)
    return out


def weil6_generators(D):
    """Hodge-Weil classes of simple CM sixfolds B (factors of Fermat curve Jacobians) of Weil type for an imaginary
    quadratic k in the CM field: the wedge of the six eigenlines over one embedding of k, when its signature is (3,3).
    Algebraic by Markman's Theorem 1.5.1 IF the discriminant condition (-1) holds; used only as a test of reach."""
    U, info = line_data(D)
    idx = {s: i for i, s in enumerate(U)}
    gens, seen = [], set()
    for tau, (t, delta, orb, H) in info.items():
        if delta != 6 or orb in seen:
            continue
        seen.add(orb)
        for chi in odd_quadratic_characters(D):
            if any(chi[h] != 1 for h in H):
                continue
            Gk = [s for s in U if chi[s] == 1]
            reps, covered = [], set()
            for s in Gk:
                if s not in covered:
                    reps.append(s)
                    covered |= {s * h % D for h in H}
            if all(2 * sum(tau[idx[(s * x) % D]] for s in reps) == delta for x in U):
                gens.append([(s * a) % D for s in reps for a in t])
    return gens


# ---------------------------------------------------------------------------------------------------------------
# rigidity at CM binary forms  G = u(u^m - v^m) (d = m+1) and uv(u^m - v^m) (d = m+2): Y_G is the cyclic cover
# y^n = t (t-1)^m of P^1, n = d m, and Lambda^N H_u(Y_G) corresponds to sum over j = u mod d of the triples
# (j, j m, -j(m+1)) mod n;  ts-powers Lemma 8.10.1 identifies it with d [u/d].

def rigidity_element(d, m, u):
    n = d * m
    tup = []
    for i in range(m):
        j = (u + d * i) % n
        tri = (j, (j * m) % n, (-j * (m + 1)) % n)
        if all(tri):
            tup += list(tri)
    assert len(tup) == 3 * (d - 2)
    return tup, [(m * u) % n] * d, n


# ---------------------------------------------------------------------------------------------------------------
# Galois characters of Hodge classes: epsilon_alpha(P) = j(alpha)(P) / Np^{n/2} (Weil's Jacobi-sum character), and
# Kummer fits over the cyclotomic S-units

def isprime(n):
    return n > 1 and all(n % s for s in range(2, int(n ** 0.5) + 1))


def prim_root(p):
    m, q, fac = p - 1, 2, []
    while q * q <= m:
        if m % q == 0:
            fac.append(q)
            while m % q == 0:
                m //= q
        q += 1
    if m > 1:
        fac.append(m)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in fac):
            return g


def gauss_sums(p, d, g0):
    dl = np.zeros(p, dtype=np.int64)
    x = 1
    for a in range(p - 1):
        dl[x] = a
        x = x * g0 % p
    xs = np.arange(1, p)
    add = np.exp(2j * np.pi * xs / p)
    return {a: complex(np.sum(np.exp(2j * np.pi * a * dl[xs] / d) * add)) for a in range(1, d)}


def eps_exponent(alpha, d, p, G):
    """j with j(alpha)/p^{n/2} = exp(2 pi i j / 2d), chi(g0) = exp(2 pi i/d); None if not a 2d-th root of unity."""
    val = 1.0 + 0j
    for a in alpha:
        val *= G[a % d]
    val /= p * p ** ((len(alpha) - 2) / 2)
    for j in range(2 * d):
        if abs(val - cmath.exp(2j * cmath.pi * j / (2 * d))) < 1e-6:
            return j
    return None


def kummer_system(d, alpha, N, nprimes):
    """rows: Kummer symbols (x/P_u)_{2d} of the generators x (-1, zeta_N, primes q | N, 1 - zeta_N^a) at the primes
    P_u (zeta_N -> g0^{e u}) above p = 1 mod N; rhs: epsilon_alpha(P_u) (= eps of u^{-1} alpha with fixed chi)."""
    M = 2 * d
    UN = [u for u in range(1, N) if math.gcd(u, N) == 1]
    qs = [q for q in range(2, N + 1) if N % q == 0 and isprime(q)]
    desc = ["-1", "zeta_N"] + ["q%d" % q for q in qs] + ["1-z^%d" % a for a in range(1, N)]
    rows, rhs = [], []
    p, used = 1, 0
    while used < nprimes:
        p += N
        if not isprime(p):
            continue
        used += 1
        g0 = prim_root(p)
        dl = {}
        x = 1
        for a in range(p - 1):
            dl[x] = a
            x = x * g0 % p
        G = gauss_sums(p, d, g0)
        e = (p - 1) // N
        for u in UN:
            s = pow(g0, e * u, p)
            uinv, up = pow(u, -1, M), pow(u, -1, d)
            val = eps_exponent([up * a % d for a in alpha], d, p, G)
            if val is None:
                raise ValueError("epsilon is not a root of unity: alpha is not a Hodge tuple?")
            c = lambda xx: (dl[xx % p] * uinv) % M
            rows.append([c(p - 1), c(s)] + [c(q) for q in qs] + [c(1 - pow(s, a, p)) for a in range(1, N)])
            rhs.append(val)
    return rows, rhs, desc


def solve_mod(rows, rhs, M, cols=None, scale=None):
    """an integer solution c of sum_i rows[.][cols[i]] * scale[i] * c_i = rhs (mod M), or None."""
    if cols is None:
        cols = list(range(len(rows[0])))
    if scale is None:
        scale = [1] * len(cols)
    sub = [[r[i] * s % M for i, s in zip(cols, scale)] for r in rows]
    m, n = len(sub), len(cols)
    Ncols = n + 1 + m
    A = [sub[i] + [-rhs[i]] + [(-M if j == i else 0) for j in range(m)] for i in range(m)]
    B = flint.fmpz_mat([[A[i][j] for i in range(m)] + [1 if k == j else 0 for k in range(Ncols)]
                        for j in range(Ncols)])
    H = B.hnf()
    ker = []
    for i in range(Ncols):
        r = [int(H[i, j]) for j in range(m + Ncols)]
        if all(x == 0 for x in r[:m]):
            ker.append(r[m:])
    if not ker:
        return None
    K = flint.fmpz_mat([[v[n]] + v[:n] for v in ker]).hnf()
    first = [int(K[0, j]) for j in range(K.ncols())]
    if abs(first[0]) != 1:
        return None
    return [(-x if first[0] < 0 else x) for x in first[1:]]
