"""
fourprimes.py -- checks for ../NOTE.txt, Section 12 (degrees q with four elements in P(q): 420 = 4.3.5.7,
660 = 4.3.5.11, 780 = 4.3.5.13, 924 = 4.3.7.11, 1155 = 3.5.7.11, ...).  The labels continue at (AM) ((AL) is the
last label of eightp.py).

Notation as in NOTE Section 12: P(q) = {l_1, ..., l_r} (4 and the odd primes dividing q); an entry of order m_S
(S a set of odd primes; for 4 | q times 1, 2 or 4) is written a = (q / m) a', a' a point of G_S = prod_{l in S}
(Z/l)^x (times (Z/4)^x = {+-1} at the 2-adic level 4); P_S is the multiset of the a' at the level S, P_S^- its odd
part, [f]_T the T-component of a function f on G_S (the part of its Fourier expansion at the characters whose
support is exactly T), K = {+-1}^r, and eps_q(alpha) the number of primitive entries of alpha in a coset of K,
mod 2.

 (AM) Theorem 50 (the Fourier form of the Hodge condition): the tuples of a basis of L_q = B_q / D_q satisfy the
      relations (H_T) (odd q), resp. (I_T), (II_T) (q = 4m), and the relations have rank phi(q)/2 on R_q / D_q, so
      that they cut out B_q exactly (over Q); for 60, 84, 105, 165 (three factors) and 420, 660, 780 with a Z-basis,
      for 1155 with a 2-saturated basis (integer vectors spanning a sublattice of odd index in L_q, see
      hodge_basis; statements mod 2 on it hold on L_q).
 (AN) Propositions 51, 54 and Corollary 53: the coset counts kill D_q and every standard tuple of every degree
      dividing q, all cosets give one functional on B_q, equal to 2^(r-1) pi_U [P^-]_U mod 2 for every U of size
      r - 1; the GF(2)-dual of B_q / (D_q + S_q) has dimension 7, the lifts of the six two-factor B_q' have rank 6
      in it and rank 7 together with an element with eps_q = 1; the Gamma-coset witnesses kill D_q and S_q and,
      with eps_q, span the same subspace of the dual as all sign-blind witnesses (those invariant under the sign
      changes of the coordinates): dimension 7, 6, 5 at 420, 660, 780, vanishing on the lift of xi_55 at 660 and,
      at 780, on the span of the lift of xi_39 and the sum of the lifts of xi_20, xi_52, xi_65; the criterion of
      Proposition 54(iii) agrees on the pairs of odd primes; for each l in P(q) the kernel on the lifted classes of
      the witnesses invariant under the sign at l alone is as stated in Proposition 54(iv); B_q / (D_q + S_q) =
      (Z/2)^7 by Smith normal form (420, 660, 780; for 1155 only the first three statements, on the 2-saturated
      basis); recorded: the criterion at 924, 1020, 1092, 1140, 1155, with the parities of the order-l parts of
      xi_{l l'} against Theorem 3(i).
 (AO) Theorem 55: at 420 an element of length 15 (16 with the entry 210) with eps = 1; for every order d | 420
      the exact least number of entries of order d of an element of B_420 with eps = 1, i.e. the least L1 norm in
      the coset Proj_d(XI420) + Proj_d(ker eps), by breadth-first search on Z^n / Proj_d(ker eps) (canonical forms
      from the Hermite normal form; integers only): 1, 1, 1, 3, 3 at 15, 35, 60, 105, 140 and 0 at the other
      orders below 420, and at 420 the same search for the relaxation below (a lower bound) gives 6, the count of
      XI420; total 15; the same minima, exactly, with only the listed relations of Theorem 50(b) and one odd
      coset; at 105 and 140 the mod-3 counts of the proof of Theorem 55(c) vanish on B_420 and Z^24 /
      Proj_d(ker eps) is Z/6; the parities of the numbers of entries of orders 3, 4, 5, 7, 12, 21, 28 kill D + S,
      have rank 7 with eps on B_420, and vanish on XI420 (its class is the only one of length 15); the lower bound
      (a) for several degrees; the product elements of 660, 780, 924, 1155 (Hodge, eps_q = 1, lengths 26, 32, 16,
      62) and their counts at the levels P(q) - {l} against the expression of (b); and 924: all odd primes are 3
      mod 4, so (e) adds 1 to (a), and the element of length 16 is optimal.
 (AP) Theorem 57(i) at 420 and d = 1260: the six extended coset witnesses are even, kill all standard tuples of
      level d, are pairwise disjoint, equal 1 on the lift of the element of length 15 and avoid the lifts of R_d'
      for every divisor d' of d with P(d') != P(q).
 (AQ) Proposition 52 and Corollary 58: the parity of lambda_{q/l} is e_l [G_S : Gamma_l] eps_q on B_q for every l,
      and lambda_q = N_{P(q)} eps_q (420, 660, 780 on a Z-basis, 924 and 1155 on 2-saturated bases); at 420 the
      counts equal to eps_420, the counts vanishing on B_420, and the dimension 8 of the space of all combinations
      of counts by orders vanishing on B_420 (spanned by the listed ones); and eps_420(gamma_e) = 0 on random
      cyclic-cover families of degree 420 (N = 4, 6, 8).
Usage:  python -I fourprimes.py OUT.json [labels]      Expected last line "ALL CHECKS PASS: True".  About 10
minutes.  Needs python-flint and numpy, and ../../{cyclic-covers, fermat-frontier, ts-powers}/code.  All checks
use integer and rational arithmetic only (earlier versions minimized the counts of (AO) by HiGHS integer programs
in floating point, which do not certify lower bounds).
MIT licence (../../LICENSE).
"""

import collections
import itertools
import json
import os
import random
import sys
import time
from fractions import Fraction as Fr
from functools import reduce
from math import gcd

import flint
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _sub in ("cyclic-covers", "fermat-frontier", "ts-powers"):
    sys.path.insert(0, os.path.join(HERE, "..", "..", _sub, "code"))
from v2_exotic import evec, hnf_rows, hodge4, integer_kernel, is_hodge, lattice_L, standard, units  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# degrees, levels and coordinates

def isprime(n):
    return n > 1 and all(n % r for r in range(2, int(n ** 0.5) + 1))


def factors(q):
    """P(q): 4 (if 4 | q) and the odd primes dividing q; q must be a product of distinct such factors."""
    out = [4] if q % 4 == 0 else []
    m = q // 4 if q % 4 == 0 else q
    assert m % 2 == 1
    p = 3
    while m > 1:
        if m % p == 0:
            out.append(p)
            m //= p
            assert m % p
        p += 2
    return out


class Degree:
    """q = product of the elements of P(q) (4 and odd primes).  Levels: (e, S) with e the 2-part of the order
    (1, 2, 4; always 1 for odd q) and S the tuple of odd primes dividing the order.  The coordinates of an entry a
    at the level (e, S) are (x, u_l for l in S): x = a' mod 4 for e = 4 (else None) and u_l = a' mod l."""

    def __init__(self, q):
        self.q = q
        self.fac = factors(q)
        self.has4 = 4 in self.fac
        self.odd = tuple(l for l in self.fac if l != 4)
        self.units = {l: [x for x in range(1, l) if gcd(x, l) == 1] for l in self.fac}
        self.idx = {l: {x: i for i, x in enumerate(self.units[l])} for l in self.fac}
        self.lev = {}
        for a in range(1, q):
            self.lev[a] = self._level(a)

    def _level(self, a):
        q = self.q
        m = q // gcd(a, q)
        ap = (a // (q // m)) % m
        e = 4 if m % 4 == 0 else (2 if m % 2 == 0 else 1)
        S = tuple(l for l in self.odd if m % l == 0)
        x = ap % 4 if e == 4 else None
        return (e, S), (x,) + tuple(ap % l for l in S)

    def levels(self):
        out = []
        for k in range(len(self.odd) + 1):
            for S in itertools.combinations(self.odd, k):
                for e in ((1, 2, 4) if self.has4 else (1,)):
                    if not S and e != 4:
                        continue
                    out.append((e, S))
        return out

    def is_top(self, lv):
        return lv[1] == self.odd and (lv[0] == 4 or not self.has4)


def odd_parts(D, vec, Kp):
    """vec: signed multiplicities on the residues Kp (u < q/2) of an element of R_q / D_q (the entry q/2 ignored).
    Returns {level: ndarray}: P^-_S on G_S (axis 0 = x in (1, 3) for e = 4, then the odd coordinates)."""
    out = {}
    for lv in D.levels():
        e, S = lv
        shape = ((2,) if e == 4 else ()) + tuple(len(D.units[l]) for l in S)
        out[lv] = np.zeros(shape, dtype=np.int64)
    for u, c in zip(Kp, vec):
        if c == 0 or 2 * u == D.q:
            continue
        for a, s in ((u, c), (D.q - u, -c)):
            lv, co = D.lev[a]
            ix = (((0 if co[0] == 1 else 1),) if lv[0] == 4 else ()) + tuple(
                D.idx[l][x] for l, x in zip(lv[1], co[1:]))
            out[lv][ix] += s
    return out


def component(f, axes_T, nax):
    """|G_S| * [f]_T as an integer array over the axes in axes_T (others summed): sum over U in T of
    (-1)^{|T - U|} |G_U| (sum of f over the axes not in U)."""
    sizes = f.shape
    tot = None
    for k in range(len(axes_T) + 1):
        for U in itertools.combinations(axes_T, k):
            others = tuple(a for a in range(nax) if a not in U)
            g = f.sum(axis=others) if others else f.copy()
            gu = 1
            for a in U:
                gu *= sizes[a]
            # broadcast g (over axes U) to the axes axes_T
            shp = [sizes[a] if a in U else 1 for a in axes_T]
            g = g.reshape(shp) * gu * (-1) ** (len(axes_T) - k)
            tot = g if tot is None else tot + g
    return np.broadcast_to(tot, tuple(sizes[a] for a in axes_T)).copy()


def mult_perm(D, l, mod):
    """permutation of the index set of (Z/mod)^x given by multiplication by l."""
    return np.array([D.idx[mod][(l * x) % mod] for x in D.units[mod]])


def apply_mult(D, f, mods, l, twist4=False):
    """(f(l .))(u) on prod (Z/mod)^x (mods: the factor of each axis; 4 stands for (Z/4)^x)."""
    g = f
    for ax, mod in enumerate(mods):
        g = np.take(g, mult_perm(D, l, mod), axis=ax)
    return g


def delta(D, f, mods, l, chi4_twist=False):
    """Delta_l f = f(l .) - f, or chi_4(l) f(l .) - f (the twisted operator of (II))."""
    s = (1 if l % 4 == 1 else -1) if chi4_twist else 1
    return s * apply_mult(D, f, mods, l) - f


def mult_of(D, l):
    """the multiplier of the factor l in the relations: l itself for odd l, 2 for the factor 4."""
    return 2 if l == 4 else l


def relations(D, parts):
    """the residuals of the relations of Theorem 50 (odd q: (H_T), T nonempty) or 50(b) (q = 4m: (I_T), T nonempty,
    and (II_T), T arbitrary), scaled to integers; all vanish iff the element is Hodge."""
    out = []
    O = D.odd
    gsize = {S: int(np.prod([len(D.units[l]) for l in S])) if S else 1 for k in range(len(O) + 1)
             for S in itertools.combinations(O, k)}
    Gall = gsize[O]
    if not D.has4:
        for k in range(1, len(O) + 1):
            for T in itertools.combinations(O, k):
                tot = 0
                for S in [S for j in range(len(O) + 1) for S in itertools.combinations(O, j) if set(T) <= set(S)]:
                    f = parts[(1, S)]
                    axes = tuple(S.index(l) for l in T)
                    c = component(f, axes, len(S)) * (Gall // gsize[S])
                    for l in S:
                        if l not in T:
                            c = -delta(D, c, T, l)
                    tot = tot + c
                out.append(np.asarray(tot).ravel())
        return out
    # q = 4m: X_S = 2 P^-_{1S} - Delta_2 (2 P^-_{2S} + 2 A_S) and 2 B_S
    X, B = {}, {}
    for j in range(len(O) + 1):
        for S in itertools.combinations(O, j):
            f4 = parts[(4, S)]
            A2, B2 = f4[0] + f4[1], f4[0] - f4[1]
            B[S] = B2
            if S:
                X[S] = 2 * parts[(1, S)] - delta(D, 2 * parts[(2, S)] + A2, S, 2)
    subsets = [S for j in range(len(O) + 1) for S in itertools.combinations(O, j)]
    for k in range(1, len(O) + 1):                                   # (I_T)
        for T in itertools.combinations(O, k):
            tot = 0
            for S in [S for S in subsets if set(T) <= set(S)]:
                axes = tuple(S.index(l) for l in T)
                c = component(X[S], axes, len(S)) * (Gall // gsize[S])
                for l in S:
                    if l not in T:
                        c = -delta(D, c, T, l)
                tot = tot + c
            out.append(np.asarray(tot).ravel())
    for k in range(0, len(O) + 1):                                   # (II_T)
        for T in itertools.combinations(O, k):
            tot = 0
            for S in [S for S in subsets if set(T) <= set(S)]:
                axes = tuple(S.index(l) for l in T)
                if S:
                    c = component(B[S], axes, len(S)) * (Gall // gsize[S])
                else:
                    c = np.asarray(B[S]) * Gall
                for l in S:
                    if l not in T:
                        c = -delta(D, c, T, l, chi4_twist=True) if T else -((1 if l % 4 == 1 else -1) - 1) * c
                tot = tot + c
            out.append(np.asarray(tot).ravel())
    return out


def is_zero(rel):
    return all(not np.any(r) for r in rel)


def tuple_vec(t, q, Kp):
    idx = {u: i for i, u in enumerate(Kp)}
    v = [0] * len(Kp)
    for a in t:
        a %= q
        if a == 0 or 2 * a == q:
            continue
        if 2 * a < q:
            v[idx[a]] += 1
        else:
            v[idx[q - a]] -= 1
    return v


# ---------------------------------------------------------------------------------------------------------------
# the lattice B_q / D_q, standard tuples, witnesses

SATURATION = {}


def saturate2(V):
    """2-saturation of a basis V (integer vectors) of a sublattice of finite index in a saturated lattice L (L =
    Z^n cap (L (x) Q)):
    while the vectors are dependent mod 2, take a basis of the dependencies in reduced form (each dependency c has
    an index i_c where it is 1 and the others are 0) and replace each V[i_c] by (sum_i c_i V[i]) / 2, an integer
    vector of L.  The old vectors stay in the span (V[i_c] = 2 new - the other terms), so each round enlarges the
    lattice by a power of 2 and the loop ends; afterwards V is independent mod 2, so its span has odd index in L
    and a statement mod 2 checked on V holds on L.  Returns (V, number of halvings)."""
    V = [[x // reduce(gcd, v) for x in v] for v in V]
    steps = 0
    while True:
        N, k = flint.nmod_mat([[x % 2 for x in v] for v in V], 2).transpose().nullspace()
        if k == 0:
            return V, steps
        C = [[int(N[i, j]) for i in range(len(V))] for j in range(k)]
        new = list(V)
        for j, c in enumerate(C):
            i0 = next(i for i in range(len(V)) if c[i] and all(C[jj][i] == 0 for jj in range(k) if jj != j))
            s = [sum(V[i][t] for i in range(len(V)) if c[i]) for t in range(len(V[0]))]
            assert all(x % 2 == 0 for x in s)
            new[i0] = [x // 2 for x in s]
            steps += 1
        V = new


def hodge_basis(q, zbasis=True):
    """(Kp, vectors): a Z-basis of L_q = B_q / D_q (lattice_L of ts-powers, Hermite normal form) if zbasis, else a
    2-saturated basis of L_q (O): the flint nullspace (integer vectors spanning a sublattice of finite, possibly
    even, index; at 1155 its reduction mod 2 has rank 1), made independent mod 2 by saturate2, so that its index
    in L_q is odd; on the coordinates Kp = {u : 2u < q}."""
    if zbasis:
        return lattice_L(q)
    t0 = time.time()
    Kp = [u for u in range(1, q) if 2 * u < q]
    w = lambda k: (q - 2 * (k % q)) if (k % q) and 2 * (k % q) != q else 0
    rows = [[w(t * u) for u in Kp] for t in units(q) if 2 * t < q]
    X, nul = flint.fmpz_mat(rows).nullspace()
    V0 = [[int(X[i, j]) for i in range(len(Kp))] for j in range(nul)]
    r0 = rank2([[x % 2 for x in v] for v in V0])
    V, steps = saturate2(V0)
    assert all(sum(a * b for a, b in zip(r, v)) == 0 for r in rows for v in V)
    assert rank2([[x % 2 for x in v] for v in V]) == len(V) == nul
    SATURATION[q] = {"rank": nul, "rank_mod_2_before": r0, "halvings": steps, "sec": round(time.time() - t0, 1)}
    return Kp, V


class Lattice:
    """B_q / D_q on the coordinates Kp = {u : 2u < q} (and q/2 for q even, as in threeprime.py): the vectors
    (e, sum e mod 2) for a basis e of L_q, and (0, 2) for q even; the standard tuples of every degree dividing q,
    lifted (with the linear pair (q/2, q/2))."""

    def __init__(self, q, zbasis=True):
        self.q = q
        self.Kp, self.Lb = hodge_basis(q, zbasis)
        self.m = len(self.Kp)
        self.n = self.m + (1 if q % 2 == 0 else 0)
        self.B = [list(e) + ([sum(e) % 2] if q % 2 == 0 else []) for e in self.Lb]
        if q % 2 == 0:
            self.B.append([0] * self.m + [2])
        self.S = []
        for d in range(3, q + 1):
            if q % d == 0:
                for s in standard(d):
                    self.S.append(evec(tuple((q // d) * x for x in s), q))
        if q % 2 == 0:
            self.S.append([0] * self.m + [2])

    def f2(self, f):
        """the values mod 2 of a function f on Kp (f(q/2) = 0) on the generators of B_q."""
        return tuple(sum(f[i] * b[i] for i in range(self.m)) % 2 for b in self.B)

    def kills_standard(self, f):
        return all(sum(f[i] * s[i] for i in range(self.m)) % 2 == 0 for s in self.S)

    def dual_rank(self):
        """dim_F2 of the functionals on B_q that kill D_q + S_q (restrictions of witnesses)."""
        Sm = flint.nmod_mat([[x % 2 for x in s[:self.m]] for s in self.S], 2)
        N, nul = Sm.nullspace()
        F = [[int(N[i, j]) for i in range(self.m)] for j in range(nul)]
        return rank2([self.f2(f) for f in F]), F

    def lift(self, q1):
        """the lifts of a basis of B_{q1} / D_{q1} (q1 | q) as vectors on the coordinates of B_q."""
        Kp1, Lb1 = lattice_L(q1)
        k = self.q // q1
        out = []
        for e in Lb1:
            v = [0] * self.n
            for u, x in zip(Kp1, e):
                v[k * u - 1] += x
            if self.q % 2 == 0 and sum(e) % 2:
                v[-1] += 1
            out.append(v)
        return out


def rank2(rows):
    rows = [list(r) for r in rows if any(r)]
    if not rows:
        return 0
    return flint.nmod_mat(rows, 2).rank()


def snf_quotient(L):
    """invariant factors > 1 of B_q / (D_q + S_q) (as threeprime.quotient, with the standard span only).  The
    standard tuples are first reduced to one vector up to sign (the same lattice; the Hermite form is then fast)."""
    n = L.n
    uniq = set()
    for s in L.S:
        j = next((i for i, x in enumerate(s) if x), None)
        if j is not None:
            uniq.add(tuple(s) if s[j] > 0 else tuple(-x for x in s))
    H = hnf_rows([list(s) for s in sorted(uniq)], n)
    Bh = hnf_rows([list(b) for b in L.B], n)
    piv = [next(j for j in range(n) if r[j]) for r in Bh]
    Bs = flint.fmpq_mat([[r[j] for j in piv] for r in Bh])
    Hs = flint.fmpq_mat([[r[j] for j in piv] for r in H])
    X = Hs * Bs.inv()
    assert all(X[i, j].q == 1 for i in range(X.nrows()) for j in range(X.ncols()))
    Xz = [[int(X[i, j].p) for j in range(X.ncols())] for i in range(X.nrows())]
    assert all(sum(Xz[i][k] * Bh[k][j] for k in range(len(Bh))) == H[i][j] for i in range(len(H)) for j in range(n))
    S = flint.fmpz_mat(Xz).snf()
    diag = [int(S[i, i]) for i in range(min(S.nrows(), S.ncols()))]
    return [x for x in diag if x not in (0, 1)] + [0] * (len(Bh) - sum(1 for x in diag if x))


def coset_key(D, a):
    """the K-coset of a primitive residue a (None for the others): the coordinates up to sign."""
    lv, co = D.lev[a]
    if not D.is_top(lv):
        return None
    mods = ((4,) if D.has4 else ()) + lv[1]
    return tuple(min(c, l - c) for c, l in zip(co[(0 if D.has4 else 1):], mods))


def coset_witnesses(D, Kp):
    keys = sorted({coset_key(D, u) for u in Kp} - {None})
    return {k: [1 if coset_key(D, u) == k else 0 for u in Kp] for k in keys}


def eps_formula(D, parts, U):
    """2^(r-1) pi_U [P^-]_U on the K-cosets, from the components (U a set of r - 1 elements of P(q)); returns the
    set of its values mod 2 over all points.  For q = 4m and 4 in U the chi_4-part B of the top is used."""
    O = D.odd
    top = (4, O) if D.has4 else (1, O)
    f = parts[top]
    if D.has4:
        A2, B2 = f[0] + f[1], f[0] - f[1]
        g = B2 if 4 in U else A2                    # 2 A, 2 B on G_O
        Uo = tuple(l for l in U if l != 4)
    else:
        g, Uo = f, tuple(U)
    axes = tuple(O.index(l) for l in Uo)
    c = component(g, axes, len(O))                  # |G_O| [g]_U  (g = 2A or 2B, or P^- for odd q)
    # pi_U: sum over sign changes of the U-coordinates with the character chi_U, divided by 2^|U|
    vals = set()
    Gsz = int(np.prod(g.shape))
    for ix in itertools.product(*[range(c.shape[a]) for a in range(len(axes))]):
        tot = 0
        for signs in itertools.product((1, -1), repeat=len(axes)):
            jx = []
            for a, s, i in zip(axes, signs, ix):
                l = O[a]
                u = D.units[l][i]
                jx.append(D.idx[l][(s * u) % l])
            sgn = 1
            for s in signs:
                sgn *= s
            tot += sgn * int(c[tuple(jx)])
        # 2^(r-1) pi_U [P^-]_U = 2^(r-1) 2^-|U| tot / |G_O| (/2 for the halved A, B when 4 | q)
        num = Fr(2 ** (len(D.fac) - 1) * tot, 2 ** len(axes) * Gsz * (2 if D.has4 else 1))
        if D.has4 and 4 in U:
            num = num                                  # chi_4 sign of the coordinate x is +1 at x = 1
        assert num.denominator == 1, num
        vals.add(int(num) % 2)
    return vals


def gamma_group(D, E):
    """Gamma_E = <K_E, the multipliers of the elements of P(q) not in E> (2 for the factor 4), as a set of tuples
    in G_E = prod_{l in E} (Z/l)^x."""
    mults = [mult_of(D, l) for l in D.fac if l not in E]
    sign = [tuple(((m - 1) if x == y else 1) for y, m in zip(E, E)) for x in E]
    gens = sign + [tuple(mu % m for m in E) for mu in mults]
    grp = {tuple(1 for _ in E)}
    fr = list(grp)
    while fr:
        a = fr.pop()
        for g in gens:
            c = tuple((x * y) % m for x, y, m in zip(a, g, E))
            if c not in grp:
                grp.add(c)
                fr.append(c)
    return grp


def primitive_root(l):
    return next(g for g in range(2, l) if len({pow(g, k, l) for k in range(1, l)}) == l - 1)


def aoki_order_l_part(l, p):
    """the coordinates (mod l) of the entries of order l of Aoki's xi_{l p} (NOTE Lemma 2, l, p odd primes):
    the h in H_l with u^-1 h in H_l, u = -p^-1 mod l, H_l = {1, mu, ..., mu^((l-3)/2)}, mu the least primitive
    root (signs are irrelevant mod 2)."""
    mu = primitive_root(l)
    H = {pow(mu, k, l) for k in range((l - 1) // 2)}
    ui = pow((-pow(p, -1, l)) % l, -1, l)
    return sorted(h for h in H if ui * h % l in H)


def signblind_criterion(D, E):
    """Proposition 54(iii) for two odd primes E = (l1, l2) of P(q): ([Gamma_E : K_E], the parities of the numbers
    of entries of order l of xi_E in the cosets of Gamma_{l} (l in E), and whether a sign-blind witness detects the
    lift of xi_E: iff the index is odd or one of these numbers is odd)."""
    index = len(gamma_group(D, E)) // 2 ** len(E)
    odd_counts = {}
    for l, p in ((E[0], E[1]), (E[1], E[0])):
        g1 = {x[0] for x in gamma_group(D, (l,))}
        ent = aoki_order_l_part(l, p)
        seen, par = set(), []
        for t in D.units[l]:
            if t not in seen:
                C = {t * g % l for g in g1}
                seen |= C
                par.append(sum(1 for h in ent if h in C) % 2)
        odd_counts[l] = par
    det = index % 2 == 1 or any(any(v) for v in odd_counts.values())
    return index, odd_counts, det


def unit_with(D, res):
    """the unit mod q with the given residues (a dict over P(q)) and 1 at the other factors."""
    r = [res.get(l, 1) % l for l in D.fac]
    return crt_all(r, D.fac)


def invariant_witnesses(L, D, gs):
    """the parity witnesses (functions on Kp, i.e. even, killing every lifted standard tuple mod 2) that are
    invariant under the units in gs."""
    q = L.q
    pos = {u: i for i, u in enumerate(L.Kp)}
    rows = [[x % 2 for x in s[:L.m]] for s in L.S]
    for g in gs:
        for i, u in enumerate(L.Kp):
            v = g * u % q
            j = pos[v if 2 * v < q else q - v]
            if j != i:
                row = [0] * L.m
                row[i], row[j] = 1, 1
                rows.append(row)
    N, nul = flint.nmod_mat(rows, 2).nullspace()
    return [[int(N[i, j]) for i in range(L.m)] for j in range(nul)]


def gamma_witnesses(D, Kp):
    """Proposition 54: for E a set of one or two elements of P(q), Gamma_E = <K_E, the multipliers of the other
    elements> (2 for the factor 4) in G_E, and C a coset of Gamma_E: f = 1 at the entries whose level contains E
    and whose E-coordinates lie in C (for 4 | q and 4 not in E: only at the 2-adic levels in a set w of {1, 4}).
    Returns {(E, w, coset index): f}."""
    out = {}
    mods_all = D.fac
    for size in (1, 2):
        for E in itertools.combinations(mods_all, size):
            grp = gamma_group(D, E)
            pts = list(itertools.product(*[D.units[m] for m in E]))
            seen, cosets = set(), []
            for t in pts:
                if t not in seen:
                    C = {tuple((x * y) % m for x, y, m in zip(t, g, E)) for g in grp}
                    seen |= C
                    cosets.append(C)
            wsets = [(1,), (4,), (1, 4)] if (D.has4 and 4 not in E) else [None]
            for w in wsets:
                for ci, C in enumerate(cosets):
                    f = []
                    for u in Kp:
                        lv, co = D.lev[u]
                        e, S = lv
                        val = 0
                        if all((e == 4) if l == 4 else (l in S) for l in E) and (w is None or e in w):
                            coord = {l: (co[0] if l == 4 else co[1 + S.index(l)]) for l in E}
                            val = 1 if tuple(coord[l] for l in E) in C else 0
                        f.append(val)
                    out[(E, w, ci)] = f
    return out


def nn(l):
    return 1 if l in (3, 4) else (l - 1) // 2


def order_mod_K(m, S):
    """the order of the multiplier m in G_S / K_S (S a tuple of elements of P(q); (Z/4)^x = K_4)."""
    o = 1
    while not all(pow(m, o, l) in (1, l - 1) for l in S):
        o += 1
    return o


def twist(m, o, S):
    t = 1
    for l in S:
        t *= 1 if pow(m, o, l) == 1 else -1
    return t


def lower_bound_a(fac):
    """Theorem 55(a): phi(q)/2^r + sum_k e_k [G_S : Gamma_S], S = P(q) - {l_k}."""
    top = 1
    for l in fac:
        top *= nn(l)
    tot, terms = top, {}
    for k in fac:
        S = tuple(l for l in fac if l != k)
        m = 2 if k == 4 else k
        o = order_mod_K(m, S)
        N = 1
        for l in S:
            N *= nn(l)
        e = 1 if twist(m, o, S) == -1 else 0
        terms[k] = (e, N // o)
        tot += e * (N // o)
    return tot, top, terms


def sign_fn(H, l):
    Hs = {h % l for h in H}
    return lambda x: 1 if x % l in Hs else -1


def level3_formula(fac, H):
    """Theorem 55(b): for the primitive part prod H_l, the level P(q) - {l_k} needs at least
    (N_S / 2)(1 - prod_{l in S} delta_l(m_k)) entries, delta_l(m) = (1/n_l) sum_{h in H_l} s_l(m h)."""
    out = {}
    for k in fac:
        S = [l for l in fac if l != k]
        m = 2 if k == 4 else k
        N, prod = 1, Fr(1)
        for l in S:
            N *= nn(l)
            s = sign_fn(H[l], l)
            prod *= Fr(sum(s(m * h) for h in H[l]), len(H[l]))
        out[k] = Fr(N, 2) * (1 - prod)
    return out


def crt_all(res, mods):
    x, M = 0, 1
    for r, mo in zip(res, mods):
        while x % mo != r % mo:
            x += M
        M *= mo
    return x


def eps_of_tuple(D, t):
    """the K-coset counts of a tuple (a dict coset -> count mod 2)."""
    cnt = {}
    for a in t:
        k = coset_key(D, a % D.q)
        if k is not None:
            cnt[k] = (cnt.get(k, 0) + 1) % 2
    keys = {coset_key(D, u) for u in range(1, D.q)} - {None}
    return {k: cnt.get(k, 0) for k in keys}


def level_counts(q, t):
    out = {}
    for a in t:
        a %= q
        if a and 2 * a != q:
            d = q // gcd(a, q)
            out[d] = out.get(d, 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------------------------------------------
# explicit elements (found by the integer programs described in NOTE Theorem 55; checked here in exact arithmetic)

# 420: least length 15 (+ the entry 210), primitive part H_4 x H_3 x H_5 x H_7 with H_7 the squares mod 7
XI420 = (1, 28, 33, 37, 57, 121, 213, 236, 276, 277, 337, 344, 361, 413, 416)
HALF420 = {4: [1], 3: [1], 5: [1, 2], 7: [1, 2, 4]}
# 660, 780, 1155: product elements (primitive part prod H_l) of the stated lengths
XI660 = (1, 45, 61, 88, 97, 157, 177, 237, 265, 296, 297, 324, 337, 344, 360, 361, 397, 399, 416, 421, 457, 568,
         579, 601, 644, 651)
HALF660 = {4: [1], 3: [1], 5: [1, 2], 11: [1, 3, 6, 7, 9]}
XI780 = (1, 16, 87, 97, 121, 124, 128, 145, 157, 183, 189, 277, 301, 309, 316, 324, 345, 425, 447, 457, 533, 541,
         601, 663, 664, 676, 687, 697, 721, 736, 755, 757)
HALF780 = {4: [1], 3: [1], 5: [1, 2], 13: [1, 2, 3, 4, 6, 8]}
XI1155 = (1, 16, 22, 28, 37, 67, 96, 159, 161, 174, 232, 238, 247, 256, 279, 331, 339, 343, 352, 361, 421, 426,
          444, 448, 466, 487, 489, 504, 516, 526, 562, 591, 592, 631, 649, 652, 654, 676, 697, 721, 751, 757, 841,
          846, 847, 856, 862, 868, 907, 931, 961, 979, 982, 991, 999, 1011, 1012, 1072, 1087, 1104, 1106, 1144)
HALF1155 = {3: [1], 5: [1, 2], 7: [1, 2, 4], 11: [1, 3, 4, 5, 9]}
# 924 (all odd primes = 3 mod 4): the quadratic-residue half systems and one entry of order 4; L(924) = 16
XI924 = (1, 25, 37, 169, 289, 361, 421, 445, 529, 625, 693, 697, 709, 757, 793, 841)
HALF924 = {4: [1], 3: [1], 7: [1, 2, 4], 11: [1, 3, 4, 5, 9]}

# minimal sets of relations of Theorem 50(b) forcing the per-level minima at 420 (one K-coset odd); NOTE 55(c)
MINSETS420 = {420: [("I", (3, 5, 7))], 140: [("II", (5, 7))], 105: [("I", (3, 5, 7))],
              60: [("I", (5, 7)), ("II", (3, 5))], 35: [("I", (7,)), ("I", (3, 5, 7))],
              15: [("I", (3, 5)), ("I", (3, 5, 7))]}

_CACHE = {}


def lattice(q):
    """Z-basis for q <= 780; for 924 and 1155 a 2-saturated basis (index in L_q odd; the Hermite form of a Z-basis
    is too slow)."""
    if q not in _CACHE:
        _CACHE[q] = Lattice(q, q <= 780)
    return _CACHE[q]


def relation_labels(D):
    O = D.odd
    if D.has4:
        return ([("I", T) for k in range(1, len(O) + 1) for T in itertools.combinations(O, k)]
                + [("II", T) for k in range(0, len(O) + 1) for T in itertools.combinations(O, k)])
    return [("H", T) for k in range(1, len(O) + 1) for T in itertools.combinations(O, k)]


def relation_matrix(D, Kp):
    """columns: the relations of Theorem 50 on the unit vectors of Kp; returns [block matrix per label]."""
    cols = [relations(D, odd_parts(D, [1 if j == i else 0 for j in range(len(Kp))], Kp)) for i in range(len(Kp))]
    return [np.array([c[b] for c in cols]).T for b in range(len(cols[0]))]


def check_AM():
    rows, ok = [], True
    for q in (60, 84, 105, 165, 420, 660, 780, 1155):
        t0 = time.time()
        D = Degree(q)
        if q in (420, 660, 780, 1155):
            Kp, Lb = lattice(q).Kp, lattice(q).Lb
        else:
            Kp, Lb = hodge_basis(q)
        basis_ok = all(is_zero(relations(D, odd_parts(D, e, Kp))) for e in Lb)
        blocks = relation_matrix(D, Kp)
        M = np.concatenate(blocks, axis=0)
        M = M[np.any(M != 0, axis=1)]
        rk = flint.fmpz_mat([[int(x) for x in r] for r in M]).rank()
        phi = len(units(q))
        good = basis_ok and rk == phi // 2 and len(Kp) - rk == len(Lb)
        # the top: no component of full support (odd q: (H_P); q = 4m: (II_O)), on the lattice basis
        ok &= good
        kind = "2-saturated basis" if q in SATURATION else "Z-basis"
        rows.append({"q": q, "P": factors(q), "rank_L": len(Lb), "basis_satisfies": basis_ok, "rank_relations": rk,
                     "phi_over_2": phi // 2, "basis": kind, "saturation": SATURATION.get(q), "ok": good,
                     "sec": round(time.time() - t0, 1)})
        print("    (AM) q=%d: basis of L_q (%s) satisfies the relations: %s; rank of the relations %d = "
              "phi(q)/2 = %d; rank L_q %d: %s" % (q, kind, basis_ok, rk, phi // 2, len(Lb), good), flush=True)
        if q in SATURATION:
            print("    (AM) q=%d: the flint nullspace basis has rank %d mod 2; after %d halvings it is independent "
                  "mod 2 (odd index in L_q)" % (q, SATURATION[q]["rank_mod_2_before"], SATURATION[q]["halvings"]),
                  flush=True)
    return rows, ok


def product_element_check(q, t, H):
    """the tuple t: Hodge (with q/2 added if needed), primitive part = prod H_l, all K-coset counts odd."""
    D = Degree(q)
    tt = tuple(t) + ((q // 2,) if (sum(t) % q or len(t) % 2) else ())
    hodge = is_hodge(tt, q)
    prim = sorted(a for a in t if gcd(a, q) == 1)
    fac = D.fac
    want = sorted(crt_all(c, fac) for c in itertools.product(*[H[l] for l in fac]))
    cos = eps_of_tuple(D, t)
    return hodge, prim == want, all(v == 1 for v in cos.values())


def check_AN():
    rows, ok = [], True
    for q in (420, 660, 780, 1155):
        t0 = time.time()
        D = Degree(q)
        L = lattice(q)
        W = coset_witnesses(D, L.Kp)
        kill = all(L.kills_standard(f) for f in W.values())
        vals = {L.f2(f) for f in W.values()}
        eps = next(iter(vals))
        # the formula 2^(r-1) pi_U [P^-]_U mod 2 on the lattice basis, for every U of size r - 1
        form = True
        for e, ev in zip(L.Lb, eps):
            parts = odd_parts(D, e, L.Kp)
            for U in itertools.combinations(D.fac, len(D.fac) - 1):
                form &= eps_formula(D, parts, U) == {ev}
        if q > 780:
            # 1155: only (i), (ii) of Proposition 51, on the 2-saturated basis (odd index in L_q, so the statements
            # mod 2 hold on B_q)
            good = kill and len(vals) == 1 and form and any(eps)
            ok &= good
            rows.append({"q": q, "cosets": len(W), "coset_witnesses_kill_D_S": kill,
                         "one_functional": len(vals) == 1, "formula": form, "eps_nonzero": any(eps),
                         "basis": "2-saturated (odd index in L_q)", "ok": good, "sec": round(time.time() - t0, 1)})
            print("    (AN) q=%d (2-saturated basis): %d cosets, witnesses %s, one functional %s (nonzero %s), "
                  "formula %s: %s" % (q, len(W), kill, len(vals) == 1, any(eps), form, good), flush=True)
            continue
        rk, F = L.dual_rank()
        # generation: the lifts of B_{q'} (q' | q with two elements in P(q')) and one element with eps_q = 1
        gens = []
        two = [a * b for a, b in itertools.combinations(D.fac, 2)]
        for q1 in two:
            gens += L.lift(q1)
        xi = {420: XI420, 660: XI660, 780: XI780, 1155: XI1155}[q]
        if xi:
            v = tuple_vec(xi, q, L.Kp) + ([0] if q % 2 == 0 else [])
            gens.append(v)
        Fm = [[sum(f[i] * g[i] for i in range(L.m)) % 2 for g in gens] for f in F]
        rk_gen = rank2(Fm)
        rk_lifts = rank2([r[:-1] for r in Fm]) if xi else rank2(Fm)
        nl = len(gens) - (1 if xi else 0)
        eps_lifts = all(sum(next(iter(W.values()))[i] * g[i] for i in range(L.m)) % 2 == 0 for g in gens[:nl])
        # Gamma-coset witnesses and sign-blind witnesses (Proposition 54)
        GW = gamma_witnesses(D, L.Kp)
        gw_ok = all(L.kills_standard(f) for f in GW.values())
        rk_gw = rank2([L.f2(f) for f in GW.values()] + [eps])
        signs = {l: unit_with(D, {l: -1}) for l in D.fac}
        SB = invariant_witnesses(L, D, list(signs.values()))
        rk_sb = rank2([L.f2(f) for f in SB] + [eps])
        # each lift class: a lifted basis vector of B_{q'} outside D + S (some witness is nonzero on it)
        pairs = list(itertools.combinations(D.fac, 2))
        reps = {}
        for E in pairs:
            reps[E] = next(v for v in L.lift(E[0] * E[1])
                           if any(sum(f[i] * v[i] for i in range(L.m)) % 2 for f in F))

        def kernel_span(ws):
            # the sums of lift classes on which every witness in ws vanishes, as a set of frozensets of pairs
            M = flint.nmod_mat([[sum(f[i] * reps[E][i] for i in range(L.m)) % 2 for E in pairs] for f in ws], 2)
            N, nul = M.nullspace()
            basis = [[int(N[k, j]) for k in range(len(pairs))] for j in range(nul)]
            out = set()
            for cs in itertools.product((0, 1), repeat=nul):
                v = [sum(c * b[k] for c, b in zip(cs, basis)) % 2 for k in range(len(pairs))]
                if any(v):
                    out.add(frozenset(E for k, E in enumerate(pairs) if v[k]))
            return out
        ker_sb = kernel_span(SB)
        ker_gw = kernel_span(list(GW.values()))
        crit = {E: signblind_criterion(D, E) for E in pairs if 4 not in E}
        crit_ok = all((frozenset([E]) in ker_sb) == (not c[2]) for E, c in crit.items())
        def span_of(gens):
            # the nonzero sums of the given sums of lift classes, as a set of frozensets of pairs
            out = set()
            for cs in itertools.product((0, 1), repeat=len(gens)):
                v = set()
                for c, b in zip(cs, gens):
                    if c:
                        v ^= set(b)
                if v:
                    out.add(frozenset(v))
            return out
        xi55, xi39, s3 = [(5, 11)], [(3, 13)], [(4, 5), (4, 13), (5, 13)]
        expect = {420: [], 660: [xi55], 780: [xi39, s3]}[q]
        span_exp = span_of(expect)
        # single signs: the kernel for the witnesses invariant under the sign at l alone (Proposition 54(iv), sharp)
        ker_single = {l: kernel_span(invariant_witnesses(L, D, [signs[l]])) for l in D.fac}
        single_exp = {660: {4: [xi55], 3: [xi55], 5: [xi55], 11: []},
                      780: {4: [xi39], 3: [s3], 5: [xi39, s3], 13: [xi39, s3]}}.get(q, {l: [] for l in D.fac})
        single_ok = all(ker_single[l] == span_of(single_exp[l]) for l in D.fac)
        good = kill and len(vals) == 1 and form and rk == 7 and rk_lifts == 6 and eps_lifts and gw_ok
        good &= rk_gw == rk_sb and ker_sb == ker_gw == span_exp and rk_sb + len(expect) == 7 and crit_ok
        good &= single_ok
        good &= (rk_gen == 7) if xi else True
        row = {"q": q, "cosets": len(W), "coset_witnesses_kill_D_S": kill, "one_functional": len(vals) == 1,
               "formula": form, "dual_rank": rk, "rank_lifts": rk_lifts, "rank_lifts_plus_xi": rk_gen,
               "eps_kills_lifts": eps_lifts, "gamma_witnesses": len(GW), "gamma_ok": gw_ok, "rank_gamma_eps": rk_gw,
               "signblind_witnesses_dim": len(SB), "rank_signblind_eps": rk_sb,
               "kernel_on_lifts": sorted(sorted(map(list, k)) for k in ker_sb),
               "criterion_odd_pairs": {"%d,%d" % E: [c[0], {str(l): v for l, v in c[1].items()}, c[2]]
                                       for E, c in crit.items()},
               "criterion_agrees": crit_ok,
               "single_sign_kernels": {str(l): sorted(sorted(map(list, k)) for k in ks)
                                       for l, ks in ker_single.items()},
               "single_sign_claims": single_ok}
        # the Smith normal form of B_q / (D_q + S_q): (Z/2)^7, so the seven classes span it (Corollary 53)
        t1 = time.time()
        snf = snf_quotient(L)
        row["snf"] = snf
        row["snf_sec"] = round(time.time() - t1, 1)
        good &= snf == [2] * 7
        ok &= good
        row["ok"] = good
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        print("    (AN) q=%d: %d cosets, witnesses %s, one functional %s, formula %s, dual rank %d, lifts %d, "
              "+ xi %d, Gamma-witnesses %d (rank with eps %d), sign-blind witnesses rank with eps %d, their kernel "
              "on the lifts %s, criterion %s, single signs %s%s: %s" % (
                  q, len(W), kill, len(vals) == 1, form, rk, rk_lifts, rk_gen, len(GW), rk_gw, rk_sb,
                  row["kernel_on_lifts"], crit_ok, single_ok, ", SNF %s" % row["snf"], good),
              flush=True)
    # the criterion of Proposition 54(iii) at further degrees (odd pairs only; recorded), with the total parities
    # of the order-l parts against Theorem 3(i)
    rec, t3 = {}, True
    for q in (924, 1020, 1092, 1140, 1155):
        D = Degree(q)
        rec[q] = {}
        for E in itertools.combinations(D.odd, 2):
            idx, oc, det = signblind_criterion(D, E)
            rec[q]["%d,%d" % E] = [idx, {str(l): v for l, v in oc.items()}, det]
            for l, p in ((E[0], E[1]), (E[1], E[0])):
                t3 &= len(aoki_order_l_part(l, p)) % 2 == (1 if pow(p, (l - 1) // 2, l) == l - 1 else 0)
    ok &= t3
    rows.append({"criterion_recorded": rec, "theorem3_parity": t3})
    print("    (AN) criterion of Proposition 54(iii), odd pairs without a sign-blind witness: %s; order-l parts "
          "against Theorem 3(i): %s" % ({q: [E for E, v in r.items() if not v[2]] for q, r in rec.items()}, t3),
          flush=True)
    return rows, ok


def hnf_reducer(gens, n):
    """canonical representatives modulo the subgroup M of Z^n spanned by gens (of any rank): with the rows h_i of
    the Hermite normal form of M (pivots p_1 < p_2 < ...), subtract multiples of h_1, h_2, ... in turn to bring the
    coordinate p_i into [0, h_i[p_i]); two vectors are congruent mod M iff their reductions agree (if w - w' =
    sum c_i h_i with c_i the first nonzero coefficient, the coordinate p_i of w - w' is c_i h_i[p_i]).  Returns
    (reduce, rank of M, order of Z^n / M or None if it is infinite); reduce(w, j) assumes that w is reduced except
    at the coordinate j."""
    H = hnf_rows([list(g) for g in gens], n)
    piv = [next(j for j in range(n) if r[j]) for r in H]

    def red(w, start=0):
        w = list(w)
        for r, p in zip(H, piv):
            if p >= start:
                c = w[p] // r[p]
                if c:
                    for j in range(p, n):
                        w[j] -= c * r[j]
        return tuple(w)
    order = None
    if len(H) == n:
        order = 1
        for r, p in zip(H, piv):
            order *= r[p]
    return red, len(H), order


def coset_min(gens, base, n):
    """the least L1 norm of a vector of base + M, M the span of gens in Z^n (exact): breadth-first search from 0 in
    Z^n / M with the steps +-e_j until the class of base is reached (at the latest at depth |base|_1); the depth
    reached is the least number of unit steps adding up to an element of that class.  Returns (minimum, rank of M,
    order of Z^n / M or None, number of classes visited)."""
    red, rk, order = hnf_reducer(gens, n)
    target = red(base)
    start = red([0] * n)
    dist = {start: 0}
    queue = collections.deque([start])
    while queue:
        x = queue.popleft()
        if x == target:
            return dist[x], rk, order, len(dist)
        for j in range(n):
            for s in (1, -1):
                y = list(x)
                y[j] += s
                y = red(y, j)
                if y not in dist:
                    dist[y] = dist[x] + 1
                    queue.append(y)
    raise AssertionError("the class of base is not reached")


def odd_coset(vectors, f):
    """for a basis of a lattice and a functional f: generators of the kernel {v : f(v) even} (the vectors with f
    even, the differences with one vector x1 with f odd, and 2 x1)."""
    par = [sum(a * b for a, b in zip(f, v)) % 2 for v in vectors]
    i0 = par.index(1)
    x1 = vectors[i0]
    K0 = [v if p == 0 else [a - b for a, b in zip(v, x1)] for i, (v, p) in enumerate(zip(vectors, par)) if i != i0]
    return K0 + [[2 * a for a in x1]]


def level_minimum(K0, base, idx):
    """the least number of entries at the coordinates idx (one order) over base + span(K0): coset_min of the
    projections."""
    return coset_min([[v[i] for i in idx] for v in K0], [base[i] for i in idx], len(idx))


def legendre(a, l):
    """chi_l(a): the Legendre symbol for l odd, chi_4 for l = 4."""
    if l == 4:
        return 1 if a % 4 == 1 else -1
    return 1 if pow(a % l, (l - 1) // 2, l) == 1 else -1


def check_AO():
    rows, ok = [], True
    # (i) 420: the element of length 15
    q = 420
    D = Degree(q)
    hodge, prim, odd = product_element_check(q, XI420, HALF420)
    lc = level_counts(q, XI420)
    good = hodge and prim and odd and len(XI420) == 15 and lc == {15: 1, 35: 1, 60: 1, 105: 3, 140: 3, 420: 6}
    rows.append({"q": 420, "element": XI420, "hodge": hodge, "primitive_part_product": prim, "eps": odd,
                 "length": len(XI420), "levels": lc})
    print("    (AO) 420: element of length 15 (+ 210): Hodge %s, primitive part prod H %s, every K-coset odd %s, "
          "levels %s: %s" % (hodge, prim, odd, lc, good), flush=True)
    ok &= good
    # (ii) 420: exact per-level minima.  The elements of B_420 / D_420 with eps = 1 are the coset XI420 + ker eps of
    # the Z-basis; for each order d the least number of entries of order d on it is the least L1 norm in the coset
    # Proj_d(XI420) + Proj_d(ker eps) (coset_min).  At the top Z^48 / Proj_420(ker eps) is infinite, and the minimum
    # is bounded below by the relaxation: the vectors satisfying (I_{3,5,7}) alone with the K-coset of 1 odd, a set
    # containing the elements of B_420 with eps = 1 (Theorem 50 and Proposition 51(i)), where the group is finite.
    # The same search with only the relations listed in MINSETS420 gives the relaxed minima of NOTE 55(c).
    L = lattice(q)
    Kp = L.Kp
    v0 = tuple_vec(XI420, q, Kp)
    key0 = coset_key(D, 1)
    cl = [1 if coset_key(D, u) == key0 else 0 for u in Kp]
    eps_v0 = sum(a * b for a, b in zip(cl, v0)) % 2
    K0 = odd_coset(L.Lb, cl)
    blocks = relation_matrix(D, Kp)
    labels = relation_labels(D)
    v0_rel = is_zero(relations(D, odd_parts(D, v0, Kp)))
    expect = {15: 1, 35: 1, 60: 1, 105: 3, 140: 3, 420: 6}
    per, total, good = {}, 0, eps_v0 == 1 and v0_rel
    for d in sorted({q // gcd(u, q) for u in Kp}):
        idx = [i for i, u in enumerate(Kp) if q // gcd(u, q) == d]
        row = {"coordinates": len(idx), "element": lc.get(d, 0)}
        if d != q:
            mn, rk, order, seen = level_minimum(K0, v0, idx)
            row.update({"min": mn, "rank": rk, "group_order": order, "classes_visited": seen})
            good &= mn == expect.get(d, 0)
        if d in MINSETS420:
            keep = MINSETS420[d]
            rel = [[int(x) for x in r] for b, lab in zip(blocks, labels) if lab in keep for r in b if any(r)]
            Kr = odd_coset(integer_kernel(rel, len(Kp)), cl)
            mr, rkr, orr, seenr = level_minimum(Kr, v0, idx)
            row.update({"relations": keep, "min_relaxed": mr, "rank_relaxed": rkr, "group_order_relaxed": orr})
            good &= mr == expect[d] and orr is not None
            if d == q:
                mn = mr                                 # lower bound, attained by XI420
                _, rk, order = hnf_reducer([[v[i] for i in idx] for v in K0], len(idx))
                row.update({"min": mn, "rank": rk, "group_order": order})
                good &= mr == lc[d]
            print("    (AO) 420, order %d: least number of entries with eps = 1: %d (exact; Z^%d / "
                  "Proj_d(ker eps): rank %d, order %s%s); with only %s and one odd K-coset: %d (group of order "
                  "%s); XI420: %d" % (d, mn, len(idx), rk, order or "infinite",
                                      "" if d != q else ", bounded below by the relaxation", keep, mr, orr, lc[d]),
                  flush=True)
        total += mn
        per[d] = row
    zero_orders = [d for d, r in per.items() if d not in expect and r.get("min") == 0]
    good &= total == 15 == len(XI420)
    ok &= good
    rows.append({"q": 420, "per_level": per, "sum_of_minima": total, "xi420_eps": eps_v0,
                 "xi420_satisfies_relations": v0_rel, "ok": good})
    print("    (AO) 420: least number 0 at the other %d orders below 420 %s; sum of the per-level minima = %d = "
          "the length of XI420: %s" % (len(zero_orders), zero_orders, total, good), flush=True)
    # (iii) 420, the proof of Theorem 55(c) at the orders 105 and 140: the count F_d of the entries (q/d) a' of
    # order d with the signs chi_3 chi_5 (a'), resp. chi_4 chi_5 (a'), vanishes mod 3 on B_420; lambda_d = eps; and
    # Z^24 / Proj_d(ker eps) has order 6, so (lambda_d, F_d) maps it isomorphically onto Z/2 x Z/3 = Z/6, with the
    # unit vectors at (1, +-1) = +-1 and the class eps = 1 (XI420: lambda_d = 1, F_d = 0) at 3
    good = True
    for d, (l1, l2) in ((105, (3, 5)), (140, (4, 5))):
        idx = [i for i, u in enumerate(Kp) if q // gcd(u, q) == d]
        sgn = [legendre(Kp[i] // (q // d), l1) * legendre(Kp[i] // (q // d), l2) for i in idx]
        F3 = all(sum(s * v[i] for s, i in zip(sgn, idx)) % 3 == 0 for v in L.Lb)
        lam = all(sum(v[i] for i in idx) % 2 == sum(a * b for a, b in zip(cl, v)) % 2 for v in L.Lb)
        _, rk, order = hnf_reducer([[v[i] for i in idx] for v in K0], len(idx))
        at_xi = (sum(v0[i] for i in idx) % 2, sum(s * v0[i] for s, i in zip(sgn, idx)) % 3)
        g3 = F3 and lam and rk == len(idx) == 24 and order == 6 and at_xi == (1, 0)
        good &= g3
        rows.append({"q": 420, "order": d, "signs": "chi_%d chi_%d" % (l1, l2), "F_vanishes_mod_3": F3,
                     "lambda_equals_eps": lam, "rank": rk, "group_order": order, "xi420": at_xi, "ok": g3})
        print("    (AO) 420, order %d: F = sum of chi_%d chi_%d (a') over the entries vanishes mod 3 on B_420 %s, "
              "lambda_%d = eps %s, Z^24 / Proj_d(ker eps) of order %s (so Z/6, unit vectors at +-1, eps = 1 at 3), "
              "XI420 at (lambda, F) = %s: %s" % (d, l1, l2, F3, d, lam, order, at_xi, g3), flush=True)
    ok &= good
    # (iv) 420, the class of XI420: the parities of the numbers of entries of orders 3, 4, 5, 7, 12, 21, 28 kill
    # D + S (every lifted standard tuple), have rank 7 together with eps on B_420, so that with eps they separate
    # the 128 classes of B_420 / (D + S) = (Z/2)^7 ((AN)), and vanish on XI420.  So the class of XI420 is the only
    # class with eps = 1 in which an element can have all its entries at the six orders of (ii)
    seven = (3, 4, 5, 7, 12, 21, 28)
    cnt = {k: [1 if q // gcd(u, q) == k else 0 for u in Kp] for k in seven}
    kill7 = all(L.kills_standard(f) for f in cnt.values())
    rk8 = rank2([L.f2(cl)] + [L.f2(f) for f in cnt.values()])
    xi_even = all(lc.get(k, 0) % 2 == 0 for k in seven)
    lifts = {}
    for E in itertools.combinations(D.fac, 2):
        vs = {tuple(sum(f[i] * v[i] for i in range(L.m)) % 2 for f in cnt.values()) for v in L.lift(E[0] * E[1])}
        lifts["%d,%d" % E] = sorted(vs)
    good = kill7 and rk8 == 7 and xi_even
    ok &= good
    rows.append({"q": 420, "seven_parities": seven, "kill_D_S": kill7, "rank_with_eps": rk8,
                 "xi420_all_even": xi_even, "parity_vectors_of_lifted_bases": lifts, "ok": good})
    print("    (AO) 420: the parities of the counts of orders %s kill D + S %s, rank with eps on B_420 %d, all "
          "even on XI420 %s (lifted classes: %s): %s" % (seven, kill7, rk8, xi_even,
                                                 {E: [v for v in vs if any(v)] for E, vs in lifts.items()}, good),
          flush=True)
    # (v) the lower bound of Theorem 55(a), and the product elements of 660, 780, 924, 1155
    for q, xi, H in ((420, XI420, HALF420), (660, XI660, HALF660), (780, XI780, HALF780),
                     (1155, XI1155, HALF1155), (924, XI924, HALF924), (1020, (), None), (1140, (), None),
                     (1365, (), None), (1428, (), None), (1540, (), None)):
        fac = factors(q)
        lb, top, terms = lower_bound_a(fac)
        row = {"q": q, "P": fac, "lower_bound_a": lb, "top": top, "terms": {str(k): v for k, v in terms.items()}}
        if xi:
            hodge, prim, odd = product_element_check(q, xi, H)
            form = level3_formula(fac, H)
            lc = level_counts(q, xi)
            lev3 = {}
            for k in fac:
                if k == 4:
                    d = q // 4
                    cnt = lc.get(d, 0) + lc.get(2 * d, 0)
                else:
                    cnt = lc.get(q // k, 0)
                lev3[k] = (cnt, str(form[k]))
            # NOTE 55(d): equal to the expression of (b) except at the level 165 = 1155 / 7 (9 against 5)
            diff = {k for k, (c, f) in lev3.items() if c != Fr(f)}
            good = hodge and prim and odd and all(c >= Fr(f) for c, f in lev3.values()) and len(xi) >= lb
            good &= diff == ({7} if q == 1155 else set())
            if q == 924:
                # Theorem 55(e): 4 | q and every odd l = 3 mod 4, so L >= (a) + 1; here (a) = 15 = len - 1
                e_cond = all(l == 4 or l % 4 == 3 for l in fac) and lb == top and lb + 1 == len(xi)
                row["theorem55e_optimal"] = e_cond
                good &= e_cond
            row.update({"element_length": len(xi), "hodge": hodge, "primitive_part_product": prim, "eps": odd,
                        "levels": lc, "level3_counts_vs_formula": {str(k): v for k, v in lev3.items()}})
            ok &= good
            print("    (AO) q=%d: lower bound (a) %d (top %d); product element of length %d: Hodge %s, eps 1 %s, "
                  "levels %s, (r-1)-levels vs formula %s: %s" % (q, lb, top, len(xi), hodge, odd, lc, lev3, good),
                  flush=True)
        else:
            print("    (AO) q=%d: lower bound (a) %d (top %d, terms %s)" % (q, lb, top, terms), flush=True)
        rows.append(row)
    return rows, ok


def lifted_coset_witness(q, l, key, D):
    """Theorem 57(i): the K-coset witness g of level q, extended to level d = l q (l an odd prime dividing q) by
    f(l y) = g(y) on the old points and f(a) = g(a mod q) on the points a prime to l (the new ones)."""
    d = l * q
    f = {}
    for a in range(1, d):
        if 2 * a == d:
            f[a] = 0
        elif a % l == 0:
            f[a] = 1 if coset_key(D, a // l) == key else 0
        else:
            f[a] = 1 if coset_key(D, a % q) == key else 0
    return f


def check_AP():
    rows, ok = [], True
    q, l = 420, 3
    d = l * q
    D = Degree(q)
    keys = sorted({coset_key(D, u) for u in range(1, q)} - {None})
    fs = [lifted_coset_witness(q, l, k, D) for k in keys]
    even = all(f[a] == f[d - a] for f in fs for a in range(1, d))
    stds = []
    for dd in range(3, d + 1):
        if d % dd == 0:
            for s in standard(dd):
                stds.append(tuple((d // dd) * x % d for x in s))
    kill = all(sum(f[a] for a in s) % 2 == 0 for f in fs for s in stds)
    disjoint = all(sum(f[a] for f in fs) <= 1 for a in range(1, d))
    lifted = [l * x for x in XI420]
    ones = all(sum(f[a % d] for a in lifted) % 2 == 1 for f in fs)
    # the lifts of R_{d'} for every divisor d' of d with P(d') != P(q) never meet the supports
    supp = {a for f in fs for a in range(1, d) if f[a]}

    def pset(n):
        return frozenset(([4] if n % 4 == 0 else []) + [p for p in D.odd if n % p == 0])
    small = [d1 for d1 in range(2, d) if d % d1 == 0 and pset(d1) != frozenset(D.fac)]
    avoid = all(all((d // d1) * u % d not in supp for u in range(1, d1)) for d1 in small)
    good = even and kill and disjoint and ones and avoid
    ok &= good
    rows.append({"q": q, "d": d, "witnesses": len(fs), "even": even, "standard_tuples": len(stds), "kill": kill,
                 "disjoint": disjoint, "one_on_lift": ones, "avoid_two_factor_lifts": avoid})
    print("    (AP) q=420, d=1260: %d extended coset witnesses: even %s, kill %d standard tuples %s, disjoint %s, "
          "1 on the lift of the element of length 15 %s, avoid the lifts of R_d' (d' | d, P(d') != P(q); %d "
          "divisors) %s: %s" % (len(fs), even, len(stds), kill, disjoint, ones, len(small), avoid, good),
          flush=True)
    return rows, ok


def count_parity(L, D, order_set):
    """the functional alpha -> (number of entries with order in order_set) mod 2 on the generators of B_q."""
    f = [1 if (D.q // gcd(u, D.q)) in order_set else 0 for u in L.Kp]
    return L.f2(f)


def check_AQ():
    rows, ok = [], True
    for q in (420, 660, 780, 924, 1155):
        D = Degree(q)
        L = lattice(q)
        eps = L.f2(next(iter(coset_witnesses(D, L.Kp).values())))
        zero = tuple(0 for _ in eps)
        lb, top, terms = lower_bound_a(D.fac)
        res = {}
        good = any(eps)
        for k in D.fac:
            e, idx = terms[k]
            d = q // 4 if k == 4 else q // k
            pred = eps if (e * idx) % 2 else zero
            got = count_parity(L, D, {d})
            res[str(k)] = {"order": d, "e": e, "index": idx, "count_equals_eps": got == eps,
                           "predicted": got == pred}
            good &= got == pred
        # lambda_q = N_{P(q)} eps_q (the primitive entries are the union of the N_{P(q)} cosets of K)
        lam_top = count_parity(L, D, {q}) == (eps if top % 2 else zero)
        good &= lam_top
        extra = {}
        if q == 420:
            for d in (15, 35, 60, 105, 140):
                extra[d] = count_parity(L, D, {d}) == eps
            for ds in ({20}, {84}, {420}, {4, 12, 28}):
                extra[str(sorted(ds))] = count_parity(L, D, ds) == zero
            # all combinations of counts by orders vanishing on B_420: dimension 8, spanned by the listed ones
            orders = sorted({q // gcd(u, q) for u in L.Kp})
            N, dim = flint.nmod_mat([list(count_parity(L, D, {d})) for d in orders], 2).transpose().nullspace()
            listed = [{20}, {84}, {420}, {4, 12, 28}] + [{15, d} for d in (35, 60, 105, 140)]
            lvan = all(count_parity(L, D, s) == zero for s in listed)
            lrk = rank2([[1 if d in s else 0 for d in orders] for s in listed])
            extra["dimension_of_vanishing_combinations"] = dim
            extra["spanned_by_listed"] = lvan and lrk == dim == 8
            good &= all(v for k, v in extra.items() if k != "dimension_of_vanishing_combinations")
        ok &= good
        rows.append({"q": q, "levels": res, "N": top, "lambda_q_equals_N_eps": lam_top,
                     "basis": "2-saturated" if q in SATURATION else "Z-basis",
                     "420_counts": {str(k): v for k, v in extra.items()}, "ok": good})
        tail = ""
        if q == 420:
            tail = ("; at 420 eps = lambda_15 = lambda_35 = lambda_60 = lambda_105 = lambda_140, and lambda_20, "
                    "lambda_84, lambda_420, lambda_4 + lambda_12 + lambda_28 vanish: %s; the combinations of "
                    "counts vanishing on B_420 have dimension %d, spanned by these and lambda_15 + lambda_d: %s"
                    % (all(extra[d] for d in (15, 35, 60, 105, 140)), extra["dimension_of_vanishing_combinations"],
                       extra["spanned_by_listed"]))
        print("    (AQ) q=%d (%s): lambda_{q/l} = e_l [G:Gamma] eps_q on B_q for every l in P(q), and lambda_q = "
              "N eps_q (N = %d): %s %s%s" % (q, "2-saturated basis" if q in SATURATION else "Z-basis", top, good,
                                             {k: (v["order"], v["e"], v["index"]) for k, v in res.items()}, tail),
              flush=True)
    # cyclic covers of degree 420: eps(gamma_e) = 0 for every e (Corollary 58)
    import composite as CO
    rng = random.Random(420)
    q = 420
    D = Degree(q)
    G = units(q)
    fams = bad = nvec = 0
    for N in (4, 4, 4, 6, 6, 6, 8, 8):
        for _ in range(3):
            while True:
                a = [rng.choice(G) for _ in range(N - 1)]
                last = (-sum(a)) % q
                if gcd(last, q) == 1:
                    a.append(last)
                    break
            K, Lf = CO.det_lattice(q, a)
            fams += 1
            for e in Lf:
                g = CO.gamma(q, a, K, e)
                nvec += 1
                cnt = eps_of_tuple(D, g)
                bad += any(cnt.values())
    good = bad == 0 and fams > 0
    ok &= good
    rows.append({"cyclic_covers_420": {"families": fams, "basis_vectors": nvec, "with_eps_1": bad}})
    print("    (AQ) cyclic covers of degree 420: %d random families (N = 4, 6, 8), %d lattice vectors, "
          "eps(gamma_e) = 1 for %d: %s" % (fams, nvec, bad, good), flush=True)
    return rows, ok


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fourprimes.json"
    only = sys.argv[2:] if len(sys.argv) > 2 else None
    res, allok, secs = {}, True, {}
    for key, fn in (("AM", check_AM), ("AN", check_AN), ("AO", check_AO), ("AP", check_AP), ("AQ", check_AQ)):
        if only and key not in only:
            continue
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
