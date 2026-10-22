"""
xi420.py -- checks for ../NOTE.txt, Section 13 ([MMRV]'s exceptional element xi_420, the classes of the Hodge
elements with a given product of half systems as primitive part, lifted lengths at 420, and the counts equal to
eps_q in the degrees of NOTE Corollary 58 without a count).  The labels continue at (AR) ((AQ) is the last label of
fourprimes.py).

Notation as in NOTE Sections 12, 13: q in Q, P(q) its factors (4 and odd primes), mu_l = l (l odd) and mu_4 = 2,
R_f / D_f represented on the units u of Z/f with 2u < f ((x) = -(f - x), (f/2) = 0, as in [MMRV] Rem. 4.6), the
d-part xi_{q,d} in R_{q/d} / D_{q/d} (the entries of gcd d), and eps_q the parity of the number of primitive
entries in one coset of K = {+-1}^4 (NOTE Proposition 51).  The seven parities are the numbers of entries of
orders 3, 4, 5, 7, 12, 21, 28 mod 2 (NOTE Theorem 55(c)); a class of B_420 / (D + S) with eps = 1 is named by
its vector pi of these parities (it satisfies pi_4 + pi_12 + pi_28 = 0).

 (AR) [MMRV] Def. 4.8: the recursion phi(d) xi_{q,d} = sum_{e | d, e < d} tau_d(xi_{q,e}) with tau_d of [MMRV]
      (24), in five readings: as printed; with tau_d replaced by -tau_d (the reading that NOTE Lemma 2 and
      Corollary 13' use for two factors); as printed with all d-parts negated at the end; and the first two with
      phi(q)/phi(q/d) in place of phi(d).  (i) For the 97 degrees l p, 4p <= 400 (l <= 19) with Aoki's half
      systems: the reading with -tau_d is integral and Hodge, and equal to parity.py's xi_q for odd q; the printed
      reading is never Hodge; negating at the end is Hodge exactly for odd q.  (ii) At 420, for all 128 products
      of half systems and all five readings, some division has no solution (first at d = 3, resp. at d = 2 for
      the readings with phi(q)/phi(q/d)); the failing levels for Aoki's half systems (all primitive roots).
      (iii) The criterion of NOTE Proposition 59(a) for the first divisions (d = l, and d = 4 for l = 4) against
      the direct computation: (a) for every product of half systems at 420, 660, 780, 924, 1155, where no first
      division is exact for any product (so (a) tests only the "only if" direction there); (b) the four-factor
      degrees <= 3036 at which some first division can be exact are 1596, 2415, 2508, 2604, 2660, 3036, and at
      each of them 30 products (seeded, half of them built invariant at such a level) give both outcomes, in
      agreement with the criterion at all four first levels.  (iv) None of the 579 degrees q <= 20000 with four
      factors satisfies the odd-order condition of Proposition 59(c); 3170089 = 13 . 43 . 53 . 107 does, and for
      the half systems of its proof the reading with -tau_d is integral at every level.  (v) The reading with
      -tau_d, solved over Q, is Hodge at 420, 660, 780, 924, 1155 with Aoki's half systems for all primitive
      roots (denominators 4); the printed and the negated readings are not (least primitive roots).
 (AS) The classes at 420 (NOTE Theorems 61, 62): the 128 products of half systems form two orbits under
      (Z/420)^x (32 with H_7 the squares or the non-squares mod 7, and 96); each is the primitive part of a Hodge
      element; the Hodge elements without primitive entries meet every class of ker(eps) / (D + S) (the seven
      parities have rank 6 on them), so the completions of every product meet all 64 classes with eps = 1; a
      completion of length 17 of Aoki's product (least primitive roots), with the counts of NOTE Theorem 55(b),
      (c) and all seven parities even; for each of the 64 classes with eps = 1 an explicit Hodge element of
      length 15 + w (w the weight of pi) if pi_28 = 0 and 17 + w if pi_28 = 1.
 (AT) Lifts (NOTE Theorem 63): the six coset witnesses and the order counts at 105, 140, 60, 35, 15 (eleven
      witnesses with disjoint supports, each equal to eps on B_420) and the mod-3 counts F_105, F_140 of NOTE
      Theorem 55(c) (mod-3 witnesses, 0 mod 3 on B_420, +-1 exactly on the supports of lambda_105, lambda_140);
      their extensions to the levels 1260, 2100, 2940 (NOTE Theorem 57(i)), 840 (explicit) and 420 r for r = 11,
      13, 17, 19, 23 (the functions Psi; the mod-3 counts extended by 0): the eleven are even, kill every standard
      tuple of every degree dividing the level (lifted), have disjoint supports and are 1 on the lift of XI420;
      which mod-3 counts extend (for 420 r: iff the character of the count is 1 at r); Psi exists for each of
      the six coset witnesses and every class of r mod 420, and the homogeneous solutions have dimension 8 in
      all 576 systems; the resulting lower bounds 13, 13, 11, 15 at 840, 1260, 2100, 2940 and 11, 11, 13, 13,
      15 at 420 r.
 (AU) Counts equal to eps_q (NOTE Proposition 64, Corollary 65): at 420, 660, 780, 924, 1155 (Z-basis at 420,
      else 2-saturated bases) and at 1740, 2460, 2652, the first three degrees with four factors, N_{P(q)} even
      and every e_l [G_S : Gamma_l] even (the first five are listed): the orders d with lambda_d = eps_q on B_q,
      those with lambda_d = 0, and at the levels with two factors the dichotomy of Proposition 64.
Usage:  python -I xi420.py OUT.json [labels]      Expected last line "ALL CHECKS PASS: True".  About 7 minutes
(the 2-saturated bases at 2460 and 2652 take most of it).  Needs python-flint and numpy, and
../../{cyclic-covers, fermat-frontier, ts-powers}/code.  Integers and rationals only: the explicit elements below
were found by integer programs (HiGHS, floating point), which are not run here; this script checks them exactly,
and the lower bounds that make some of them least come from the NOTE.
MIT licence (../../LICENSE).
"""

import itertools
import json
import os
import random
import sys
import time
from fractions import Fraction as Fr
from math import gcd

import flint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fourprimes as FP  # noqa: E402  (adds ../../{cyclic-covers, fermat-frontier, ts-powers}/code to the path)
import parity as PA  # noqa: E402
from v2_exotic import integer_kernel, is_hodge, standard  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# [MMRV] Def. 4.8

def isprime(n):
    return n > 1 and all(n % r for r in range(2, int(n ** 0.5) + 1))


def phi(n):
    out, m, p = n, n, 2
    while p * p <= m:
        if m % p == 0:
            out -= out // p
            while m % p == 0:
                m //= p
        p += 1
    if m > 1:
        out -= out // m
    return out


def primes_of(n):
    out, p = [], 2
    while p * p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0:
                n //= p
        p += 1
    return out + ([n] if n > 1 else [])


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def add(v, x, f, c):
    """v += c (x)_f in R_f / D_f (on the residues u with 0 < 2u < f; (x) = -(f - x), (f/2) = 0)."""
    x %= f
    if c == 0 or 2 * x == f:
        return
    if 2 * x > f:
        x, c = f - x, -c
    v[x] = v.get(x, 0) + c
    if v[x] == 0:
        del v[x]


def product_part(q, H):
    """the primitive part prod_l H_l (Def. 4.7: the residues with the given components)."""
    fac = FP.factors(q)
    v = {}
    for c in itertools.product(*[H[l] for l in fac]):
        add(v, FP.crt_all(list(c), fac), q, 1)
    return v


def tau(q, d, e, part):
    """tau_d of [MMRV] (24) on the lift of the e-part (a dict on the units mod q/e): an entry (e a)_q has q' = q/e,
    d' = d/e, a' = a, factor phi(q)/phi(q') and g = prod over the primes p | d', p not dividing q/d, of
    (1) + (-p^-1), acting on (a' mod q/d)."""
    f = q // d
    fac = phi(q) // phi(q // e)
    assert fac * phi(q // e) == phi(q)
    ps = [p for p in primes_of(d // e) if f % p]
    mults = []
    for k in range(len(ps) + 1):
        for sub in itertools.combinations(ps, k):
            m = 1
            for p in sub:
                m = m * (-pow(p, -1, f)) % f
            mults.append(m)
    out = {}
    for a, c in part.items():
        for m in mults:
            add(out, m * a, f, c * fac)
    return out


READINGS = (("printed", 1, "phi(d)", False), ("-tau", -1, "phi(d)", False), ("negated", 1, "phi(d)", True),
            ("printed, phi(q)/phi(q/d)", 1, "ratio", False), ("-tau, phi(q)/phi(q/d)", -1, "ratio", False))


def recursion(q, P, sign, coef, negate):
    """the d-parts (over Q) and the list of levels d where phi(d) xi_{q,d} = sign sum tau_d(xi_{q,e}) has no
    solution in R_{q/d} / D_{q/d}."""
    ds = [d for d in divisors(q) if d < q]
    parts = {1: dict(P)}
    fails = []
    for d in ds[1:]:
        f = q // d
        S = {}
        for e in ds:
            if e < d and d % e == 0 and parts[e]:
                for u, c in tau(q, d, e, parts[e]).items():
                    add(S, u, f, c)
        c = phi(d) if coef == "phi(d)" else phi(q) // phi(q // d)
        if any(Fr(s) % c for s in S.values()):
            fails.append(d)
        parts[d] = {u: sign * Fr(s, c) for u, s in S.items()}
    if negate:
        for d in ds[1:]:
            parts[d] = {u: -c for u, c in parts[d].items()}
    return parts, fails


def assemble(q, parts):
    v = {}
    for d, part in parts.items():
        for u, c in part.items():
            add(v, d * u, q, c)
    return v


def hodge_vec(q, v):
    """the element v of R_q / D_q (rational coefficients allowed) satisfies the Hodge condition: sum over the
    entries of {t a / q} - 1/2 vanishes for every unit t."""
    for t in range(1, q):
        if gcd(t, q) == 1:
            s = 0
            for u, c in v.items():
                k = t * u % q
                if 2 * k != q:
                    s += c * (q - 2 * k)
            if s:
                return False
    return True


def half_systems(l):
    if l == 4:
        return [[1], [3]]
    reps = [x for x in range(1, l) if 2 * x < l]
    return [sorted(s * x % l for s, x in zip(signs, reps))
            for signs in itertools.product((1, -1), repeat=len(reps))]


def aoki_half(l, mu):
    return [1] if l in (3, 4) else sorted(pow(mu, j, l) for j in range((l - 1) // 2))


def primitive_roots(l):
    return [g for g in range(2, l) if len({pow(g, k, l) for k in range(1, l)}) == l - 1]


def first_division_exact(q, P, l):
    """whether the first division at the element l of P(q) has a solution (d = l for l odd; for l = 4 the level
    d = 4, after xi_{q,2} = -tau_2(xi_{q,1})).  The odd factor (1 -+ 2) at d = 4 does not matter."""
    if l != 4:
        S = tau(q, l, 1, P)
        return all(s % phi(l) == 0 for s in S.values())
    x2 = {u: -c for u, c in tau(q, 2, 1, P).items()}
    S = tau(q, 4, 1, P)
    for u, c in tau(q, 4, 2, x2).items():
        add(S, u, q // 4, c)
    return all(s % 2 == 0 for s in S.values())


def criterion(fac, H, l):
    """NOTE Proposition 59(a): mu_l H_l' = sigma H_l' for every l' != l, with one sign sigma."""
    mu = 2 if l == 4 else l
    for s in (1, -1):
        if all(sorted(mu * h % lp for h in H[lp]) == sorted(s * h % lp for h in H[lp]) for lp in fac if lp != l):
            return True
    return False


def odd_order(x, l):
    x %= l
    o, y = 1, x
    while y != 1:
        y, o = y * x % l, o + 1
    return o % 2 == 1


def odd_order_condition(fac):
    """Proposition 59(c): for every l there is a sign sigma with sigma mu_l of odd order mod every other l'."""
    return all(any(all(odd_order(s * (2 if l == 4 else l), lp) for lp in fac if lp != l) for s in (1, -1))
               for l in fac)


def first_level_signs(fac, l):
    """the signs sigma with sigma mu_l of odd order mod every l' != l: by the proof of Proposition 59(c), some
    product of half systems makes the first division at l exact iff this list is nonempty."""
    mu = 2 if l == 4 else l
    return [s for s in (1, -1) if all(odd_order(s * mu, lp) for lp in fac if lp != l)]


def four_factor_degrees(bound):
    """the q <= bound with four factors in P = {4} + {odd primes}, as (q, factors), sorted by q."""
    P_ = sorted([4] + [p for p in range(3, bound // 60 + 1) if isprime(p)])
    out = []

    def grow(start, S, m):
        if len(S) == 4:
            out.append((m, S))
            return
        for i in range(start, len(P_)):
            if m * P_[i] ** (4 - len(S)) > bound:
                break
            grow(i + 1, S + (P_[i],), m * P_[i])
    grow(0, (), 1)
    return sorted(out)


def random_half(l, rng):
    if l == 4:
        return [rng.choice((1, 3))]
    return sorted(x if rng.random() < 0.5 else l - x for x in range(1, (l + 1) // 2))


def random_invariant_half(lp, x, rng):
    """a random half system of G_lp stable under multiplication by x (x of odd order mod lp)."""
    if lp in (3, 4):
        return random_half(lp, rng)
    grp, y = {1}, x % lp
    while y != 1:
        grp.add(y)
        y = y * x % lp
    seen, H = set(), []
    for a in range(1, lp):
        if a not in seen:
            orb = {a * g % lp for g in grp}
            neg = {(-b) % lp for b in orb}
            seen |= orb | neg
            H += sorted(orb if rng.random() < 0.5 else neg)
    return sorted(H)


def invariant_half_systems(fac):
    """half systems H_l' invariant under Gamma_l' = <sigma_l mu_l : l != l'> (proof of Proposition 59(c))."""
    sig = {l: next(s for s in (1, -1) if all(odd_order(s * (2 if l == 4 else l), lp) for lp in fac if lp != l))
           for l in fac}
    H = {}
    for lp in fac:
        gens = [(sig[l] * (2 if l == 4 else l)) % lp for l in fac if l != lp]
        grp, fr = {1}, [1]
        while fr:
            a = fr.pop()
            for g in gens:
                b = a * g % lp
                if b not in grp:
                    grp.add(b)
                    fr.append(b)
        seen, Hl = set(), []
        for x in range(1, lp):
            if x not in seen:
                orb = {x * g % lp for g in grp}
                seen |= orb | {(-y) % lp for y in orb}
                Hl += sorted(orb)
        H[lp] = sorted(Hl)
    return H


def check_AR():
    rows, ok = {}, True
    # (i) two factors
    bad, n = [], 0
    for l, p in PA.degrees():
        q = l * p
        H = {x: ([1] if x == 4 else PA.half(x)) for x in (l, p)}
        P = product_part(q, H)
        res = {}
        for name, s, c, neg in READINGS[:3]:
            parts, fails = recursion(q, P, s, c, neg)
            res[name] = (not fails, hodge_vec(q, assemble(q, parts)))
        good = res["-tau"] == (True, True) and not res["printed"][1] and res["negated"][1] == (l != 4)
        if l != 4:
            prim, pl, pp = PA.xi_odd(l, p)
            w = {}
            for a in PA.as_tuple(q, [(1, q, prim), (p, l, pl), (l, p, pp)]):
                add(w, a, q, 1)
            good &= w == {u: int(c) for u, c in assemble(q, recursion(q, P, -1, "phi(d)", False)[0]).items()}
        n += 1
        if not good:
            bad.append(q)
    ok &= not bad
    rows["two_factors"] = {"degrees": n, "failures": bad}
    print("    (AR) (i) %d degrees l p, 4p <= 400: -tau integral, Hodge and equal to parity.py's xi_q (odd q); "
          "printed never Hodge; negated Hodge iff q odd: %s" % (n, not bad), flush=True)
    # (ii) 420: all products of half systems, all readings
    q = 420
    fac = FP.factors(q)
    first, allfail, cnt = {}, True, 0
    for Hs in itertools.product(*[half_systems(l) for l in fac]):
        P = product_part(q, dict(zip(fac, Hs)))
        cnt += 1
        for name, s, c, neg in READINGS:
            fails = recursion(q, P, s, c, neg)[1]
            allfail &= bool(fails)
            first.setdefault(name, set()).add(fails[0] if fails else None)
    aoki = {}
    for m5 in primitive_roots(5):
        for m7 in primitive_roots(7):
            H = {4: [1], 3: [1], 5: aoki_half(5, m5), 7: aoki_half(7, m7)}
            aoki["%d,%d" % (m5, m7)] = recursion(q, product_part(q, H), -1, "phi(d)", False)[1]
    want_first = {name: ({2} if c == "ratio" else {3}) for name, s, c, neg in READINGS}
    lv = [3, 4, 5, 7, 12, 15, 20, 21, 28, 60, 84]
    good = cnt == 128 and allfail and first == want_first and all(v == lv for v in aoki.values())
    ok &= good
    rows["420"] = {"products": cnt, "all_fail": allfail, "first_failing_level": {k: sorted(v) for k, v in
                                                                                 first.items()},
                   "aoki_failing_levels": aoki, "ok": good}
    print("    (AR) (ii) 420: %d products x 5 readings, every one has a division without solution: %s; first "
          "failing level %s; Aoki's half systems (mu_5, mu_7 = %s), -tau: failing levels %s: %s" % (
              cnt, allfail, {k: sorted(v) for k, v in first.items()}, sorted(aoki), lv, good), flush=True)
    # (iii)(a) the criterion at the first divisions, all products, five degrees (no first division exact)
    crit_rows, gc = {}, True
    for q in (420, 660, 780, 924, 1155):
        fac = FP.factors(q)
        agree, nprod, anyall, nex = True, 0, False, [0] * len(fac)
        for Hs in itertools.product(*[half_systems(l) for l in fac]):
            H = dict(zip(fac, Hs))
            P = product_part(q, H)
            ex = [first_division_exact(q, P, l) for l in fac]
            agree &= ex == [criterion(fac, H, l) for l in fac]
            anyall |= all(ex)
            nex = [n + e for n, e in zip(nex, ex)]
            nprod += 1
        passable = [l for l in fac if first_level_signs(fac, l)]
        g = agree and not anyall and not odd_order_condition(fac) and not any(nex) and not passable
        gc &= g
        crit_rows[q] = {"products": nprod, "criterion_agrees": agree, "some_product_exact_at_all_first_levels":
                        anyall, "odd_order_condition": odd_order_condition(fac),
                        "exact_first_divisions": dict(zip(fac, nex))}
        print("    (AR) (iii)(a) q=%d: %d products; exactness of the first divisions = criterion: %s; number of "
              "exact first divisions at l = %s: %s (none, so only the 'only if' direction is tested)" % (
                  q, nprod, agree, fac, nex), flush=True)
    # (iii)(b) degrees where some first division can be exact: both outcomes occur
    pos = [(m, S) for m, S in four_factor_degrees(3036) if any(first_level_signs(S, l) for l in S)]
    gpos = [m for m, S in pos] == [1596, 2415, 2508, 2604, 2660, 3036]
    rng = random.Random(59)
    for m, S in pos:
        lv = [l for l in S if first_level_signs(S, l)]
        agree, nex = True, {l: 0 for l in S}
        for k in range(30):
            H = {l: random_half(l, rng) for l in S}
            if k % 2 == 0:
                l = lv[(k // 2) % len(lv)]
                x = first_level_signs(S, l)[0] * (2 if l == 4 else l)
                H.update({lp: random_invariant_half(lp, x % lp, rng) for lp in S if lp != l})
            P = product_part(m, H)
            for l in S:
                e = first_division_exact(m, P, l)
                agree &= e == criterion(S, H, l)
                nex[l] += e
        g = agree and all(0 < nex[l] < 30 for l in lv) and all(nex[l] == 0 for l in S if l not in lv)
        gpos &= g
        crit_rows[m] = {"factors": S, "first_levels_that_can_pass": lv, "products": 30, "criterion_agrees":
                        agree, "exact_first_divisions": nex}
        print("    (AR) (iii)(b) q=%d = %s: first levels that can be exact %s; 30 products, exact first divisions "
              "%s; criterion agrees: %s" % (m, ".".join(map(str, S)), lv, nex, agree), flush=True)
    print("    (AR) (iii)(b) the four-factor degrees <= 3036 where some first division can be exact: %s: %s" % (
        [m for m, S in pos], gpos), flush=True)
    ok &= gc and gpos
    rows["criterion"] = crit_rows
    # (iv) the odd-order condition for q with four factors <= 20000, and 3170089
    four = four_factor_degrees(20000)
    sat = [m for m, S in four if odd_order_condition(S)]
    t0 = time.time()
    big = (13, 43, 53, 107)
    Hb = invariant_half_systems(big)
    qb = 13 * 43 * 53 * 107
    Pb = product_part(qb, Hb)
    exact_b = all(criterion(big, Hb, l) for l in big)
    fails_b = recursion(qb, Pb, -1, "phi(d)", False)[1]
    good = not sat and len(four) == 579 and odd_order_condition(big) and exact_b and not fails_b
    ok &= good
    rows["four_factor_degrees"] = {"count_le_20000": len(four), "satisfying": sat, "big": qb,
                                   "big_half_systems_criterion": exact_b, "big_failing_levels": fails_b,
                                   "big_sec": round(time.time() - t0, 1)}
    print("    (AR) (iv) %d degrees with four factors <= 20000, none with the odd-order condition: %s; %d = "
          "13.43.53.107 has it, and the reading -tau is integral at every level for the invariant half systems "
          "(%d primitive entries, %.0f s): %s" % (len(four), not sat, qb, len(Pb), time.time() - t0,
                                                  not fails_b), flush=True)
    # (v) over Q
    rat = {}
    for q in (420, 660, 780, 924, 1155):
        fac = FP.factors(q)
        H = {l: aoki_half(l, primitive_roots(l)[0] if l > 4 else None) for l in fac}
        P = product_part(q, H)
        r = {}
        for name, s, c, neg in READINGS[:3]:
            v = assemble(q, recursion(q, P, s, c, neg)[0])
            r[name] = (hodge_vec(q, v), max(Fr(x).denominator for x in v.values()))
        # the reading with -tau_d for Aoki's half systems with all primitive roots
        allr = set()
        for gs in itertools.product(*[primitive_roots(l) if l > 4 else [None] for l in fac]):
            v = assemble(q, recursion(q, product_part(q, {l: aoki_half(l, g) for l, g in zip(fac, gs)}), -1,
                                      "phi(d)", False)[0])
            allr.add((hodge_vec(q, v), max(Fr(x).denominator for x in v.values())))
        r["-tau, all primitive roots"] = sorted(allr)
        rat[q] = r
    good = all(r["-tau"] == (True, 4) and not r["printed"][0] and not r["negated"][0] and
               r["-tau, all primitive roots"] == [(True, 4)] for r in rat.values())
    ok &= good
    rows["over_Q"] = {str(q): r for q, r in rat.items()}
    print("    (AR) (v) over Q (Hodge, largest denominator): %s: %s" % (rat, good), flush=True)
    return rows, ok


# ---------------------------------------------------------------------------------------------------------------
# the classes at 420

SEVEN = (3, 4, 5, 7, 12, 21, 28)
# a completion of Aoki's product {1} x {1} x {1, 2} x {1, 3, 2} (least primitive roots 2 mod 5, 3 mod 7)
AOKI17 = (1, 8, 37, 121, 125, 157, 183, 212, 241, 249, 261, 287, 288, 304, 337, 364, 395)
AOKI_HALF = {4: [1], 3: [1], 5: [1, 2], 7: [1, 2, 3]}
# for each class with eps = 1 (key: the parities of the orders 3, 4, 5, 7, 12, 21, 28) a Hodge element (found by
# integer programs; checked below), of length 15 + w if the parity at 28 is 0 and 17 + w if it is 1
CLASS_ELEMENTS = """
0000000: 68 87 92 129 179 187 196 203 216 239 248 253 327 359 367
0000010: 20 31 56 91 123 144 173 183 239 277 292 327 337 352 356 359
0000101: 37 69 91 97 112 123 151 156 175 176 225 271 275 284 292 309 349 395 403
0000111: 3 17 75 92 108 139 184 185 187 191 196 269 272 301 303 307 321 325 340 385
0001000: 4 76 89 120 141 183 188 193 196 209 261 288 307 361 371 373
0001010: 12 49 88 88 101 103 169 172 183 224 249 293 353 359 360 380 387
0001101: 28 49 52 55 69 81 107 148 175 179 239 261 283 285 335 344 359 360 384 407
0001111: 35 37 52 55 101 104 119 123 141 172 224 225 271 276 309 320 335 353 360 391 407
0010000: 28 37 87 96 101 119 164 183 223 236 284 336 353 359 363 391
0010010: 1 84 89 99 104 107 112 171 172 223 268 269 340 348 353 413 417
0010101: 7 17 27 28 55 116 132 175 179 187 208 271 303 307 335 336 345 369 391 412
0010111: 1 19 35 56 104 107 123 132 141 201 225 252 268 269 285 337 340 344 353 405 413
0011000: 16 39 41 77 96 179 180 193 196 212 257 297 304 336 359 377 411
0011010: 47 53 60 79 81 97 136 196 251 252 261 303 304 304 311 312 320 413
0011101: 24 43 49 92 151 153 163 165 175 196 225 227 229 248 252 285 291 316 349 360 417
0011111: 7 17 28 52 60 115 132 141 175 179 183 208 225 257 307 336 340 344 359 369 391 395
0100001: 25 31 47 49 65 72 129 195 224 232 239 261 283 284 315 327 401 404 407
0100011: 20 31 49 52 57 105 107 165 167 211 219 224 283 333 335 344 355 359 388 396
0100100: 83 92 105 129 133 143 144 196 207 211 212 229 248 261 383 385 409
0100110: 3 11 19 35 104 107 132 139 224 268 287 315 337 340 344 347 381 387
0101001: 25 27 28 47 109 119 132 167 180 195 229 232 265 268 315 347 349 369 381 416
0101011: 20 53 89 96 105 112 139 155 164 165 173 219 235 241 279 284 297 360 404 407 413
0101100: 11 19 48 139 173 175 180 184 196 207 212 248 293 309 315 343 347 381
0101110: 81 89 96 100 103 112 116 139 175 207 217 261 283 284 284 315 360 361 407
0110001: 16 37 39 45 64 105 161 177 179 196 213 223 244 275 336 353 359 372 391 415
0110011: 25 47 57 79 104 135 167 168 172 199 217 219 224 236 265 315 320 339 347 379 396
0110100: 28 49 53 59 96 113 175 207 211 233 243 261 284 315 328 336 388 401
0110110: 35 36 39 56 73 79 80 83 177 232 251 311 313 315 329 336 412 416 417
0111001: 71 75 75 84 107 128 132 136 159 171 184 191 237 255 269 277 315 360 364 407 413
0111011: 17 28 75 75 77 104 120 132 141 179 195 208 257 289 303 307 315 336 340 344 387 391
0111100: 28 49 88 93 96 99 101 103 175 219 239 283 284 315 336 356 359 360 407
0111110: 37 57 91 128 128 132 175 177 179 180 196 212 219 271 283 293 315 336 391 400
1000000: 32 69 77 140 156 179 183 188 193 212 239 261 307 359 373 392
1000010: 7 23 27 52 101 112 169 208 220 263 280 307 309 344 359 381 408
1000101: 1 11 25 57 121 140 164 187 192 195 224 237 259 263 284 305 356 377 385 417
1000111: 37 45 68 121 129 139 140 155 175 217 220 223 228 235 256 272 309 319 327 392 403
1001000: 16 39 77 96 99 140 179 180 193 212 239 257 304 359 377 392 411
1001010: 56 107 108 109 111 117 140 161 180 223 239 260 268 268 341 347 352 393
1001101: 1 33 88 93 96 99 103 125 191 224 245 255 265 280 283 284 293 343 360 361 388
1001111: 7 12 17 35 88 101 120 165 169 207 208 224 253 257 280 285 340 341 356 363 387 405
1010000: 23 24 84 87 88 109 140 201 203 224 249 307 328 341 367 391 404
1010010: 36 40 51 56 67 84 116 140 177 197 203 299 317 349 409 411 412 416
1010101: 3 37 45 84 89 112 139 140 172 175 199 223 236 283 284 285 309 369 405 408 413
1010111: 37 49 52 56 85 87 151 160 172 183 195 204 236 245 252 280 305 337 359 381 391 403
1011000: 11 96 139 140 164 168 180 187 191 201 224 249 263 284 296 307 309 371
1011010: 23 56 60 139 140 144 176 191 199 201 249 252 260 284 292 307 321 329 367
1011101: 32 55 57 59 67 77 140 165 168 175 215 219 237 239 253 276 317 360 364 368 376 401
1011111: 28 69 75 75 87 101 103 119 140 172 176 223 236 243 255 271 312 320 336 353 360 385 391
1100001: 23 56 75 85 139 140 144 176 191 199 201 249 265 284 292 307 315 321 367 371
1100011: 9 23 75 101 104 104 123 140 195 203 220 224 249 255 289 307 315 344 367 372 391
1100100: 28 47 59 97 105 119 140 156 157 176 281 296 297 299 328 385 393 417
1100110: 23 27 77 100 105 109 112 132 193 232 239 245 268 280 323 359 369 381 416
1101001: 44 69 69 75 75 140 156 176 191 193 209 224 255 257 296 307 309 315 329 360 361
1101011: 33 37 75 85 93 101 104 140 160 161 164 219 224 265 271 284 315 347 360 384 391 407
1101100: 28 33 52 60 105 148 153 164 179 187 203 245 280 307 312 359 367 391 417
1101110: 29 40 60 105 112 149 163 164 167 201 241 245 249 280 287 288 292 309 403 416
1110001: 5 11 23 48 75 83 89 140 153 168 224 251 263 284 315 328 365 371 393 404 417
1110011: 23 40 72 75 77 81 101 124 127 140 205 248 249 265 308 315 321 336 359 367 368 419
1110100: 12 35 56 88 91 101 103 164 169 207 249 280 293 315 336 341 347 387 416
1110110: 8 16 27 105 127 131 137 140 181 188 196 247 252 267 329 340 348 385 387 389
1111001: 37 51 52 56 101 105 120 140 161 164 165 165 219 225 284 312 336 353 359 391 407 417
1111011: 4 68 81 105 107 121 139 140 165 165 180 223 225 243 269 287 303 304 308 320 336 353 384
1111100: 64 68 92 93 105 107 132 139 191 196 199 237 245 252 280 300 333 347 407 413
1111110: 4 8 11 19 37 140 141 175 180 196 209 252 304 309 312 315 320 337 347 381 413
"""


def parse_classes(text):
    out, key = {}, None
    for line in text.strip().splitlines():
        if ":" in line:
            key, rest = line.split(":")
            out[key] = [int(x) for x in rest.split()]
        else:
            out[key] += [int(x) for x in line.split()]
    return out


def tuple_with_half(t, q):
    """t with the entry q/2 added if the Hodge condition needs it (odd length or sum not 0 mod q)."""
    return tuple(t) + ((q // 2,) if (len(t) % 2 or sum(t) % q) else ())


def parities(t, q=420):
    lc = FP.level_counts(q, t)
    return tuple(lc.get(k, 0) % 2 for k in SEVEN)


def eps_one(D, t):
    return all(v == 1 for v in FP.eps_of_tuple(D, t).values())


def check_AS():
    rows, ok = {}, True
    q = 420
    D = FP.Degree(q)
    fac = FP.factors(q)
    G = [t for t in range(1, q) if gcd(t, q) == 1]
    # (i) orbits of the products of half systems under (Z/420)^x
    prods = [tuple(frozenset(h) for h in Hs) for Hs in itertools.product(*[half_systems(l) for l in fac])]
    seen, orbits = set(), []
    for H in prods:
        if H in seen:
            continue
        orb = {tuple(frozenset(t * x % l for x in h) for h, l in zip(H, fac)) for t in G}
        seen |= orb
        orbits.append((len(orb), any(set(h) in ({1, 2, 4}, {3, 5, 6}) for h, l in zip(H, fac) if l == 7)))
    good = sorted(orbits) == [(32, True), (96, False)]
    # (ii) the Hodge elements without primitive entries
    L = FP.lattice(q)
    Kp, Lb, m = L.Kp, L.Lb, L.m
    prim = [i for i, u in enumerate(Kp) if gcd(u, q) == 1]
    order = [q // gcd(u, q) for u in Kp]
    key0 = FP.coset_key(D, 1)
    cl = [1 if FP.coset_key(D, u) == key0 else 0 for u in Kp]
    C = integer_kernel([[Lb[j][i] for j in range(len(Lb))] for i in prim], len(Lb))
    M0 = [[sum(c[j] * Lb[j][i] for j in range(len(Lb)) if c[j]) for i in range(m)] for c in C]
    par = [[sum(v[i] for i in range(m) if order[i] == k) % 2 for k in SEVEN] for v in M0]
    rk = FP.rank2(par)
    eps0 = all(sum(a * b for a, b in zip(cl, v)) % 2 == 0 for v in M0)
    good &= rk == 6 and eps0 and len(M0) == len(Lb) - 41
    # (iii) a completion of every product
    t0 = time.time()
    comp = 0
    for H in prods:
        Pv = [0] * m
        for u, c in product_part(q, dict(zip(fac, [sorted(h) for h in H]))).items():
            Pv[Kp.index(u)] += c
        target = [Pv[i] for i in prim]
        rows_ = [[Lb[j][i] for j in range(len(Lb))] + [-target[k]] for k, i in enumerate(prim)]
        K = integer_kernel(rows_, len(Lb) + 1)
        g = 0
        for k in K:
            g = gcd(g, k[-1])
        comp += g == 1
    good &= comp == 128
    rows["orbits"] = orbits
    rows["M0"] = {"rank": len(M0), "rank_of_seven_parities": rk, "eps_zero": eps0}
    rows["products_with_completion"] = comp
    print("    (AS) (i)-(iii) 420: two orbits of products (sizes, H_7 = squares or non-squares) %s; elements "
          "without primitive entries: rank %d, seven parities of rank %d, eps = 0 %s; products with a Hodge "
          "completion %d of 128 (%.0f s): %s" % (orbits, len(M0), rk, eps0, comp, time.time() - t0, good),
          flush=True)
    ok &= good
    # (iv) the completion of length 17 of Aoki's product
    hodge, prim_ok, odd = FP.product_element_check(q, AOKI17, AOKI_HALF)
    lc = FP.level_counts(q, AOKI17)
    form = {k: str(v) for k, v in FP.level3_formula(fac, AOKI_HALF).items()}
    form_qr = {k: str(v) for k, v in FP.level3_formula(fac, FP.HALF420).items()}
    good = hodge and prim_ok and odd and lc == {15: 1, 35: 1, 60: 1, 84: 2, 105: 3, 140: 3, 420: 6}
    good &= form == {4: "3", 3: "3", 5: "2", 7: "1"} and form_qr == {4: "3", 3: "3", 5: "0", 7: "1"}
    good &= parities(AOKI17) == (0,) * 7 and len(AOKI17) == 17
    ok &= good
    rows["aoki17"] = {"hodge": hodge, "primitive_part": prim_ok, "eps": odd, "levels": lc,
                      "level3_formula": form, "level3_formula_qr": form_qr}
    print("    (AS) (iv) completion of Aoki's product of length 17: Hodge %s, primitive part %s, eps 1 %s, levels "
          "%s, expression of Theorem 55(b) at 105, 140, 84, 60: %s (squares mod 7: %s), seven parities even: %s"
          % (hodge, prim_ok, odd, lc, form, form_qr, good), flush=True)
    # (v) one element per class
    els = parse_classes(CLASS_ELEMENTS)
    seen_cls, bad = set(), []
    for key, t in els.items():
        pi = tuple(int(c) for c in key)
        w = sum(pi)
        tt = tuple_with_half(t, q)
        g = is_hodge(tt, q) and eps_one(D, t) and parities(t) == pi
        g &= len(t) == 15 + w + (2 if pi[6] else 0)
        if not g:
            bad.append(key)
        seen_cls.add(pi)
    want = {p for p in itertools.product((0, 1), repeat=7) if (p[1] + p[4] + p[6]) % 2 == 0}
    good = not bad and seen_cls == want
    ok &= good
    ones = {k: len(t) for k, t in els.items() if sum(int(c) for c in k) <= 1}
    rows["classes"] = {"checked": len(els), "bad": bad, "weight_at_most_one": ones}
    print("    (AS) (v) 64 classes with eps = 1, one Hodge element each of length 15 + w (+2 if odd at 28): %s; "
          "weight <= 1: %s" % (good, ones), flush=True)
    return rows, ok


# ---------------------------------------------------------------------------------------------------------------
# witnesses at the lifts

def lifted_standard(d):
    out = []
    for dd in range(3, d + 1):
        if d % dd == 0:
            for s in standard(dd):
                out.append(tuple((d // dd) * x % d for x in s))
    return out


def base_witnesses(D):
    q = D.q
    keys = sorted({FP.coset_key(D, u) for u in range(1, q)} - {None})
    W = {}
    for k in keys:
        W[("coset",) + k] = [0] + [1 if FP.coset_key(D, a) == k else 0 for a in range(1, q)]
    for d in (105, 140, 60, 35, 15):
        W[("order", d)] = [0] + [1 if q // gcd(a, q) == d else 0 for a in range(1, q)]
    return W


def is_witness(f, d, tuples):
    if d % 2 == 0 and f[d // 2]:
        return False
    if any(f[a] != f[d - a] for a in range(1, d)):
        return False
    return all(sum(f[a] for a in s) % 2 == 0 for s in tuples)


def extend_prime_of_q(g, l, q=420):
    """NOTE Theorem 57(i): f(l y) = g(y), f(a) = g(a mod q) for a prime to l."""
    d = l * q
    return [0] + [g[a // l] if a % l == 0 else g[a % q] for a in range(1, d)]


def extend_840(key, g, q=420):
    """level 840: f(2y) = g(y); at the odd a, f(a) = g(a mod 420) if a = +-3 mod 8 and 0 if a = +-1 mod 8, except
    for the order count at 140, which is extended on the points 3y of order 280 (y a unit mod 280) by
    [y = +-1 mod 8] + [y = +-1 mod 5] mod 2, and is 0 at the other odd points."""
    d = 2 * q
    f = [0] * d
    for a in range(1, d):
        if a % 2 == 0:
            f[a] = g[a // 2]
        elif key == ("order", 140):
            if gcd(a, d) == 3:
                y = a // 3
                f[a] = ((y % 8 in (1, 7)) + (y % 5 in (1, 4))) % 2
        elif a % 8 in (3, 5):
            f[a] = g[a % q]
    return f


def psi_solutions(D, g, rinv):
    """the functions Psi on the primitive residues mod q with Psi(u) + Psi(-u) = g(u) + g(r^-1 u) and an even sum
    on every line (one coordinate varying over (Z/l)^x, l in P(q)): (particular solution, kernel basis) or None."""
    q = D.q
    G = [a for a in range(1, q) if gcd(a, q) == 1]
    gi = {a: i for i, a in enumerate(G)}
    rows, rhs = [], []
    for a in G:
        row = [0] * len(G)
        row[gi[a]] ^= 1
        row[gi[q - a]] ^= 1
        rows.append(row)
        rhs.append((g[a] + g[rinv * a % q]) % 2)
    for l in D.fac:
        seen = set()
        for a in G:
            if a not in seen:
                line = [b for b in G if all(b % m == a % m for m in D.fac if m != l)]
                seen |= set(line)
                row = [0] * len(G)
                for b in line:
                    row[gi[b]] ^= 1
                rows.append(row)
                rhs.append(0)
    n = len(G)
    R = flint.nmod_mat([r + [b] for r, b in zip(rows, rhs)], 2).rref()[0]
    sol = [0] * n
    for i in range(R.nrows()):
        piv = next((j for j in range(n + 1) if int(R[i, j])), None)
        if piv is None:
            continue
        if piv == n:
            return None
        sol[piv] = int(R[i, n])
    N, k = flint.nmod_mat(rows, 2).nullspace()
    return G, sol, [[int(N[i, j]) for i in range(n)] for j in range(k)]


def affine(sol, basis):
    for cs in itertools.product((0, 1), repeat=len(basis)):
        v = list(sol)
        for c, b in zip(cs, basis):
            if c:
                v = [(x + y) % 2 for x, y in zip(v, b)]
        yield v


def extend_prime_not_q(W, r, assign, q=420):
    """level r q, r a prime not dividing q: f(r y) = g(y) on the old points; for a coset witness, at the units a
    of Z/rq, f(a) = Psi(a mod q) if a = w0 mod r, Psi(-a mod q) if a = -w0 mod r, and 0 otherwise (assign: coset ->
    (w0, support of Psi)); the order counts are 0 at the new points."""
    d = r * q
    F = {}
    for key, g in W.items():
        f = [0] * d
        for a in range(r, d, r):
            f[a] = g[a // r]
        if key[0] == "coset":
            w0, S = assign[key]
            for a in range(1, d):
                if a % r and gcd(a, q) == 1:
                    if a % r == w0:
                        f[a] = 1 if a % q in S else 0
                    elif a % r == r - w0:
                        f[a] = 1 if (q - a % q) in S else 0
        F[key] = f
    return F


def check_family(F, d, lift, label):
    T = lifted_standard(d)
    wit = all(is_witness(f, d, T) for f in F.values())
    disj = all(sum(f[a] for f in F.values()) <= 1 for a in range(1, d))
    one = all(sum(f[lift * a % d] for a in FP.XI420) % 2 == 1 for f in F.values())
    good = wit and disj and one and len(F) == 11
    print("    (AT) level %d (%s): %d functions, witnesses on %d standard tuples %s, disjoint supports %s, 1 on "
          "the lift of XI420 %s: %s" % (d, label, len(F), len(T), wit, disj, one, good), flush=True)
    return {"level": d, "tuples": len(T), "witnesses": wit, "disjoint": disj, "one_on_lift": one}, good


def mod3_counts(q=420):
    """F_105 = chi_3 chi_5 (a') on the entries 4a' of order 105 and F_140 = chi_4 chi_5 (a') on the entries 3a' of
    order 140 (NOTE Theorem 55(c)); odd functions with values in {0, +-1}."""
    F = {}
    for d, (l1, l2) in ((105, (3, 5)), (140, (4, 5))):
        f = [0] * q
        for a in range(1, q):
            if q // gcd(a, q) == d:
                f[a] = FP.legendre(a // (q // d), l1) * FP.legendre(a // (q // d), l2)
        F[d] = f
    return F


def is_witness3(f, d, tuples):
    """f odd mod 3, 0 at d/2, and summing to 0 mod 3 over every lifted standard tuple."""
    if d % 2 == 0 and f[d // 2] % 3:
        return False
    if any((f[a] + f[d - a]) % 3 for a in range(1, d)):
        return False
    return all(sum(f[a] for a in s) % 3 == 0 for s in tuples)


def zero_extension(f, k):
    d = k * len(f)
    g = [0] * d
    for a in range(k, d, k):
        g[a] = f[a // k]
    return g


def check_AT():
    rows, ok = [], True
    q = 420
    D = FP.Degree(q)
    W = base_witnesses(D)
    F3 = mod3_counts(q)
    L = FP.lattice(q)
    key0 = FP.coset_key(D, 1)
    eps = L.f2([1 if FP.coset_key(D, u) == key0 else 0 for u in L.Kp])
    on_B = all(L.f2([g[u] for u in L.Kp]) == eps for g in W.values())
    T = lifted_standard(q)
    w3 = all(is_witness3(f, q, T) for f in F3.values())
    on_B3 = all(sum(f[u] * v[i] for i, u in enumerate(L.Kp)) % 3 == 0 for f in F3.values() for v in L.Lb)
    same = all([abs(x) for x in F3[d]] == W[("order", d)] for d in (105, 140))
    r0, g0 = check_family(W, q, 1, "the eleven witnesses")
    r0.update({"equal_to_eps_on_B": on_B, "mod3_witnesses": w3, "mod3_vanish_on_B": on_B3,
               "mod3_support_equals_order_count": same})
    print("    (AT) each of the eleven equals eps on B_420: %s; F_105, F_140 are mod-3 witnesses %s, vanish mod 3 "
          "on B_420 %s, are +-1 exactly on the supports of lambda_105, lambda_140 %s" % (on_B, w3, on_B3, same),
          flush=True)
    rows.append(r0)
    ok &= g0 and on_B and w3 and on_B3 and same
    bounds = {q: 15}
    expect = {840: 13, 1260: 13, 2100: 11, 2940: 15, 4620: 11, 5460: 11, 7140: 13, 7980: 13, 9660: 15}

    def mod3_part(d, ext):
        """which of F_105, F_140 extend (ext: the extension rule), and the resulting lower bound."""
        T = lifted_standard(d)
        good = {k: is_witness3(ext(f), d, T) for k, f in F3.items()}
        bound = 6 + 3 + sum(3 if v else 1 for v in good.values())
        print("    (AT) level %d: mod-3 counts that extend: %s; lower bound %d" % (d, good, bound), flush=True)
        return good, bound
    for l in (3, 5, 7):
        d = l * q
        r, g = check_family({k: extend_prime_of_q(f, l) for k, f in W.items()}, d, l, "Theorem 57(i)")
        r["mod3"], bounds[d] = mod3_part(d, lambda f, l=l: extend_prime_of_q(f, l))
        rows.append(r)
        ok &= g
    r, g = check_family({k: extend_840(k, f) for k, f in W.items()}, 2 * q, 2, "explicit")
    r["mod3"], bounds[2 * q] = mod3_part(2 * q, lambda f: zero_extension(f, 2))
    rows.append(r)
    ok &= g
    # Psi for every class of r mod 420 (only r^-1 mod 420 enters the equations)
    G = [a for a in range(1, q) if gcd(a, q) == 1]
    cos = sorted(k for k in W if k[0] == "coset")
    dims, exist = set(), True
    for k in cos:
        for ri in G:
            s = psi_solutions(D, W[k], ri)
            exist &= s is not None
            if s is not None:
                dims.add(len(s[2]))
    rows.append({"psi_exists_for_all_6_cosets_and_96_classes": exist, "homogeneous_dimensions": sorted(dims)})
    print("    (AT) Psi exists for each of the six coset witnesses and every class of r mod 420 (576 systems): %s; "
          "dimension of the homogeneous solutions %s" % (exist, sorted(dims)), flush=True)
    ok &= exist and dims == {8}
    for r in (11, 13, 17, 19, 23):
        rinv = pow(r, -1, q)
        sols = {}
        for k in cos:
            Gs, sol, basis = psi_solutions(D, W[k], rinv)
            sols[k] = [frozenset(Gs[i] for i in range(len(Gs)) if v[i]) for v in affine(sol, basis)]
        assign = {}
        if r == 11:
            # five classes +-w0 for six cosets: two cosets share w0 = 1 with disjoint supports of Psi
            pair = next((k1, k2, s1, s2) for k1, k2 in itertools.combinations(cos, 2) for s1 in sols[k1]
                        for s2 in sols[k2] if not s1 & s2)
            assign[pair[0]], assign[pair[1]] = (1, pair[2]), (1, pair[3])
        w = 1 + (r == 11)
        for k in cos:
            if k not in assign:
                assign[k] = (w, sols[k][0])
                w += 1
        r_, g = check_family(extend_prime_not_q(W, r, assign), r * q, r, "Psi")
        r_["mod3"], bounds[r * q] = mod3_part(r * q, lambda f, r=r: zero_extension(f, r))
        pred = {105: FP.legendre(r, 3) * FP.legendre(r, 5) == 1, 140: FP.legendre(r, 4) * FP.legendre(r, 5) == 1}
        g &= r_["mod3"] == pred
        rows.append(r_)
        ok &= g
    good = all(bounds[d] == b for d, b in expect.items())
    ok &= good
    rows.append({"lower_bounds": {str(d): b for d, b in sorted(bounds.items())}})
    print("    (AT) lower bounds for every element with eps = 1 (the lift of XI420 has 15 entries): %s: %s" % (
        dict(sorted(bounds.items())), good), flush=True)
    return rows, ok


# ---------------------------------------------------------------------------------------------------------------
# counts equal to eps_q

def check_AU():
    rows, ok = [], True
    expect = {420: ([15, 35, 60, 105, 140], [20, 84, 420]),
              1740: ([15, 87], [20, 60, 116, 145, 348, 435, 580, 1740]),
              2460: ([15, 205], [20, 60, 123, 164, 492, 615, 820, 2460]),
              2652: ([39, 221], [51, 52, 68, 156, 204, 663, 884, 2652])}
    # the degrees with four factors, N_{P(q)} even and every e_l [G_S : Gamma_l] even, up to 3315
    P_ = sorted([4] + [p for p in range(3, 3315 // 60 + 1) if isprime(p)])
    case = []
    for S in itertools.combinations(P_, 4):
        m = S[0] * S[1] * S[2] * S[3]
        if m <= 3315:
            lb, top, terms = FP.lower_bound_a(list(S))
            if top % 2 == 0 and all((e * i) % 2 == 0 for e, i in terms.values()):
                case.append(m)
    case.sort()
    ok &= case == [1740, 2460, 2652, 3045, 3315]
    rows.append({"degrees_without_a_count_up_to_3315": case})
    print("    (AU) degrees with four factors, N_{P(q)} even and every e_l [G_S : Gamma_l] even, up to 3315: %s"
          % case, flush=True)
    for q in (420, 660, 780, 924, 1155, 1740, 2460, 2652):
        t0 = time.time()
        D = FP.Degree(q)
        if q in FP._CACHE:
            Kp, V = FP._CACHE[q].Kp, FP._CACHE[q].Lb
        else:
            Kp, V = FP.hodge_basis(q, zbasis=False)
        key0 = FP.coset_key(D, 1)
        cl = [1 if FP.coset_key(D, u) == key0 else 0 for u in Kp]
        eps = tuple(sum(a * b for a, b in zip(cl, v)) % 2 for v in V)
        zero = tuple(0 for _ in eps)
        orders = sorted({q // gcd(u, q) for u in Kp})
        lam = {d: tuple(sum(v[i] for i, u in enumerate(Kp) if q // gcd(u, q) == d) % 2 for v in V) for d in orders}
        is_eps = [d for d in orders if lam[d] == eps]
        is_zero = [d for d in orders if lam[d] == zero]
        lb, top, terms = FP.lower_bound_a(D.fac)
        no_k = top % 2 == 0 and all((e * i) % 2 == 0 for e, i in terms.values())
        # Proposition 64: for E = {l, l'} the count at the order m_E is 0 or eps iff n_l n_l' is even
        two = {}
        for l1, l2 in itertools.combinations(D.fac, 2):
            mE = l1 * l2
            nn = FP.nn(l1) * FP.nn(l2)
            two[mE] = ("eps" if lam[mE] == eps else "0" if lam[mE] == zero else "class", nn % 2)
        g = all((v[0] == "class") == (v[1] == 1) for v in two.values())
        if q in expect:
            g &= (is_eps, is_zero) == (sorted(set(expect[q][0])), sorted(expect[q][1])) or (
                q == 420 and set(expect[q][0]) <= set(is_eps) and set(expect[q][1]) <= set(is_zero))
        g &= no_k == (q in (1740, 2460, 2652))
        ok &= g
        rows.append({"q": q, "no_count_of_Proposition_52": no_k, "orders_equal_eps": is_eps,
                     "orders_vanishing": is_zero, "two_factor_levels": {str(k): v for k, v in two.items()},
                     "basis": "Z-basis" if q in FP._CACHE else "2-saturated", "ok": g,
                     "sec": round(time.time() - t0, 1)})
        print("    (AU) q=%d (case of Corollary 58 without a count: %s): orders with lambda = eps %s, vanishing "
              "%s; two-factor levels (value, n n' mod 2) %s (%.0f s): %s" % (q, no_k, is_eps, is_zero, two,
                                                                             time.time() - t0, g), flush=True)
    return rows, ok


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "xi420.json"
    only = sys.argv[2:] if len(sys.argv) > 2 else None
    res, allok, secs = {}, True, {}
    for key, fn in (("AR", check_AR), ("AS", check_AS), ("AT", check_AT), ("AU", check_AU)):
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
