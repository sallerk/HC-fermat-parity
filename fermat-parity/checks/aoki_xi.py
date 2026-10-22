"""Aoki's exceptional element xi_q, implemented literally from [Aok83] (Math. Ann. 266, 1983):
  p. 26: R_m, products (a)(b) = (ab) or 0, D_m = <(a) + (-a)> (+ (m/2) for m even).
  p. 28: tau_d(a) = phi(m)/phi(m') * prod_{p | d/(m,a), p not | m/d} (1, -p^-1) * (a')  if (m,a) | d, else 0,
         m' = m/(m,a), a' = a/(m,a), p^-1 taken mod m/d, (1, -p^-1) = (1) + (-p^-1).
  p. 37: xi_{q,1} = prod_{p | q} (1, g_p, ..., g_p^(a_p - 1)), a_p = phi(p)/2 (a_3 = a_4 = 1), g_p a primitive root;
         phi(d) xi_{q,d} + sum_{d' | d, d' != d} tau_d((d') xi_{q,d'}) = 0 (mod D_{q/d}).
  sign = -1 is Aoki's equation (phi(d) xi_{q,d} = -sum ...); sign = +1 is [MMRV] Def. 4.8 as printed.
Independent of the fermat-parity code.  Usage: python -I aoki_xi.py
"""
import sys
from math import gcd
from fractions import Fraction
from itertools import product as iproduct

sys.stdout.reconfigure(encoding="utf-8")


def factor(n):
    f, p = {}, 2
    while p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def phi(n):
    r = n
    for p in factor(n):
        r = r // p * (p - 1)
    return r


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def prim_roots(p):
    """Primitive roots mod p (p an odd prime or 4)."""
    if p == 4:
        return [3]
    return [g for g in range(2, p) if all(pow(g, (p - 1) // r, p) != 1 for r in factor(p - 1))]


def P_of(q):
    """Aoki's P(q) minus {2}: the elements of {4} + {odd primes} dividing q (q squarefree away from 4)."""
    f = factor(q)
    out = [p for p in f if p != 2]
    if f.get(2, 0) == 2:
        out.append(4)
    elif f.get(2, 0) not in (0,):
        raise ValueError("q must be odd or exactly divisible by 4")
    return sorted(out)


def crt(residues):
    """residues: list of (modulus, residue) with pairwise coprime moduli -> residue mod product."""
    x, m = 0, 1
    for mi, ri in residues:
        # solve x' = x mod m, x' = ri mod mi
        t = ((ri - x) * pow(m, -1, mi)) % mi
        x, m = x + m * t, m * mi
    return x % m


def primitive_part(q, roots):
    """xi_{q,1} = prod_{p | q} (1, g_p, ..., g_p^(a_p - 1)), as a dict on Z/q."""
    Ps = P_of(q)
    halves = []
    for p in Ps:
        a_p = phi(p) // 2
        g = roots[p]
        halves.append([(p, pow(g, i, p)) for i in range(a_p)])
    out = {}
    for combo in iproduct(*halves):
        c = crt(list(combo))
        out[c] = out.get(c, 0) + 1
    return out


def lift(e, x, q):
    """(e) x: R_{q/e} -> R_q, (b) -> (e b)."""
    out = {}
    for b, c in x.items():
        a = (e * b) % q
        assert a != 0
        out[a] = out.get(a, 0) + c
    return out


def tau(x, q, d):
    """Aoki's tau_d : R_q -> R_{q/d} (p. 28)."""
    n = q // d
    out = {}
    if n == 1:
        return out
    for a, c in x.items():
        g = gcd(q, a)
        if d % g:
            continue
        mp, ap = q // g, a // g
        assert phi(q) % phi(mp) == 0
        f = phi(q) // phi(mp)
        terms = {ap % n: 1}
        for p in factor(d // g):
            if n % p == 0:
                continue
            pinv = pow(p, -1, n)
            new = {}
            for z, cz in terms.items():
                for y in (1, (-pinv) % n):
                    w = (z * y) % n
                    assert w != 0
                    new[w] = new.get(w, 0) + cz
            terms = new
        for z, cz in terms.items():
            out[z] = out.get(z, 0) + f * c * cz
    return out


def mod_D(x, n):
    """Coordinates of x in R_n / D_n, free on (r), 0 < r < n/2."""
    out = {}
    for a, c in x.items():
        a %= n
        if 2 * a == n:
            continue
        if a < n - a:
            out[a] = out.get(a, 0) + c
        else:
            out[n - a] = out.get(n - a, 0) - c
    return {k: v for k, v in out.items() if v != 0}


def xi(q, roots, sign=-1, stop_at_first=False):
    """Run the recursion; return (parts, failures).  Divisions that are not exact are continued over Q."""
    parts = {1: primitive_part(q, roots)}
    failures = []
    for d in divisors(q)[1:-1]:
        n = q // d
        rhs = {}
        for e in divisors(d)[:-1]:
            for z, cz in tau(lift(e, parts[e], q), q, d).items():
                rhs[z] = rhs.get(z, 0) + sign * cz
        red = mod_D(rhs, n)
        ph = phi(d)
        bad = {r: v for r, v in red.items() if Fraction(v) % ph != 0}
        if bad:
            failures.append((d, ph, len(bad), len(red), sorted(bad.items())[:3]))
            if stop_at_first:
                return parts, failures
        parts[d] = {r: Fraction(v, ph) for r, v in red.items()}
    return parts, failures


def assemble(q, parts):
    out = {}
    for d, x in parts.items():
        for a, c in lift(d, x, q).items():
            out[a] = out.get(a, 0) + c
    return {a: c for a, c in out.items() if c != 0}


def is_hodge(q, x):
    """theta(x) = 0: sum_a c_a (<t a / q> - 1/2) = 0 for all t in (Z/q)^x  (Aoki p. 26)."""
    for t in range(1, q):
        if gcd(t, q) != 1:
            continue
        s = sum(c * (2 * ((t * a) % q) - q) for a, c in x.items())
        if s != 0:
            return False
    return True


def least_roots(q):
    return {p: prim_roots(p)[0] for p in P_of(q)}


def all_roots(q):
    Ps = P_of(q)
    for combo in iproduct(*[prim_roots(p) for p in Ps]):
        yield dict(zip(Ps, combo))


def in_Q(q):
    try:
        Ps = P_of(q)
    except ValueError:
        return False
    f = factor(q)
    if any(e > 1 for p, e in f.items() if p != 2):
        return False
    return len(Ps) % 2 == 0 and len(Ps) > 0


if __name__ == "__main__":
    # 1. q = 12 worked by hand from p. 37 and Lemma 5.3
    parts, fails = xi(12, least_roots(12), -1)
    x12 = assemble(12, parts)
    print("q=12 parts:", {d: dict(v) for d, v in parts.items()}, "failures:", fails)
    print("q=12 xi (mod D coords):", mod_D(x12, 12), "Hodge:", is_hodge(12, x12))
    print("(1,8,9) Hodge:", is_hodge(12, {1: 1, 8: 1, 9: 1}),
          " (1,8,9) - xi_12 mod D:", mod_D({a: {1: 1, 8: 1, 9: 1}.get(a, 0) - x12.get(a, 0) for a in range(1, 12)}, 12))

    # 2. two factors: every q in Q with two factors up to 600, all primitive roots, both signs
    two = [q for q in range(2, 601) if in_Q(q) and len(P_of(q)) == 2]
    stats = {-1: [0, 0, 0], 1: [0, 0, 0]}  # runs, integral, hodge
    for q in two:
        for roots in all_roots(q):
            for sign in (-1, 1):
                parts, fails = xi(q, roots, sign)
                stats[sign][0] += 1
                if not fails:
                    stats[sign][1] += 1
                    if is_hodge(q, assemble(q, parts)):
                        stats[sign][2] += 1
    print("two factors: %d degrees q <= 600; [runs, integral, Hodge] by sign:" % len(two), stats)

    # 3. four factors: first failure and all failing levels, every choice of primitive roots, both signs
    four = [q for q in range(2, 3100) if in_Q(q) and len(P_of(q)) == 4]
    print("four-factor q <= 3099:", four)
    for q in four:
        res = []
        for roots in all_roots(q):
            for sign in (-1, 1):
                parts, fails = xi(q, roots, sign)
                res.append((sign, tuple(sorted(roots.items())), [f[0] for f in fails]))
        firsts = sorted(set(r[2][0] if r[2] else None for r in res), key=lambda v: (v is None, v))
        print("q=%d P=%s choices=%d first failing levels=%s; all runs fail: %s"
              % (q, P_of(q), len(res), firsts, all(r[2] for r in res)))
        if q in (420, 1155):
            for r in res:
                print("   sign=%+d roots=%s failing d=%s" % r)

    # 4. q = 420, Aoki's sign, least roots: the first failures in detail; the Q-solution is Hodge?
    parts, fails = xi(420, least_roots(420), -1)
    for f in fails[:2]:
        print("420 failure at d=%d: phi(d)=%d, %d of %d coordinates not divisible; e.g. %s" % f)
    print("420: rational solution Hodge:", is_hodge(420, assemble(420, parts)))
