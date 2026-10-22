"""Independent re-checks of the numbers quoted in Part II of the fermat-parity note (exact arithmetic).
Usage: python -I part2_checks.py"""
from fractions import Fraction
from math import gcd
from itertools import product


def primes_upto(n):
    s = [True] * (n + 1)
    s[0] = s[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = [False] * len(s[i * i::i])
    return [i for i in range(n + 1) if s[i]]


def phi(n):
    r, m, p = n, n, 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            r -= r // p
        p += 1
    if m > 1:
        r -= r // m
    return r


def is_hodge(q, coeffs):
    """coeffs: dict a -> c (a in Z/q, a != 0). Hodge iff sum c (2 (ta mod q) - q) = 0 over ta != 0, all units t."""
    for t in range(1, q):
        if gcd(t, q) != 1:
            continue
        s = 0
        for a, c in coeffs.items():
            b = (t * a) % q
            if b:
                s += c * (2 * b - q)
        if s:
            return False
    return True


def tup(entries):
    d = {}
    for a in entries:
        d[a] = d.get(a, 0) + 1
    return d


def order(q, a):
    return q // gcd(q, a)


def factors_P(q):
    """P(q): 4 (if 4 | q) and the odd primes dividing q."""
    out = []
    if q % 4 == 0:
        out.append(4)
    m = q
    while m % 2 == 0:
        m //= 2
    p = 3
    while p <= m:
        if m % p == 0:
            out.append(p)
            m //= p
        p += 2
    return out


def coset_key(q, u, P):
    """class of a unit u modulo K = {+-1}^r: tuple of (u mod l) up to sign."""
    return tuple(min(u % l, (-u) % l) for l in P)


def eps_cosets(q, coeffs):
    P = factors_P(q)
    cnt = {}
    for u in range(1, q):
        if gcd(u, q) == 1:
            cnt.setdefault(coset_key(q, u, P), 0)
    for a, c in coeffs.items():
        if gcd(a, q) == 1:
            k = coset_key(q, a, P)
            cnt[k] += c
    return {k: v % 2 for k, v in cnt.items()}


def mult_order(x, n):
    x %= n
    k, y = 1, x
    while y != 1:
        y = (y * x) % n
        k += 1
    return k


def e_and_index(q):
    """(e_l, [G_S : Gamma_l]) for l in P(q), S = P(q) - {l}, mu_l = l (odd) or 2 (l = 4)."""
    P = factors_P(q)
    res = {}
    for l in P:
        mu = 2 if l == 4 else l
        S = [x for x in P if x != l]
        # order of mu in G_S/K_S: least o with mu^o = +-1 mod every x in S
        o = 1
        while True:
            if all(pow(mu, o, x) in (1, x - 1) for x in S):
                break
            o += 1
        tau = 1
        for x in S:
            tau *= 1 if pow(mu, o, x) == 1 else -1
        N_S = 1
        for x in S:
            N_S *= (2 if x == 4 else x - 1) // 2
        res[l] = (1 if tau == -1 else 0, N_S // o)
    return res


def bound_a(q):
    N = phi(q) // 16
    return N + sum(e * i for e, i in e_and_index(q).values())


print("== Proposition 7.3 / Theorem 8.1(a) ==")
for q in (420, 660, 780, 924, 1155):
    print(q, e_and_index(q), "phi/16 =", phi(q) // 16, "bound (a) =", bound_a(q))

print("== XI420 ==")
XI420 = (1, 28, 33, 37, 57, 121, 213, 236, 276, 277, 337, 344, 361, 413, 416)
x = tup(XI420)
print("Hodge:", is_hodge(420, x), "len", len(XI420))
lv = {}
for a in XI420:
    lv[order(420, a)] = lv.get(order(420, a), 0) + 1
print("orders:", dict(sorted(lv.items())))
print("cosets odd:", set(eps_cosets(420, x).values()))
print("primitive part (mod 4,3,5,7):", sorted((a % 4, a % 3, a % 5, a % 7) for a in XI420 if gcd(a, 420) == 1))

print("== AOKI17 ==")
AOKI17 = (1, 8, 37, 121, 125, 157, 183, 212, 241, 249, 261, 287, 288, 304, 337, 364, 395)
y = tup(AOKI17)
print("Hodge:", is_hodge(420, y), "len", len(AOKI17))
lv = {}
for a in AOKI17:
    lv[order(420, a)] = lv.get(order(420, a), 0) + 1
print("orders:", dict(sorted(lv.items())))
prim = sorted((a % 4, a % 3, a % 5, a % 7) for a in AOKI17 if gcd(a, 420) == 1)
print("primitive part:", prim)
H5 = {pow(2, j, 5) for j in range(2)}
H7 = {pow(3, j, 7) for j in range(3)}
print("Aoki product g5=2,g7=3:", sorted((1, 1, b, c) for b in H5 for c in H7) == prim)

print("== 924 ==")
P924 = [u for u in range(1, 924) if gcd(u, 924) == 1 and u % 4 == 1 and u % 3 == 1
        and u % 7 in (1, 2, 4) and u % 11 in (1, 3, 4, 5, 9)]
print("primitive part size", len(P924))
for extra in (231, 693):
    z = tup(P924 + [extra])
    print("with", extra, "Hodge:", is_hodge(924, z), "cosets odd:", set(eps_cosets(924, z).values()))

print("== Proposition 9.1(c) ==")


def passes_c(P):
    for l in P:
        mu = 2 if l == 4 else l
        ok = False
        for s in (1, -1):
            good = True
            for lp in P:
                if lp == l:
                    continue
                v = (s * mu) % lp
                if lp in (3, 4):
                    if v != 1:
                        good = False
                        break
                elif mult_order(v, lp) % 2 == 0:
                    good = False
                    break
            if good:
                ok = True
                break
        if not ok:
            return False
    return True


elems = [4] + [p for p in primes_upto(20000) if p > 2]
cnt, sat = 0, []


def rec(start, chosen, prod_):
    global cnt
    if len(chosen) == 4:
        cnt += 1
        if passes_c(chosen):
            sat.append(prod_)
        return
    for i in range(start, len(elems)):
        np_ = prod_ * elems[i]
        if np_ > 20000:
            break
        rec(i + 1, chosen + [elems[i]], np_)


rec(0, [], 1)
print("four-factor q <= 20000 in Q:", cnt, "satisfying (c):", sat)
print("3170089 = 13*43*53*107:", 13 * 43 * 53 * 107, "passes (c):", passes_c([13, 43, 53, 107]))
print("420 per l:", {l: passes_c.__name__ for l in []})


def passes_one(P, l):
    mu = 2 if l == 4 else l
    for s in (1, -1):
        if all(((s * mu) % lp == 1) if lp in (3, 4) else (mult_order((s * mu) % lp, lp) % 2 == 1)
               for lp in P if lp != l):
            return True
    return False


print("420: some product passes (a) at l:", {l: passes_one([4, 3, 5, 7], l) for l in (4, 3, 5, 7)})

print("== Theorem 8.4: primes r with (r/3) = chi_4(r) = (r/5) ==")


def leg(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


rs = [r for r in primes_upto(120) if r > 7 and leg(r, 3) == (1 if r % 4 == 1 else -1) == leg(r, 5)]
print(rs)

print("== Aoki's recursion over Q at 420 (Aoki's sign, coefficient phi(d)) ==")


def tau_d(m, d, a):
    """Aoki's tau_d((a)) in R_{m/d}: returns dict b -> coefficient (b in Z/(m/d), nonzero entries only)."""
    g = gcd(m, a)
    if d % g:
        return {}
    mq = m // d
    mp = m // g
    ap = (a // g) % mq
    fac = phi(m) // phi(mp)
    res = {ap: Fraction(fac)}
    dd = d // g
    p = 2
    ps = []
    t = dd
    while p <= t:
        if t % p == 0:
            ps.append(p)
            while t % p == 0:
                t //= p
        p += 1
    for p in ps:
        if mq % p == 0:
            continue
        inv = pow(p, -1, mq) if mq > 1 else 0
        new = {}
        for b, c in res.items():
            for mult in (1, (-inv) % mq):
                bb = (b * mult) % mq
                new[bb] = new.get(bb, 0) + c
        res = new
    return {b: c for b, c in res.items() if b % mq != 0 and c != 0}


def modD(f, coeffs):
    """reduce an element of R_f (dict) modulo D_f: coordinates on 0 < u < f/2."""
    out = {}
    for b, c in coeffs.items():
        b %= f
        if b == 0 or 2 * b == f:
            continue
        if 2 * b < f:
            out[b] = out.get(b, 0) + c
        else:
            out[f - b] = out.get(f - b, 0) - c
    return {b: c for b, c in out.items() if c != 0}


q = 420
H = {4: [1], 3: [1], 5: [pow(2, j, 5) for j in range(2)], 7: [pow(3, j, 7) for j in range(3)]}
prim = []
for u in range(1, q):
    if gcd(u, q) == 1 and u % 4 in H[4] and u % 3 in H[3] and u % 5 in H[5] and u % 7 in H[7]:
        prim.append(u)
parts = {1: tup(prim)}
fails = []
divs = [d for d in range(2, q) if q % d == 0]
for d in divs:
    acc = {}
    for e in parts:
        if d % e:
            continue
        for b, c in parts[e].items():
            for bb, cc in tau_d(q, d, e * b).items():
                acc[bb] = acc.get(bb, 0) + c * cc
    rhs = modD(q // d, acc)
    xd = {b: -c / phi(d) for b, c in rhs.items()}
    if any(c.denominator != 1 for c in xd.values()):
        fails.append(d)
    parts[d] = xd
print("non-integral divisions at d =", fails)
xi = {}
for e, pe in parts.items():
    for b, c in pe.items():
        xi[(e * b) % q] = xi.get((e * b) % q, 0) + c
print("rational solution Hodge:", is_hodge(q, xi))
