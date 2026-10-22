"""Independent check of the witnesses in the proof of Theorem 8.4 (lifts of the class with eps_420 = 1).
For each level d = 420k, k prime, builds the extended maps as described in the note and checks:
even (resp. odd for F), 0 at d/2, vanishing on every standard element of R_d (mod 2, resp. mod 3),
value 1 (resp. 0 mod 3) on the lift of XI420, and disjoint supports. Also Peterson's w_1, w_2.
Usage: python -I check_lifts420.py"""
from math import gcd

Q = 420
XI420 = (1, 28, 33, 37, 57, 121, 213, 236, 276, 277, 337, 344, 361, 413, 416)


def primes_of(n):
    out, p = [], 2
    while p * p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0:
                n //= p
        p += 1
    if n > 1:
        out.append(n)
    return out


def std_elements(d):
    for p in primes_of(d):
        k = d // p
        for i in range(1, d):
            if (p * i) % d == 0 or (d % 2 == 0 and (p * i) % d == d // 2):
                continue
            if p == 2:
                yield [i, (i + k) % d, (-2 * i) % d, k]
            else:
                yield [(i + j * k) % d for j in range(p)] + [(-p * i) % d]


def leg(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def chi4(a):
    a %= 4
    return 0 if a % 2 == 0 else (1 if a == 1 else -1)


def order(m, a):
    return m // gcd(m, a)


# maps at level 420: dict name -> function on Z/420 (values in Z)
def coset_key(u):
    return (min(u % 5, (-u) % 5), min(u % 7, (-u) % 7))


COSETS = sorted({coset_key(u) for u in range(1, Q) if gcd(u, Q) == 1})


def g_coset(c):
    return lambda a: 1 if (a % Q and gcd(a, Q) == 1 and coset_key(a) == c) else 0


def g_lam(m):
    return lambda a: 1 if (a % Q and order(Q, a) == m) else 0


def F105(a):
    a %= Q
    if a == 0 or order(Q, a) != 105:
        return 0
    ap = (a // 4) % 105
    return leg(ap, 3) * leg(ap, 5)


def F140(a):
    a %= Q
    if a == 0 or order(Q, a) != 140:
        return 0
    ap = (a // 3) % 140
    return chi4(ap) * leg(ap, 5)


EVEN = {("coset", c): g_coset(c) for c in COSETS}
for m in (105, 140, 60, 35, 15):
    EVEN[("lam", m)] = g_lam(m)
ODD = {"F105": F105, "F140": F140}


STD = {}


def check_map(d, f, modulus, odd, name, lift):
    if d not in STD:
        STD[d] = list(std_elements(d))
    arr = [0] + [f(a) for a in range(1, d)]
    bad = 0
    for a in range(1, d):
        v, w = arr[a] % modulus, arr[(-a) % d] % modulus
        if (odd and (v + w) % modulus) or ((not odd) and (v - w) % modulus):
            bad += 1
    if d % 2 == 0 and arr[d // 2] % modulus:
        bad += 1
    nstd = 0
    for s in STD[d]:
        if sum(arr[e] for e in s) % modulus:
            nstd += 1
    val = sum(arr[e] for e in lift) % modulus
    return bad, nstd, val


def report(d, maps, lift):
    supports = {}
    ok = True
    for name, (f, modulus, odd) in maps.items():
        bad, nstd, val = check_map(d, f, modulus, odd, name, lift)
        want = 0 if modulus == 3 else 1
        good = bad == 0 and nstd == 0 and val == want
        ok &= good
        if not good:
            print("  FAIL", d, name, "parity/sign defects", bad, "standard failures", nstd, "value", val)
        if modulus == 2:
            supports[name] = {a for a in range(1, d) if f(a) % 2}
    names = list(supports)
    disjoint = all(not (supports[x] & supports[y]) for i, x in enumerate(names) for y in names[i + 1:])
    print("level", d, ": all maps pass:", ok, "; supports of the", len(names), "mod-2 maps disjoint:", disjoint)
    return ok and disjoint


# levels 420 l, l = 3, 5, 7
def ext_l(l, g):
    def f(a):
        if a % l == 0:
            return g((a // l) % Q)
        return g(a % Q)
    return f


for l in (3, 5, 7):
    d = 420 * l
    lift = [(l * a) % d for a in XI420]
    maps = {name: (ext_l(l, g), 2, False) for name, g in EVEN.items()}
    for name, g in ODD.items():
        maps[name] = (ext_l(l, g), 3, True)
    print("== d =", d)
    # report individually which F extends
    for name, g in ODD.items():
        bad, nstd, val = check_map(d, ext_l(l, g), 3, True, name, lift)
        print("  ", name, "extends (as f(ly)=F(y), f(a)=F(a mod 420)):", bad == 0 and nstd == 0)
    report(d, {k: v for k, v in maps.items() if v[1] == 2}, lift)

# level 840
d = 840
lift = [(2 * a) % d for a in XI420]


def ext840(g, bracket):
    def f(a):
        if a % 2 == 0:
            return g((a // 2) % Q)
        if not bracket:
            return 0
        return g(a % Q) if a % 8 in (3, 5) else 0
    return f


def ext840_140(a):
    if a % 2 == 0:
        return g_lam(140)((a // 2) % Q)
    if gcd(a, 840) == 3:
        y = (a // 3) % 280
        return ((1 if y % 8 in (1, 7) else 0) + (1 if y % 5 in (1, 4) else 0)) % 2
    return 0


maps = {}
for name, g in EVEN.items():
    if name[0] == "coset" or name == ("lam", 60):
        maps[name] = (ext840(g, True), 2, False)
    elif name == ("lam", 140):
        maps[name] = (ext840_140, 2, False)
    else:
        maps[name] = (ext840(g, False), 2, False)
print("== d = 840")
report(d, maps, lift)
bad, nstd, val = check_map(d, ext840(F105, False), 3, True, "F105", lift)
print("   F105 extends by 0 at the odd points:", bad == 0 and nstd == 0, "value on lift mod 3:", val)
bad, nstd, val = check_map(d, ext840(F140, False), 3, True, "F140", lift)
print("   (F140 by 0 at the odd points: passes =", bad == 0 and nstd == 0, ")")


# levels 420 r: identification a -> (r^-1 a mod 420, a mod r)
def solve_psi(r, c):
    """Psi on G_420 with Psi(u)+Psi(-u) = g(u)+g(ru) and even sums on all lines; returns (particular, kernel basis)."""
    units = [u for u in range(1, Q) if gcd(u, Q) == 1]
    idx = {u: i for i, u in enumerate(units)}
    g = g_coset(c)
    rows = []
    for u in units:
        v = (-u) % Q
        if idx[u] < idx[v]:
            rows.append(((1 << idx[u]) | (1 << idx[v]), (g(u) + g((r * u) % Q)) % 2))
    for l in (4, 3, 5, 7):
        fib = {}
        for u in units:
            key = tuple(u % x for x in (4, 3, 5, 7) if x != l)
            fib.setdefault(key, []).append(u)
        for us in fib.values():
            m = 0
            for u in us:
                m |= 1 << idx[u]
            rows.append((m, 0))
    n = len(units)
    piv = {}
    for m, b in rows:
        for col in range(n):
            if not (m >> col) & 1:
                continue
            if col in piv:
                pm, pb = piv[col]
                m ^= pm
                b ^= pb
            else:
                piv[col] = (m, b)
                break
        else:
            if b:
                return None, None, units
    # back substitution: particular solution with free variables 0
    sol = 0
    for col in sorted(piv, reverse=True):
        m, b = piv[col]
        rest = m & ~(1 << col)
        val = b ^ (bin(rest & sol).count("1") % 2)
        if val:
            sol |= 1 << col
    free = [c2 for c2 in range(n) if c2 not in piv]
    kern = []
    for fc in free:
        v = 1 << fc
        for col in sorted(piv, reverse=True):
            m, b = piv[col]
            rest = m & ~(1 << col)
            if bin(rest & v).count("1") % 2:
                v |= 1 << col
        kern.append(v)
    return sol, kern, units


def ext_r(r, g, psi_bits, units, w0, odd_F=None):
    d = Q * r
    rinv = pow(r, -1, Q)
    pset = {units[i] for i in range(len(units)) if (psi_bits >> i) & 1} if psi_bits is not None else set()

    def f(a):
        u, w = (rinv * a) % Q, a % r
        if w == 0:
            return g(u)
        if psi_bits is None:
            return 0
        if w == w0 % r:
            return 1 if u in pset else 0
        if w == (-w0) % r:
            return 1 if ((-u) % Q) in pset else 0
        return 0
    return f


for r in (11, 13, 17, 19, 23):
    d = Q * r
    rinv = pow(r, -1, Q)
    lift = [(r * a) % d for a in XI420]
    print("== d = 420 *", r)
    maps = {}
    sols = {}
    for k, c in enumerate(COSETS):
        sol, kern, units = solve_psi(r, c)
        if sol is None:
            print("  no Psi for coset", c)
            continue
        sols[c] = (sol, kern, units)
    w0s = {}
    if r >= 13:
        for k, c in enumerate(COSETS):
            w0s[c] = k + 1
    else:
        # r = 11: five pairs +-w0; look for two cosets with disjoint Psi (particular + kernel combinations)
        found = None
        for i, c1 in enumerate(COSETS):
            for c2 in COSETS[i + 1:]:
                s1, k1, _ = sols[c1]
                s2, k2, _ = sols[c2]
                span1 = [s1]
                for v in k1:
                    span1 += [x ^ v for x in span1]
                span2 = [s2]
                for v in k2:
                    span2 += [x ^ v for x in span2]
                for a1 in span1:
                    for a2 in span2:
                        if a1 & a2 == 0:
                            found = (c1, c2, a1, a2)
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        print("  r = 11: two cosets with disjoint Psi:", found is not None)
        c1, c2, a1, a2 = found
        sols[c1] = (a1, sols[c1][1], sols[c1][2])
        sols[c2] = (a2, sols[c2][1], sols[c2][2])
        others = [c for c in COSETS if c not in (c1, c2)]
        w0s[c1] = w0s[c2] = 1
        for k, c in enumerate(others):
            w0s[c] = k + 2
    for c in COSETS:
        sol, kern, units = sols[c]
        maps[("coset", c)] = (ext_r(r, g_coset(c), sol, units, w0s[c]), 2, False)
    for m in (105, 140, 60, 35, 15):
        maps[("lam", m)] = (ext_r(r, g_lam(m), None, None, 0), 2, False)
    report(d, maps, lift)
    for name, F, chi in (("F105", F105, leg(r, 3) * leg(r, 5)), ("F140", F140, chi4(r) * leg(r, 5))):
        bad, nstd, val = check_map(d, ext_r(r, F, None, None, 0), 3, True, name, lift)
        print("   ", name, "extends by 0:", bad == 0 and nstd == 0, "; predicted (chi(r) = 1):", chi == 1)

print("== Peterson's w_1, w_2")


def is_hodge(m, entries):
    for t in range(1, m):
        if gcd(t, m) != 1:
            continue
        s = 0
        for a in entries:
            b = (t * a) % m
            if b:
                s += 2 * b - m
        if s:
            return False
    return True


w1 = (56, 69, 176, 191, 193, 199, 204, 209, 210, 236, 243, 253, 257, 267, 296, 301)
w2 = (1, 17, 28, 132, 141, 168, 191, 199, 236, 253, 257, 267, 296, 371, 387, 416)
for name, w in (("w_1", w1), ("w_2", w2)):
    orders = {}
    for a in w:
        orders[order(Q, a)] = orders.get(order(Q, a), 0) + 1
    cos = {}
    for a in w:
        if gcd(a, Q) == 1:
            cos[coset_key(a)] = cos.get(coset_key(a), 0) + 1
    print(name, "Hodge:", is_hodge(Q, w), "orders:", dict(sorted(orders.items())),
          "coset counts odd:", all(cos.get(c, 0) % 2 for c in COSETS))

print("== Psi systems for all 96 classes of r mod 420")
units420 = [u for u in range(1, Q) if gcd(u, Q) == 1]
nsolv, dims = 0, set()
for r in units420:
    for c in COSETS:
        sol, kern, _ = solve_psi(r, c)
        if sol is not None:
            nsolv += 1
            dims.add(len(kern))
print("solvable systems:", nsolv, "of", len(units420) * len(COSETS), "; kernel dimensions:", dims)
