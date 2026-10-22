"""
cert_core.py -- exact certification of lifted lengths l_{2,d} (NOTE.txt Section 7; review of 2026-10-21).

Re-implemented here, independently of lifts.py: half, vec, the Hodge rows, Aoki's standard tuples ([MMRV] Def. 3.7,
same convention as ts-powers/code/v2_exotic.standard), the Hodge lattice L_d (Z-basis, verified saturated), the span
S_d of the standard vectors (all degrees e | d, lifted), and the GF(2) class functionals F.  Checked exactly:
L_d / S_d = (Z/2)^k (index [L:S] = 2^k and Smith normal form), k = rank of F on L_d; compared with
lifts.class_functionals.  Targets (xi_q lifted, plus extra classes) come from lifts.lifted.

Lower bound "no alpha in the class with sum |alpha_u| <= B": exhaustive meet-in-the-middle search.
  * Symmetry: the units t mod d act by signed permutations of the coordinates; t preserves the class C iff
    t L = L, t S = S and F(t target) = F(target); checked exactly on generators of (Z/d)^x (the conditions are
    multiplicative).  C = -C always.  Orbits O_1, O_2, ... (largest first); case j: supp(alpha) meets O_j but no
    earlier orbit; after a unit and a sign, alpha = e_{g_j} + w1 + w2 with w1, w2 supported on O_j u O_{j+1} u ...,
    |w1| <= ceil((B-1)/2), |w2| <= floor((B-1)/2) (w1, w2 = sub-multisets of the signed unit vectors of alpha).
  * Matching: h(v) = z . (R v) mod 2^64 (z random; h = 0 on every Hodge vector) and the F-values; keys
    h + mix(F) mod 2^64.  A Hodge alpha in C forces equal keys, so no key match = no such alpha (exact modular
    arithmetic).  Every key match is re-checked with exact integer arithmetic (Hodge rows, F).
No floating point is used anywhere.
"""
import os
import sys

sys.dont_write_bytecode = True
import random  # noqa: E402
import time  # noqa: E402
from math import comb, gcd  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")  # the top folder of the repository
CODE = os.path.join(SRC, "fermat-parity", "code")
for _sub in ("cyclic-covers", "fermat-frontier", "ts-powers"):
    sys.path.insert(0, os.path.join(SRC, _sub, "code"))
sys.path.insert(0, CODE)

import flint  # noqa: E402
import numpy as np  # noqa: E402

import lifts as LF  # noqa: E402

np.seterr(over="ignore")
MASK = (1 << 64) - 1
LIFTS_JSON = os.path.join(SRC, "fermat-parity", "outputs", "lifts.json")


# ------------------------------------------------------------------------------------------------ basic objects

def units(d):
    return [t for t in range(1, d) if gcd(t, d) == 1]


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


def wfun(k, d):
    k %= d
    return 0 if (k == 0 or 2 * k == d) else d - 2 * k


def hodge_rows(d, Kp):
    return [[wfun(t * u, d) for u in Kp] for t in units(d) if 2 * t < d]


def is_prime(n):
    return n > 1 and all(n % r for r in range(2, int(n ** 0.5) + 1))


def standard_tuples(e):
    """[MMRV] Def. 3.7: degree e = p k, p prime, k = e/p >= 3, p not dividing i; (i, k+i, -2i, k) for p = 2,
    (i, k+i, ..., (p-1)k+i, -p i) for p odd; tuples with an entry 0 are dropped."""
    out = []
    for p in range(2, e + 1):
        if e % p or not is_prime(p) or e // p < 3:
            continue
        k = e // p
        for i in range(1, e):
            if i % p == 0:
                continue
            if p == 2:
                s = (i, k + i, (-2 * i) % e, k)
            else:
                s = tuple((j * k + i) % e for j in range(p)) + ((-p * i) % e,)
            if all(x % e for x in s):
                out.append(s)
    return out


def S_vectors(d, idx):
    seen, out = set(), []
    for e in range(3, d + 1):
        if d % e:
            continue
        m = d // e
        for s in standard_tuples(e):
            v = vec(tuple(m * x for x in s), d, idx)
            if not any(v):
                continue
            key, nkey = tuple(v), tuple(-x for x in v)
            if key in seen or nkey in seen:
                continue
            seen.add(key)
            out.append(v)
    return out


def tuple_hodge(t, d):
    t = [a % d for a in t]
    if any(a == 0 for a in t):
        return False
    if sum(t) % d:
        if d % 2 or 2 * (sum(t) % d) != d:
            return False
        t = t + [d // 2]
    if len(t) % 2:
        return False
    return all(sum((s * a) % d for a in t) == (len(t) // 2) * d for s in units(d))


def lp_of(q):
    if q % 4 == 0:
        return 4, q // 4
    r = next(x for x in range(3, q + 1, 2) if q % x == 0)
    return r, q // r


# ------------------------------------------------------------------------------------------------ exact algebra

def bits_of(v):
    b = 0
    for i, x in enumerate(v):
        if x & 1:
            b |= 1 << i
    return b


def gf2_nullspace(rows, n):
    piv = {}
    for r in rows:
        while r:
            h = r.bit_length() - 1
            if h in piv:
                r ^= piv[h]
            else:
                piv[h] = r
                break
    keys = sorted(piv)
    for h in keys:
        rh = piv[h]
        for h2 in keys:
            if h2 != h and (piv[h2] >> h) & 1:
                piv[h2] ^= rh
    out = []
    for j in range(n):
        if j in piv:
            continue
        f = 1 << j
        for h, r in piv.items():
            if (r >> j) & 1:
                f |= 1 << h
        out.append(f)
    for f in out:                                   # exact check: orthogonal to every row
        assert all(((f & r).bit_count() & 1) == 0 for r in rows)
    return out


def select_independent(funcs, Lbits):
    """maximal subset of funcs (bitmasks) independent as functionals on the lattice with basis Lbits (mod 2)."""
    piv, keep = {}, []
    for f in funcs:
        x = 0
        for i, b in enumerate(Lbits):
            if (f & b).bit_count() & 1:
                x |= 1 << i
        while x:
            h = x.bit_length() - 1
            if h in piv:
                x ^= piv[h]
            else:
                piv[h] = x
                keep.append(f)
                break
    return keep


def to_int_rows(M):
    return [[int(x) for x in row] for row in M.tolist()]


def kernel_basis(R, n):
    m = len(R)
    A = flint.fmpz_mat([[R[i][j] for i in range(m)] + [1 if c == j else 0 for c in range(n)] for j in range(n)])
    H = to_int_rows(A.hnf())
    return [row[m:] for row in H if not any(row[:m]) and any(row[m:])]


def saturated(P, n):
    r = len(P)
    H = to_int_rows(flint.fmpz_mat(P).transpose().hnf())
    return all(H[i][j] == (1 if i == j else 0) for i in range(n) for j in range(r))


def hnf_rows_chunked(vecs, n, chunk=400):
    H = []
    for s in range(0, len(vecs), chunk):
        M = to_int_rows(flint.fmpz_mat(H + vecs[s:s + chunk]).hnf())
        H = [row for row in M if any(row)]
    return H


def index_snf(P, H, n):
    """[L : S] and the Smith form of S in a basis of L (H = HNF basis of S, P = basis of L)."""
    r = len(P)
    if len(H) != r:
        return None, None
    Ph = [row for row in to_int_rows(flint.fmpz_mat(P).hnf()) if any(row)]
    piv = [next(j for j in range(n) if Ph[i][j]) for i in range(r)]
    A = flint.fmpz_mat([[H[i][j] for j in piv] for i in range(r)])
    Bm = flint.fmpz_mat([[Ph[i][j] for j in piv] for i in range(r)])
    dA, dB = abs(int(A.det())), abs(int(Bm.det()))
    index = dA // dB if dB and dA % dB == 0 else None
    diag = None
    try:
        XT = flint.fmpq_mat(Bm.transpose()).solve(flint.fmpq_mat(A.transpose()))
        rows = XT.tolist()
        if all(x.q == 1 for row in rows for x in row):
            X = flint.fmpz_mat([[int(x.p) for x in row] for row in rows]).transpose()
            assert X * Bm == A
            Sm = X.snf()
            diag = [abs(int(Sm[i, i])) for i in range(r)]
    except Exception as exc:                        # pragma: no cover
        diag = ["error: %r" % (exc,)]
    return index, diag


def perm_sign(t, d, Kp, idx):
    perm = np.empty(len(Kp), np.int64)
    sign = np.empty(len(Kp), np.int64)
    for i, u in enumerate(Kp):
        a = t * u % d
        if 2 * a < d:
            perm[i], sign[i] = idx[a], 1
        else:
            perm[i], sign[i] = idx[d - a], -1
    return perm, sign


def act(M, perm, sign):
    out = np.zeros_like(M)
    out[..., perm] = M * sign
    return out


def unit_generators(d):
    sub, gens = {1}, []
    for t in units(d):
        if t in sub:
            continue
        gens.append(t)
        stack = list(sub)
        while stack:
            x = stack.pop()
            for g in gens:
                y = x * g % d
                if y not in sub:
                    sub.add(y)
                    stack.append(y)
    assert sub == set(units(d))
    return gens


def orbits_from(perms, n):
    seen, out = [False] * n, []
    for i0 in range(n):
        if seen[i0]:
            continue
        seen[i0] = True
        orb, stack = [i0], [i0]
        while stack:
            x = stack.pop()
            for pm in perms:
                y = int(pm[x])
                if not seen[y]:
                    seen[y] = True
                    orb.append(y)
                    stack.append(y)
        out.append(sorted(orb))
    out.sort(key=lambda O: (-len(O), O[0]))
    return out


def gcd_orbit_sizes(d):
    return sorted([sum(1 for u in half(d) if gcd(u, d) == g) for g in range(1, d) if d % g == 0 and 2 * g < d],
                  reverse=True)


# ------------------------------------------------------------------------------------------------ level setup

def setup_level(d, seed=20261021, compare_lifts=True):
    T0 = time.time()
    info = {"d": d}
    Kp = half(d)
    n = len(Kp)
    idx = {u: i for i, u in enumerate(Kp)}
    R = hodge_rows(d, Kp)
    m = len(R)
    R_np = np.array(R, dtype=np.int64)
    rankR = int(flint.fmpz_mat(R).rank())
    P = kernel_basis(R, n)
    maxP = max(abs(x) for row in P for x in row)
    if maxP > 2 ** 30:
        P = to_int_rows(flint.fmpz_mat(P).lll())
        maxP = max(abs(x) for row in P for x in row)
    r = len(P)
    P_np = np.array(P, dtype=np.int64)
    info.update(n=n, hodge_rows=m, rank_L=r, max_entry_L_basis=maxP)
    info["L_in_kernel"] = not (R_np @ P_np.T).any()
    info["L_rank_ok"] = (r == n - rankR)
    info["L_saturated"] = saturated(P, n)
    std_ok = True
    for e in range(3, d + 1):
        if d % e == 0:
            std_ok &= sorted(standard_tuples(e)) == sorted(tuple(s) for s in LF.standard(e))
    info["standard_tuples_equal_v2_exotic"] = std_ok
    Svecs = S_vectors(d, idx)
    S_np = np.array(Svecs, dtype=np.int64)
    info["S_generators"] = len(Svecs)
    info["S_hodge"] = not (S_np @ R_np.T).any()
    H = hnf_rows_chunked(Svecs, n)
    index, diag = index_snf(P, H, n)
    info["rank_S"] = len(H)
    info["index_L_S"] = index
    info["snf_nonunit"] = None if diag is None else [x for x in diag if x != 1]
    nulls = gf2_nullspace([bits_of(v) for v in Svecs], n)
    Lbits = [bits_of(b) for b in P]
    F = select_independent(nulls, Lbits)
    k = len(F)
    F_np = np.array([[(f >> i) & 1 for i in range(n)] for f in F], dtype=np.int64).reshape(k, n)
    info["k"] = k
    info["F_kills_S"] = not ((S_np @ F_np.T) % 2).any()
    snf_ok = diag is not None and all(isinstance(x, int) for x in diag) and all(x in (1, 2) for x in diag) \
        and diag.count(2) == k
    info["pins_class"] = bool(info["L_in_kernel"] and info["L_rank_ok"] and info["L_saturated"] and info["S_hodge"]
                              and len(H) == r and index == 2 ** k and snf_ok and info["F_kills_S"] and std_ok)
    if compare_lifts:
        Kp2, idx2, rows2, F2, Lb2 = LF.class_functionals(d)
        F2bits = [sum(1 << i for i, x in enumerate(f) if x % 2) for f in F2]
        F2_np = np.array(F2, dtype=np.int64).reshape(len(F2), n)
        info["lifts_rows_equal"] = bool(list(Kp2) == Kp and [list(x) for x in rows2] == R)
        info["lifts_k"] = len(F2)
        info["lifts_F_kill_S"] = not ((S_np @ F2_np.T) % 2).any()
        info["lifts_F_rank_on_L"] = len(select_independent(F2bits, Lbits))
        info["lifts_same_classes"] = bool(info["lifts_F_kill_S"] and info["lifts_F_rank_on_L"] == k == len(F2))
    gens = unit_generators(d)
    perms, sym_ok = {}, True
    for t in gens:
        perm, sign = perm_sign(t, d, Kp, idx)
        perms[t] = (perm, sign)
        sym_ok &= not (R_np @ act(P_np, perm, sign).T).any()
        sym_ok &= not ((act(S_np, perm, sign) @ F_np.T) % 2).any()
    info["unit_generators"] = gens
    info["sym_level_ok"] = bool(sym_ok)
    rng = random.Random(seed * 1000003 + d)
    z = [rng.getrandbits(64) for _ in range(m)]
    c = [sum(z[j] * R[j][i] for j in range(m)) & MASK for i in range(n)]
    if k <= 64:
        proj = None
        fbit = [sum(((F[j] >> i) & 1) << j for j in range(k)) for i in range(n)]
    else:
        proj = [rng.getrandbits(64) for _ in range(k)]
        fbit = []
        for i in range(n):
            x = 0
            for j in range(k):
                if (F[j] >> i) & 1:
                    x ^= proj[j]
            fbit.append(x)
    info["setup_sec"] = round(time.time() - T0, 1)
    return dict(d=d, Kp=Kp, idx=idx, n=n, R=R, R_np=R_np, P_np=P_np, S_np=S_np, F=F, F_np=F_np, k=k, c=c,
                fbit=fbit, proj=proj, gens=gens, perms=perms, info=info)


def setup_target(lev, target):
    d, idx, n = lev["d"], lev["idx"], lev["n"]
    tv = np.array(vec(target, d, idx), dtype=np.int64)
    info = {"target_entries": len(target)}
    info["target_hodge"] = not (lev["R_np"] @ tv).any()
    cvec = (lev["F_np"] @ tv) % 2
    info["class_bits"] = "".join(str(int(x)) for x in cvec)
    info["target_nontrivial"] = bool(cvec.any())
    if lev["proj"] is None:
        cbits = sum(int(b) << j for j, b in enumerate(cvec))
    else:
        cbits = 0
        for j, b in enumerate(cvec):
            if b:
                cbits ^= lev["proj"][j]
    ok = lev["info"]["sym_level_ok"]
    for t in lev["gens"]:
        perm, sign = lev["perms"][t]
        ok &= bool((((lev["F_np"] @ act(tv, perm, sign)) % 2) == cvec).all())
    if ok:
        orbits = orbits_from([lev["perms"][t][0] for t in lev["gens"]], n)
        info["symmetry"] = "full unit group, generators %s verified" % (lev["gens"],)
    else:
        Hs = []
        for t in units(d):
            perm, sign = perm_sign(t, d, lev["Kp"], idx)
            good = not (lev["R_np"] @ act(lev["P_np"], perm, sign).T).any()
            good &= not ((act(lev["S_np"], perm, sign) @ lev["F_np"].T) % 2).any()
            good &= bool((((lev["F_np"] @ act(tv, perm, sign)) % 2) == cvec).all())
            if good:
                Hs.append((t, perm))
        Hset = {t for t, _ in Hs}
        closed = all((a * b) % d in Hset for a in Hset for b in Hset)
        orbits = orbits_from([pm for _, pm in Hs], n) if closed else [[i] for i in range(n)]
        info["symmetry"] = "stabilizer of order %d verified elementwise (closed: %s)" % (len(Hset), closed)
    info["orbit_sizes"] = [len(O) for O in orbits]
    return dict(tv=tv, cvec=cvec, cbits=cbits, orbits=orbits, info=info)


def check_rep(lev, tg, rep):
    if not rep:
        return {"present": False, "ok": False}
    d, idx = lev["d"], lev["idx"]
    t = [int(a) for a in rep]
    v = np.array(vec(t, d, idx), dtype=np.int64)
    out = {"present": True, "length": len(t), "norm": int(np.abs(v).sum())}
    out["hodge_rows"] = not (lev["R_np"] @ v).any()
    out["hodge_tuple"] = tuple_hodge(t, d)
    out["in_class"] = bool(((((lev["F_np"] @ v) % 2) == tg["cvec"]).all()))
    out["ok"] = bool(out["hodge_rows"] and out["hodge_tuple"] and out["in_class"] and out["norm"] == len(t))
    out["vec"] = v
    return out


# ------------------------------------------------------------------------------------------------ the search

def splitmix(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def level0(N):
    return (np.array([N], np.int32), np.zeros(1, np.uint64), np.zeros(1, np.uint64), np.zeros(1, np.uint64))


def build_level(prev, N, ch, fb):
    """all nondecreasing item sequences one longer than those of prev (prev sorted by first item)."""
    pfirst, ph, pf, pp = prev
    starts = np.searchsorted(pfirst, np.arange(N, dtype=np.int32), side="left").astype(np.int64)
    sizes = len(pfirst) - starts
    total = int(sizes.sum())
    nf = np.repeat(np.arange(N, dtype=np.int32), sizes)
    offs = np.cumsum(sizes) - sizes
    ix = np.arange(total, dtype=np.int64)
    ix -= np.repeat(offs - starts, sizes)
    h = ph[ix]
    h += ch[nf]
    f = pf[ix]
    f ^= fb[nf]
    pk = pp[ix]
    pk *= np.uint64(1024)
    pk += nf.astype(np.uint64)
    del ix
    return (nf, h, f, pk)


def case_cost(N, B):
    Bp = B - 1
    b1, b2 = (Bp + 1) // 2, Bp // 2
    w1 = comb(N + b1 - 1, b1)
    pre = comb(N + b1 - 2, b1 - 1) if (b1 >= 1 and w1 > 3_000_000) else w1
    return w1, max(comb(N + b2 - 1, b2), pre)


def plan_cost(orbit_sizes, B):
    w1, w2 = 0, 0
    for j in range(len(orbit_sizes)):
        a, b = case_cost(2 * sum(orbit_sizes[j:]) + 1, B)
        w1 += a
        w2 = max(w2, b)
    return w1, w2


class Case:
    def __init__(self, lev, tg, U, base, B, coll, collect_cap):
        self.lev, self.n, self.base, self.B, self.coll, self.cap = lev, lev["n"], base, B, coll, collect_cap
        Bp = B - 1
        self.b1, self.b2 = (Bp + 1) // 2, Bp // 2
        N = 2 * len(U) + 1
        self.N = N
        cU = [lev["c"][i] for i in U]
        ch = np.zeros(N, np.uint64)
        fb = np.zeros(N, np.uint64)
        ch[1::2] = np.array(cU, dtype=np.uint64)
        ch[2::2] = np.array([(-x) & MASK for x in cU], dtype=np.uint64)
        fU = np.array([lev["fbit"][i] for i in U], dtype=np.uint64)
        fb[1::2] = fU
        fb[2::2] = fU
        self.ch, self.fb = ch, fb
        colmap = np.full(N, self.n, np.int64)
        colmap[1::2] = U
        colmap[2::2] = U
        sgn = np.zeros(N, np.int64)
        sgn[1::2] = 1
        sgn[2::2] = -1
        self.colmap, self.sgn = colmap, sgn
        self.Th = np.uint64((-lev["c"][base]) & MASK)
        self.Tf = np.uint64(tg["cbits"] ^ lev["fbit"][base])
        self.RT, self.FT, self.cvec = lev["R_np"].T.copy(), lev["F_np"].T.copy(), tg["cvec"]
        self.stats = {"N": N, "b1": self.b1, "b2": self.b2, "w1": 0, "w2": 0, "key_matches": 0, "genuine": 0,
                      "min_norm": None, "rep": None}

    def verify(self, pk1, pk2):
        n = self.n
        for s in range(0, len(pk1), 50000):
            a1, a2 = pk1[s:s + 50000], pk2[s:s + 50000]
            mm = len(a1)
            A = np.zeros((mm, n + 1), np.int64)
            rows = np.arange(mm)
            A[:, self.base] += 1
            for code, b in ((a1, self.b1), (a2, self.b2)):
                for jj in range(b):
                    it = ((code >> np.uint64(10 * jj)) & np.uint64(1023)).astype(np.int64)
                    np.add.at(A, (rows, self.colmap[it]), self.sgn[it])
            A = A[:, :n]
            good = ~((A @ self.RT).any(axis=1))
            good &= (((A @ self.FT) % 2) == self.cvec).all(axis=1)
            g = int(good.sum())
            if g:
                st = self.stats
                st["genuine"] += g
                G = A[good]
                norms = np.abs(G).sum(axis=1)
                j = int(np.argmin(norms))
                if st["min_norm"] is None or int(norms[j]) < st["min_norm"]:
                    st["min_norm"] = int(norms[j])
                    st["rep"] = {int(self.lev["Kp"][i]): int(G[j, i]) for i in np.nonzero(G[j])[0]}
                if self.coll is not None:
                    for row in G:
                        if len(self.coll) < self.cap:
                            self.coll.add(row.astype(np.int8).tobytes())

    def match(self, K2s, P2s, h1, f1, pack1):
        key1 = (self.Th - h1) + splitmix(f1 ^ self.Tf)
        self.stats["w1"] += len(key1)
        pos = np.searchsorted(K2s, key1, side="left")
        hit = K2s[np.minimum(pos, len(K2s) - 1)] == key1
        if not hit.any():
            return
        sel = np.nonzero(hit)[0]
        lo = pos[sel]
        hi = np.searchsorted(K2s, key1[sel], side="right")
        cnt = (hi - lo).astype(np.int64)
        tot = int(cnt.sum())
        self.stats["key_matches"] += tot
        rep1 = np.repeat(sel, cnt)
        j2 = np.repeat(lo, cnt) + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        self.verify(pack1(rep1), P2s[j2])

    def run(self):
        N, b1, b2 = self.N, self.b1, self.b2
        if self.B < 1:
            return self.stats
        w1full = comb(N + b1 - 1, b1)
        materialize = w1full <= 3_000_000
        top = max(b2, b1 if materialize else b1 - 1)
        levels = [level0(N)]
        for _ in range(top):
            levels.append(build_level(levels[-1], N, self.ch, self.fb))
        _, h2, f2, p2 = levels[b2]
        K2 = h2 + splitmix(f2)
        order = np.argsort(K2)
        K2s, P2s = K2[order], p2[order]
        del K2, order, h2, f2, p2
        if b2 > (b1 if materialize else b1 - 1):
            levels[b2] = None                       # not needed for W1: free memory
        self.stats["w2"] = len(K2s)
        if materialize:
            _, h1, f1, p1 = levels[b1]
            self.match(K2s, P2s, h1, f1, lambda sel: p1[sel])
        else:
            lf, lh, lff, lpk = levels[b1 - 1]
            starts = np.searchsorted(lf, np.arange(N, dtype=np.int32), side="left")
            for i in range(N):
                s = int(starts[i])
                if s >= len(lf):
                    continue
                h1 = lh[s:] + self.ch[i]
                f1 = lff[s:] ^ self.fb[i]
                self.match(K2s, P2s, h1, f1,
                           lambda sel, s=s, i=i: lpk[s + sel] * np.uint64(1024) + np.uint64(i))
        return self.stats


def search(lev, tg, B, collect=False, collect_cap=2_000_000):
    """exhaustive: is there alpha in the class (Hodge, F(alpha) = F(target)) with sum |alpha_u| <= B ?"""
    t0 = time.time()
    orbits = tg["orbits"]
    coll = set() if collect else None
    out = {"B": B, "cases": [], "w1": 0, "w2_max": 0, "key_matches": 0, "genuine": 0, "min_norm": None,
           "rep": None}
    for j in range(len(orbits)):
        U = sorted(x for O in orbits[j:] for x in O)
        st = Case(lev, tg, U, orbits[j][0], B, coll, collect_cap).run()
        out["cases"].append({"orbit_size": len(orbits[j]), "U": len(U), "N": st["N"], "b1": st["b1"],
                             "b2": st["b2"], "w1": st["w1"], "w2": st["w2"], "key_matches": st["key_matches"],
                             "genuine": st["genuine"]})
        out["w1"] += st["w1"]
        out["w2_max"] = max(out["w2_max"], st["w2"])
        out["key_matches"] += st["key_matches"]
        out["genuine"] += st["genuine"]
        if st["min_norm"] is not None and (out["min_norm"] is None or st["min_norm"] < out["min_norm"]):
            out["min_norm"], out["rep"] = st["min_norm"], st["rep"]
    out["sec"] = round(time.time() - t0, 1)
    return out, coll


def orbit_images(lev, v):
    """all +-t v, t a unit mod d (as int8 bytes)."""
    imgs = set()
    for t in units(lev["d"]):
        perm, sign = perm_sign(t, lev["d"], lev["Kp"], lev["idx"])
        w = act(v, perm, sign)
        imgs.add(w.astype(np.int8).tobytes())
        imgs.add((-w).astype(np.int8).tobytes())
    return imgs
