#!/usr/bin/env python3
"""
Supplementary material for
"One Map, Four Collatz Worlds: The Dynamics of 3n + 1111".

Finite computation for T(n) = n/2 (n even), T(n) = 3n + 1111 (n odd).

What it does
  1. Reconstructs the eleven cycles from their halving vectors (Tables 2 and 3)
     and checks closure, distinct entries, length, (m, h), minimum, maximum,
     gcd layer, and the necessary condition (1111/d) | 2^h - 3^m.
  2. For every start 1 <= n <= LIMIT (default 10^8, endpoint included),
     iterates T in exact integer arithmetic until the orbit first reaches an
     element of one of the eleven certified cycles; records which cycle and the
     number of steps to that first entry.
     If an orbit has not reached a certified cycle after MAX_STEPS = 20,000
     steps, it is followed further looking for a repeated value (an unlisted
     cycle) for up to FALLBACK_STEPS more steps; either outcome is reported.
  3. Checks that every start lies in the same gcd(n, 1111) layer as its cycle,
     and that the class counts equal the exact inclusion-exclusion layer counts.

Usage
  python3 check_T1111.py                      # full run, n <= 10^8
  python3 check_T1111.py --limit 1000000      # quick test
  python3 check_T1111.py | tee check_T1111_output.txt
The run checkpoints after every block and resumes automatically if interrupted
(delete checkpoint.json or pass --fresh to start over).
Python integers are arbitrary precision, so overflow cannot occur.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from math import gcd
from multiprocessing import Pool, cpu_count

K = 1111
MAX_STEPS = 20_000
FALLBACK_STEPS = 1_000_000

# (o_1, halving vector (v_1,...,v_m)); order: Gamma first, then Table 3 order.
CERTS = [
    (43,   (3, 3, 1, 4, 1, 5, 2, 2, 1, 1, 1, 1, 1, 1, 4, 2, 2, 2, 6, 4,
            3, 1, 2, 1, 1, 1, 2, 3, 1, 8)),
    (77,   (1, 2, 1, 2, 1, 7)),
    (121,  (1, 1, 1, 2, 5, 4)),
    (209,  (1, 1, 5)),
    (253,  (1, 2, 4)),
    (319,  (2, 1, 4)),
    (341,  (1, 3, 3)),
    (407,  (2, 2, 3)),
    (101,  (1, 5)),
    (1313, (1, 1, 2, 2, 1, 1, 3, 3)),
    (1111, (2,)),
]
# (length, m, h, maximum) as stated in the paper
EXPECTED_SHAPE = [
    (100, 30, 70, 54640), (20, 6, 14, 9856), (20, 6, 14, 10252),
    (10, 3, 7, 6688), (10, 3, 7, 4048), (10, 3, 7, 5104), (10, 3, 7, 4312),
    (10, 3, 7, 3256), (8, 2, 6, 3232), (22, 8, 14, 25048), (3, 1, 2, 4444),
]
# basin sizes reported in the paper for LIMIT = 10^8 (same order as CERTS)
EXPECTED_BASINS_1E8 = [90009001, 640818, 3483125, 188007, 1249570, 934913,
                       88599, 2415868, 195688, 704402, 90009]


def build_cycle(o, vs):
    cyc, x = [], o
    for v in vs:
        assert x % 2 == 1
        y = 3 * x + K
        cyc.append(x)
        for _ in range(v):
            assert y % 2 == 0
            cyc.append(y)
            y //= 2
        x = y
    assert x == o, "cycle does not close"
    assert len(set(cyc)) == len(cyc), "repeated entry"
    return cyc


def canonical(cyc):
    i = cyc.index(min(cyc))
    return tuple(cyc[i:] + cyc[:i])


def _setup():
    cycles = [build_cycle(o, vs) for o, vs in CERTS]
    idx = {canonical(c): i for i, c in enumerate(cycles)}
    elem = {}
    for i, c in enumerate(cycles):
        for x in c:
            assert x not in elem, "cycles overlap"
            elem[x] = i
    cgcd = []
    for c in cycles:
        g = gcd(c[0], K)
        assert all(gcd(x, K) == g for x in c)
        cgcd.append(g)
    return cycles, idx, elem, cgcd


CYCLES, CYCIDX, ELEM, CYC_GCD = _setup()


def verify_certificates():
    print("Certified cycles (reconstructed from halving vectors)")
    print(f"{'#':>2} {'min':>5} {'len':>4} {'m':>3} {'h':>3} {'max':>6} {'gcd':>5}")
    for i, ((o, vs), cyc) in enumerate(zip(CERTS, CYCLES)):
        m, h = len(vs), sum(vs)
        assert len(cyc) == m + h
        assert min(cyc) == o
        shape = (len(cyc), m, h, max(cyc))
        assert shape == EXPECTED_SHAPE[i], f"cycle {o}: {shape} != {EXPECTED_SHAPE[i]}"
        g = CYC_GCD[i]
        assert (2 ** h - 3 ** m) % (K // g) == 0, "cycle-equation condition fails"
        print(f"{i:>2} {o:>5} {len(cyc):>4} {m:>3} {h:>3} {max(cyc):>6} {g:>5}")
    print("All eleven certificates verified.\n")


def steps_to_cycle(n):
    s = 0
    while n not in ELEM:
        n = n // 2 if n % 2 == 0 else 3 * n + K
        s += 1
    return s, n, ELEM[n]


def spot_checks():
    print("Spot checks quoted in the paper")
    print("  first terms of the orbit of 43:", CYCLES[0][:6])
    for n in (1, 2, 3):
        s, x, cid = steps_to_cycle(n)
        print(f"  n={n}: {s} steps to first element of cycle #{cid} (entry value {x})")
    print()


def fallback(start):
    """Follow an orbit that has not met a certified cycle; look for a repeat."""
    n, seen, path = start, {}, []
    for step in range(FALLBACK_STEPS):
        if n in ELEM:
            return ELEM[n], None
        if n in seen:
            return -2, canonical(path[seen[n]:])      # unlisted cycle
        seen[n] = step
        path.append(n)
        n = n // 2 if n % 2 == 0 else 3 * n + K
    return -1, None                                    # unresolved


def work(task):
    i, lo, hi = task
    elem = ELEM
    counts, layer = Counter(), Counter()
    unlisted, unresolved = [], []
    best_entry, peak = (-1, 0), (0, 0)
    for start in range(lo, hi):
        n, steps, top = start, 0, start
        while n not in elem:
            if steps >= MAX_STEPS:
                break
            n = 3 * n + K if n & 1 else n >> 1
            steps += 1
            if n > top:
                top = n
        if n in elem:
            cid = elem[n]
            if steps > best_entry[0]:
                best_entry = (steps, start)
        else:
            cid, cyc = fallback(start)
            if cid == -2:
                unlisted.append((start, list(cyc)))
            elif cid == -1:
                unresolved.append(start)
        counts[cid] += 1
        layer[(gcd(start, K), cid)] += 1
        if top > peak[0]:
            peak = (top, start)
    return {
        "i": i,
        "counts": {str(k): v for k, v in counts.items()},
        "layer": {f"{g},{c}": v for (g, c), v in layer.items()},
        "best_entry": list(best_entry),
        "peak": list(peak),
        "unlisted": unlisted[:20],
        "unresolved": unresolved[:20],
    }


def save(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def merge(state, r):
    for k, v in r["counts"].items():
        state["counts"][k] = state["counts"].get(k, 0) + v
    for k, v in r["layer"].items():
        state["layer"][k] = state["layer"].get(k, 0) + v
    if r["best_entry"][0] > state["best_entry"][0]:
        state["best_entry"] = r["best_entry"]
    if r["peak"][0] > state["peak"][0]:
        state["peak"] = r["peak"]
    state["unlisted"] += r["unlisted"]
    state["unresolved"] += r["unresolved"]
    state["done"].append(r["i"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10 ** 8)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--checkpoint", default="checkpoint.json")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    L = a.limit

    verify_certificates()
    spot_checks()

    tasks = [(i, lo, min(lo + a.chunk, L + 1))
             for i, lo in enumerate(range(1, L + 1, a.chunk))]
    state = None
    if os.path.exists(a.checkpoint) and not a.fresh:
        state = json.load(open(a.checkpoint))
        if state["limit"] != L or state["chunk"] != a.chunk:
            sys.exit("checkpoint was made with different --limit/--chunk; use --fresh")
        print(f"Resuming: {len(state['done'])}/{len(tasks)} blocks already done")
    if state is None:
        state = {"limit": L, "chunk": a.chunk, "done": [], "counts": {}, "layer": {},
                 "best_entry": [-1, 0], "peak": [0, 0], "unlisted": [], "unresolved": []}

    done = set(state["done"])
    pending = [t for t in tasks if t[0] not in done]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(work, pending):
            merge(state, r)
            save(a.checkpoint, state)
            print(f"  blocks done: {len(state['done'])}/{len(tasks)}  "
                  f"elapsed {time.time() - t0:,.0f}s", flush=True)

    # ------------------------------------------------------------ report
    counts = {int(k): v for k, v in state["counts"].items()}
    layer = {tuple(map(int, k.split(","))): v for k, v in state["layer"].items()}
    ok = True

    print(f"\nRESULTS for 1 <= n <= {L:,}")
    print(f"{'#':>2} {'min':>5} {'len':>4} {'m':>3} {'h':>3} {'max':>6} {'gcd':>5} {'starts':>12}")
    for i, (o, vs) in enumerate(CERTS):
        print(f"{i:>2} {o:>5} {len(CYCLES[i]):>4} {len(vs):>3} {sum(vs):>3} "
              f"{max(CYCLES[i]):>6} {CYC_GCD[i]:>5} {counts.get(i, 0):>12,}")
    print("basin counts in certificate order:",
          "; ".join(f"{counts.get(i, 0):,}" for i in range(len(CERTS))))

    def check(label, cond):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    print("\nCHECKS")
    check("every start assigned to a certified cycle (none unlisted/unresolved)",
          counts.get(-1, 0) == 0 and counts.get(-2, 0) == 0)
    check("class counts sum to LIMIT", sum(counts.values()) == L)
    mism = sum(v for (g, c), v in layer.items() if c >= 0 and g != CYC_GCD[c])
    check("every start lies in the same gcd layer as its cycle", mism == 0)
    exact = {1: L - L // 11 - L // 101 + L // 1111,
             11: L // 11 - L // 1111,
             101: L // 101 - L // 1111,
             1111: L // 1111}
    obs = {d: sum(counts.get(i, 0) for i in range(len(CERTS)) if CYC_GCD[i] == d)
           for d in exact}
    check(f"layer counts equal exact counts {exact}", obs == exact)
    check("every start coprime to 1111 reaches the 100-cycle",
          counts.get(0, 0) == exact[1] and layer.get((1, 0), 0) == exact[1])
    if L == 10 ** 8:
        check("basin counts equal those printed in the paper",
              [counts.get(i, 0) for i in range(len(CERTS))] == EXPECTED_BASINS_1E8)
    if state["unlisted"]:
        print("  unlisted cycles (first few):", state["unlisted"][:3])
    if state["unresolved"]:
        print("  unresolved starts (first few):", state["unresolved"][:10])

    print("\nSTATISTICS (output statistics, not bounds for untested starts)")
    print(f"  longest time to first entry into a certified cycle: "
          f"{state['best_entry'][0]} steps, at n = {state['best_entry'][1]:,}")
    print(f"  largest value visited before entering a certified cycle: "
          f"{state['peak'][0]:,}, from n = {state['peak'][1]:,}")

    summary = {"limit": L, "basin_counts": {str(CERTS[i][0]): counts.get(i, 0)
                                            for i in range(len(CERTS))},
               "layer_counts": obs, "best_entry": state["best_entry"],
               "peak": state["peak"], "all_checks_passed": ok}
    json.dump(summary, open("results_summary.json", "w"), indent=1)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
