"""
diagnose_reconvergence.py
===========================
READ-ONLY diagnostic. No training, no writes except printed output.

Tests the reconvergent-path hypothesis (from Bairamkulov & Friedman,
"Graphs in VLSI", Ch 4.1): AT error may concentrate at topological levels
with high RECONVERGENCE DENSITY (many paths merging into few nodes),
because our amax-only propagation discards all but the single maximum
incoming path at every such point -- discarding information the STA
theory says should be intersected across all parallel paths, not just
maxed.

Cross-references this against the earlier depth-bucket error finding on
xtea (pre-endpoint-fix data showed ~60% of residual mass concentrated in
levels 9.8-19.6, from only 3.6% of endpoints). If that band also shows
unusually high reconvergence (in-degree), it's a concrete link between
the STA theory and xtea's measured behavior -- not just a plausible
story.

USAGE:
    python diagnose_reconvergence.py --data_dir C:\\Krish\\GCNN\\data
"""
import argparse
import os
from collections import deque

from parse_real_design import build_graph


def per_node_level_and_indegree(g):
    """
    Kahn's algorithm for topological level (same method used throughout
    this project), PLUS per-node in-degree counted across BOTH net edges
    and cell edges -- in-degree > 1 at a node means multiple paths
    reconverge there.
    """
    N = len(g["pins"])
    indeg = [0] * N
    succ = [[] for _ in range(N)]
    for a, b in g["net_edges"]:
        succ[a].append(b)
        indeg[b] += 1
    for a, b in g["cell_edges"]:
        succ[a].append(b)
        indeg[b] += 1
    in_degree = list(indeg)  # copy before Kahn's algorithm consumes it

    level = [0] * N
    q = deque(i for i in range(N) if indeg[i] == 0)
    while q:
        u = q.popleft()
        for v in succ[u]:
            level[v] = max(level[v], level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return level, in_degree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"C:\Krish\GCNN\data")
    ap.add_argument("--n_buckets", type=int, default=10)
    args = ap.parse_args()

    designs_to_check = ["xtea", "aes192", "jpeg_encoder", "usbf_device", "y_huff"]

    print("=" * 90)
    print(" Reconvergence density by topological-level band, per test design")
    print(" (in-degree > 1 at a node = paths merging there = amax discards info)")
    print("=" * 90)

    for d in designs_to_check:
        print(f"\n--- {d} ---")
        try:
            g = build_graph(os.path.join(args.data_dir, d), d)
        except Exception as e:
            print(f"  [SKIP] {d}: failed to parse ({e})")
            continue

        level, in_degree = per_node_level_and_indegree(g)
        N = len(level)
        max_level = max(level) if level else 0

        edges = [max_level * i / args.n_buckets for i in range(args.n_buckets + 1)]

        print(f"  N={N} pins, max_level={max_level}")
        print(f"  {'level range':18s} | {'n_pins':>7s} | {'mean_indeg':>10s} | "
              f"{'max_indeg':>9s} | {'n_reconverge(indeg>=2)':>22s} | {'%reconverge':>11s}")
        print("  " + "-" * 90)
        for b in range(args.n_buckets):
            lo, hi = edges[b], edges[b + 1]
            idxs = [i for i in range(N) if lo <= level[i] < hi or (b == args.n_buckets - 1 and level[i] == hi)]
            if not idxs:
                continue
            indegs = [in_degree[i] for i in idxs]
            n_reconverge = sum(1 for x in indegs if x >= 2)
            pct_reconverge = 100.0 * n_reconverge / len(idxs)
            mean_indeg = sum(indegs) / len(indegs)
            max_indeg = max(indegs)
            print(f"  [{lo:6.1f},{hi:6.1f})   | {len(idxs):7d} | {mean_indeg:10.3f} | "
                  f"{max_indeg:9d} | {n_reconverge:22d} | {pct_reconverge:10.1f}%")

    print(f"\n{'='*90}")
    print(" INTERPRETATION")
    print(f"{'='*90}")
    print(" For xtea: check whether the 9.8-19.6-level band (in a 0-97 scale, roughly")
    print(" bucket index 1 of 10) shows a spike in mean_indeg / %reconverge relative")
    print(" to neighboring bands, and relative to the same band in aes192/jpeg_encoder/")
    print(" usbf_device/y_huff. A spike there would directly support the reconvergent-")
    print(" path hypothesis as the cause of xtea's earlier-measured error concentration.")
    print(" If reconvergence is flat/unremarkable across levels for xtea, the hypothesis")
    print(" is NOT supported and the error concentration needs a different explanation.")


if __name__ == "__main__":
    main()
