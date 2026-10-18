"""
diagnose_cycles.py
===================
Read-only diagnostic: builds a design's raw graph (via parse_real_design.py's
build_graph -- no modification to that file) and finds the ACTUAL cycle(s)
in the net_edges + cell_edges graph, rather than just counting how many
pins fell into "level 0 because of a cycle" (which is all the training
pipeline currently reports).

WHY THIS MATTERS: gcd showed 759/1340 pins (57%) stuck in a cycle, with
AT R2 collapsing to 0.54 and slack R2 going NEGATIVE on that design.
Before assuming this is a graph-construction bug (something wrong in
parsing) vs. a genuine circuit property (e.g. GCD's iterative
subtract-compare structure creating a real combinational feedback loop
if a register boundary was misidentified during synthesis/parsing),
we need to actually SEE example cycle edges/pins, not just a count.

This script does NOT modify or fix anything -- it only reports, per
project rule (read real data before writing any fix code).

USAGE:
    python diagnose_cycles.py --design gcd --data_dir "C:\\Krish\\GCNN\\data"
"""
import argparse
import os
from collections import defaultdict

from parse_real_design import build_graph


def find_cycles(n_pins, net_edges, cell_edges, pin_names, max_cycles_to_report=3, max_cycle_len=20):
    """
    DFS-based cycle detection (standard white/gray/black coloring) over
    the combined net_edges + cell_edges directed graph. Returns up to
    max_cycles_to_report actual cycles as lists of pin indices, each
    truncated to max_cycle_len for readability if longer.
    """
    succ = defaultdict(list)
    edge_kind = {}  # (u,v) -> "net" or "cell", for reporting which edge type
    for d, s in net_edges:
        succ[d].append(s)
        edge_kind[(d, s)] = "net"
    for i, o in cell_edges:
        succ[i].append(o)
        edge_kind[(i, o)] = "cell"

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n_pins
    parent = [-1] * n_pins
    found_cycles = []

    def dfs(start):
        stack = [(start, iter(succ[start]))]
        color[start] = GRAY
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if color[nxt] == WHITE:
                    color[nxt] = GRAY
                    parent[nxt] = node
                    stack.append((nxt, iter(succ[nxt])))
                    advanced = True
                    break
                elif color[nxt] == GRAY:
                    cycle = [nxt]
                    cur = node
                    while cur != nxt and cur != -1:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(nxt)
                    cycle.reverse()
                    found_cycles.append(cycle[:max_cycle_len])
                    if len(found_cycles) >= max_cycles_to_report:
                        return
            if not advanced:
                color[node] = BLACK
                stack.pop()
            if len(found_cycles) >= max_cycles_to_report:
                return

    for start in range(n_pins):
        if color[start] == WHITE:
            dfs(start)
            if len(found_cycles) >= max_cycles_to_report:
                break

    return found_cycles, edge_kind


def run(design, data_dir):
    design_dir = os.path.join(data_dir, design)
    print(f"[diagnose_cycles] building graph for '{design}'...")
    g = build_graph(design_dir, design)

    n_pins = len(g["pins"]) if isinstance(g["pins"], list) else len(g.get("pin_names", []))
    pin_names = g.get("pin_names", [str(i) for i in range(n_pins)])
    net_edges = g["net_edges"]
    cell_edges = g["cell_edges"]

    print(f"\n[diagnose_cycles] {n_pins} pins, {len(net_edges)} net edges, "
          f"{len(cell_edges)} cell edges")

    cycles, edge_kind = find_cycles(n_pins, net_edges, cell_edges, pin_names)

    if not cycles:
        print("\n[diagnose_cycles] No cycles found by DFS. This contradicts the "
              "training log's cycle warning -- either the cycle only appears "
              "after some other transformation, or the warning's cause is "
              "something other than a true graph cycle. Investigate further "
              "before concluding either way.")
        return

    print(f"\n[diagnose_cycles] Found {len(cycles)} example cycle(s):\n")
    for i, cycle in enumerate(cycles):
        print(f"  --- Cycle {i+1} (length {len(cycle)}, showing up to 20 nodes) ---")
        for j in range(len(cycle) - 1):
            u, v = cycle[j], cycle[j + 1]
            kind = edge_kind.get((u, v), "?")
            uname = pin_names[u] if u < len(pin_names) else f"pin_{u}"
            vname = pin_names[v] if v < len(pin_names) else f"pin_{v}"
            print(f"    {uname}  --[{kind}]-->  {vname}")
        print()

    print("[diagnose_cycles] If these edges show a genuine feedback path through "
          "REGISTER/FLIP-FLOP boundaries (e.g. a D-to-Q or clock-related pin "
          "appearing as a plain combinational cell_edge), that's a graph-"
          "construction bug: clocked elements should break the cycle by NOT "
          "treating their Q output as combinationally dependent on D within "
          "the same cycle -- the sequential element should act as a level-0 "
          "source, not a mid-cycle passthrough. If the path is purely "
          "combinational gates with no register in the loop, that indicates "
          "either a genuine synthesis artifact (real combinational loop, "
          "which OpenROAD/STA would normally flag as an error) or a parsing "
          "bug that missed a register instance's pin entirely.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()
    run(args.design, args.data_dir)
