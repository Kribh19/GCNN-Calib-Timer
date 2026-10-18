"""
diagnose_xtea_gap.py
=====================
READ-ONLY diagnostic. Does not train, does not modify any file.

Checks two concrete, falsifiable hypotheses for why xtea's AT R2 (0.8883)
lags the other high-real-AT-coverage test designs (aes192 0.9714,
jpeg_encoder 0.9793, usbf_device 0.9447, y_huff 0.9691) despite xtea
having MORE real-AT label coverage (23.75%) than aes192 (13.3%) or
jpeg_encoder (14.6%) -- so it is not a coverage-ceiling problem like
spm/synth_ram.

HYPOTHESIS A -- PI input-delay out-of-distribution:
  at_init_mlp (gnn_model.py) is a small learned head trained only on
  the 20 TRAIN designs' set_input_delay values. If xtea's real SDC
  input delays fall outside the range seen in training, the model is
  extrapolating for every PI pin in xtea, and since AT propagates via
  amax from PIs forward, this corrupts every downstream endpoint.

HYPOTHESIS B -- DAG depth / critical path length:
  Training uses groundtruth=True (teacher forcing); testing uses
  groundtruth=False (model's own predicted delays). Errors compound
  across DAG levels ONLY at test time. If xtea's critical path is
  deeper than other similarly-scored test designs, this would hit it
  disproportionately.

USAGE:
    python diagnose_xtea_gap.py --data_dir C:\\Krish\\GCNN\\data
"""
import argparse
import os
from collections import deque

from parse_real_design import build_graph

PAPER_TEST_DESIGNS = {
    "jpeg_encoder", "usbf_device", "aes192", "xtea", "spm", "y_huff", "synth_ram"
}


def discover_designs(data_dir):
    found = []
    if not os.path.isdir(data_dir):
        return found
    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        v_ok = os.path.exists(os.path.join(folder, f"{name}.synthesis_preroute.v"))
        sdf_ok = os.path.exists(os.path.join(folder, f"{name}.sdf"))
        def_ok = any(f.endswith(".def") for f in os.listdir(folder))
        if v_ok and def_ok and sdf_ok:
            found.append(name)
    return found


def dag_depth(g):
    """Max topological level (Kahn's algorithm), same method as
    real_graph_to_tensors -- but we only need the max, not per-level
    edge lists, so this is a lightweight read-only recompute."""
    N = len(g["pins"])
    indeg = [0] * N
    succ = [[] for _ in range(N)]
    for a, b in g["net_edges"]:
        succ[a].append(b)
        indeg[b] += 1
    for a, b in g["cell_edges"]:
        succ[a].append(b)
        indeg[b] += 1
    level = [0] * N
    q = deque(i for i in range(N) if indeg[i] == 0)
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in succ[u]:
            level[v] = max(level[v], level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    maxlvl = max(level) if level else 0
    n_pis = sum(1 for p in g["pins"] if p["kind"] == "PI")
    return maxlvl, N, seen, n_pis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"C:\Krish\GCNN\data")
    args = ap.parse_args()

    all_designs = discover_designs(args.data_dir)
    train_designs = [d for d in all_designs if d not in PAPER_TEST_DESIGNS]
    test_designs = [d for d in all_designs if d in PAPER_TEST_DESIGNS]

    print("=" * 78)
    print(" HYPOTHESIS A -- input_delay / clock_period distribution")
    print(" (checking whether xtea's real SDC values fall outside the")
    print("  range seen across the 20 TRAIN designs)")
    print("=" * 78)

    train_input_delays = []
    train_clock_periods = []
    per_design_summary = []

    for d in train_designs:
        try:
            g = build_graph(os.path.join(args.data_dir, d), d)
        except Exception as e:
            print(f"  [SKIP] {d}: failed to parse ({e})")
            continue
        vals = list(g.get("true_input_delay", {}).values())
        cp = g.get("clock_period", None)
        if vals:
            train_input_delays.extend(vals)
        if cp is not None:
            train_clock_periods.append(cp)
        per_design_summary.append((d, "TRAIN", vals, cp))

    print(f"\nAcross {len(train_designs)} TRAIN designs:")
    if train_input_delays:
        print(f"  set_input_delay range: [{min(train_input_delays):.4f}, "
              f"{max(train_input_delays):.4f}] ns  (n={len(train_input_delays)} values)")
    else:
        print("  set_input_delay: NO VALUES FOUND across any train design -- "
              "this itself would be a serious finding, not expected to be empty.")
    if train_clock_periods:
        print(f"  clock_period range: [{min(train_clock_periods):.4f}, "
              f"{max(train_clock_periods):.4f}] ns  (n={len(train_clock_periods)} designs)")

    print(f"\nPer-design breakdown (TRAIN):")
    for d, split, vals, cp in per_design_summary:
        rng = f"[{min(vals):.3f}, {max(vals):.3f}]" if vals else "NONE"
        print(f"  {d:20s} input_delay={rng:20s} clock_period={cp}")

    print(f"\n{'-'*78}")
    print(" TEST designs (for direct comparison against the TRAIN range above)")
    print(f"{'-'*78}")

    test_summary = []
    for d in test_designs:
        try:
            g = build_graph(os.path.join(args.data_dir, d), d)
        except Exception as e:
            print(f"  [SKIP] {d}: failed to parse ({e})")
            continue
        vals = list(g.get("true_input_delay", {}).values())
        cp = g.get("clock_period", None)
        depth, n_pins, seen, n_pis = dag_depth(g)
        test_summary.append((d, vals, cp, depth, n_pins, seen, n_pis))

    print(f"\n{'design':15s} | {'input_delay range (ns)':26s} | {'clk_period':>10s} | "
          f"{'DAG depth':>9s} | {'n_pins':>8s} | {'n_PI':>6s} | {'unreachable(cycle)':>18s}")
    print("-" * 105)
    for d, vals, cp, depth, n_pins, seen, n_pis in test_summary:
        rng = f"[{min(vals):.3f}, {max(vals):.3f}]" if vals else "NONE"
        cyc = n_pins - seen
        marker = " <-- xtea" if d == "xtea" else ""
        print(f"{d:15s} | {rng:26s} | {str(cp):>10s} | {depth:9d} | {n_pins:8d} | "
              f"{n_pis:6d} | {cyc:18d}{marker}")

    print(f"\n{'='*78}")
    print(" INTERPRETATION GUIDE (not a conclusion -- read the numbers above)")
    print(f"{'='*78}")
    print(" HYPOTHESIS A supported if: xtea's input_delay range and/or clock_period")
    print("   fall clearly outside the min/max range seen across the 20 TRAIN")
    print("   designs above. If xtea's values are well within the train range,")
    print("   hypothesis A is REFUTED and we move to hypothesis B / gate vocab.")
    print()
    print(" HYPOTHESIS B supported if: xtea's DAG depth is notably higher than")
    print("   aes192/jpeg_encoder/usbf_device/y_huff's depths (paste this whole")
    print("   table -- I don't have those four designs' depths yet either).")
    print()
    print(" Also check the 'unreachable(cycle)' column -- nonzero means some pins")
    print(" never got a topological level (cycle), which forces them to level 0")
    print(" and could distort AT for exactly the pins caught in it. If xtea shows")
    print(" a nonzero count here and the others don't, that's a third concrete")
    print(" lead, independent of A and B.")


if __name__ == "__main__":
    main()
