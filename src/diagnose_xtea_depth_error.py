"""
diagnose_xtea_depth_error.py
==============================
READ-ONLY diagnostic. Loads the ALREADY-TRAINED checkpoint from run
20260810_202939_train20_test7 and runs inference (groundtruth=False,
same as real evaluation) on xtea ONLY. No training, no weight updates,
no file writes except the printed table.

GOAL: test whether xtea's AT error concentrates at DEEP DAG levels
(supports the exposure-bias/compounding hypothesis from
diagnose_xtea_gap.py, which found xtea has the deepest DAG -- 97 levels
-- of all 7 test designs) or is spread evenly (would refute that
hypothesis and point elsewhere, e.g. gate-vocabulary gaps).

Must reproduce the EXACT same shared_gate_types vocabulary used during
training (same 20 train + 7 test designs, same sorted-union construction)
or load_state_dict will fail on a gate_onehot dimension mismatch -- this
is not optional, it's how the checkpoint's cell_lut module was sized.

USAGE:
    python diagnose_xtea_depth_error.py --data_dir C:\\Krish\\GCNN\\data \
        --checkpoint C:\\Krish\\GCNN\\src\\results\\20260810_202939_train20_test7\\model_checkpoint.pt
"""
import argparse
import os
from collections import deque, defaultdict

import torch
import numpy as np

from parse_real_design import build_graph
from interactive_train_real import real_graph_to_tensors, move_to_device, DEVICE
from gnn_model import ModelConfig, TimingGNN

PAPER_TEST_DESIGNS = {
    "jpeg_encoder", "usbf_device", "aes192", "xtea", "spm", "y_huff", "synth_ram"
}


def discover_designs(data_dir):
    found = []
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


def per_node_level(g):
    """Same Kahn's-algorithm level computation used inside
    real_graph_to_tensors -- recomputed here (read-only) because
    real_graph_to_tensors doesn't return the per-node level array,
    only per-level edge index lists."""
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
    while q:
        u = q.popleft()
        for v in succ[u]:
            level[v] = max(level[v], level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"C:\Krish\GCNN\data")
    ap.add_argument("--checkpoint",
                     default=r"C:\Krish\GCNN\src\results\20260810_202939_train20_test7\model_checkpoint.pt")
    ap.add_argument("--n_buckets", type=int, default=10,
                     help="Number of equal-width depth buckets to group endpoints into")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    # This checkpoint format is CONFIRMED (inspected directly): a dict with
    # keys state_dict, hidden_dim, net_layers, lut_k, cell_correction,
    # in_dim, edge_dim, n_gates, gate_types, n_params, best_epoch.
    # Using the STORED gate_types list directly -- this guarantees an exact
    # match to what the model was trained with, no risk of vocabulary
    # reconstruction drift (e.g. if a design file changed between the
    # training run and now, or discover_designs() ordering differs).
    state_dict = ckpt["state_dict"]
    shared_gate_types = ckpt["gate_types"]
    print(f"  shared_gate_types (from checkpoint): {len(shared_gate_types)} types")
    print(f"  hidden_dim={ckpt['hidden_dim']} net_layers={ckpt['net_layers']} "
          f"lut_k={ckpt['lut_k']} in_dim={ckpt['in_dim']} edge_dim={ckpt['edge_dim']} "
          f"n_gates={ckpt['n_gates']} best_epoch={ckpt['best_epoch']}")

    print("\nParsing xtea only (checkpoint already tells us the exact vocabulary "
          "and dims -- no need to reparse all 27 designs this time)...")
    xtea_raw = build_graph(os.path.join(args.data_dir, "xtea"), "xtea")
    xtea_g = move_to_device(real_graph_to_tensors(xtea_raw, shared_gate_types), DEVICE)
    level = per_node_level(xtea_raw)   # python list, index = pin_idx

    in_dim = xtea_g["x"].shape[1]
    edge_dim = xtea_g["edge_geo"].shape[1] if xtea_g["edge_geo"].numel() > 0 else 4
    assert in_dim == ckpt["in_dim"], (
        f"in_dim mismatch: xtea parses to {in_dim} features but checkpoint "
        f"expects {ckpt['in_dim']} -- something about parse_real_design.py "
        f"or interactive_train_real.py has changed since this checkpoint was "
        f"trained. Stop and report this, don't proceed."
    )
    assert edge_dim == ckpt["edge_dim"], (
        f"edge_dim mismatch: xtea parses to {edge_dim} but checkpoint expects "
        f"{ckpt['edge_dim']}. Stop and report this."
    )
    assert len(shared_gate_types) == ckpt["n_gates"], (
        f"n_gates mismatch: {len(shared_gate_types)} vs checkpoint's {ckpt['n_gates']}."
    )

    cfg = ModelConfig(hidden_dim=ckpt["hidden_dim"], net_layers=ckpt["net_layers"],
                       lut_k=ckpt["lut_k"], cell_correction=ckpt.get("cell_correction", True))
    model = TimingGNN(cfg, n_gates=ckpt["n_gates"], in_dim=ckpt["in_dim"],
                       edge_dim=ckpt["edge_dim"]).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print("Running inference on xtea (groundtruth=False, same as real eval)...")
    with torch.no_grad():
        out = model(xtea_g)
        ep = xtea_g["endpoints"].cpu().tolist()
        at_pred = out["AT"][xtea_g["endpoints"]].cpu().numpy()
        at_true = xtea_g["AT_true"][xtea_g["endpoints"]].cpu().numpy()

    abs_err = np.abs(at_pred - at_true)
    ep_levels = np.array([level[i] for i in ep])

    max_level = ep_levels.max()
    print(f"\nEndpoints: {len(ep)}  |  max DAG level among endpoints: {max_level}")

    # bucket endpoints by level into n_buckets equal-width bins
    edges = np.linspace(0, max_level + 1, args.n_buckets + 1)
    bucket_idx = np.digitize(ep_levels, edges) - 1
    bucket_idx = np.clip(bucket_idx, 0, args.n_buckets - 1)

    print(f"\n{'level range':18s} | {'n_endpoints':>11s} | {'mean|err|':>10s} | "
          f"{'median|err|':>12s} | {'max|err|':>9s} | {'mean AT_true':>13s}")
    print("-" * 85)
    for b in range(args.n_buckets):
        mask = bucket_idx == b
        n = mask.sum()
        if n == 0:
            continue
        lo, hi = edges[b], edges[b + 1]
        print(f"[{lo:6.1f}, {hi:6.1f})   | {n:11d} | {abs_err[mask].mean():10.4f} | "
              f"{np.median(abs_err[mask]):12.4f} | {abs_err[mask].max():9.4f} | "
              f"{at_true[mask].mean():13.4f}")

    # overall R2 as a sanity check against the reported 0.8883
    ss_res = ((at_pred - at_true) ** 2).sum()
    ss_tot = ((at_true - at_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    print(f"\nSanity check -- recomputed AT R2 for xtea: {r2:.4f} "
          f"(should match report.md's 0.8883; if it doesn't, something about "
          f"this checkpoint/vocab reload doesn't match the original run and "
          f"the bucket table above should not be trusted)")


if __name__ == "__main__":
    main()
