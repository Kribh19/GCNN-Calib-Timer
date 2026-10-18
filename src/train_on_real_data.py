"""
train_on_real_data.py
=======================
Trains our configurable Timing GNN on REAL spm data, tests on REAL usb
data (genuinely unseen), using GPU automatically if available.

RUN THIS:  python train_on_real_data.py

Before running this, make sure:
  1. inspect_files.py has been run on your real spm/ and usb/ folders
     and the output looked like a normal DEF/Verilog/SDF file
  2. Your data folders are at DATA_DIR/spm/ and DATA_DIR/usb/ (edit
     DATA_DIR below to match where you copied the data in Windows)
"""
import sys
import os
import torch
import numpy as np

from parse_real_design import build_graph
from model_configurable import ModelConfig, TimingGNN

# ----------------------------------------------------------------------
# EDIT THIS to match where you copied the dataset on your Windows machine
# ----------------------------------------------------------------------
DATA_DIR = r"C:\Krish\GCNN\data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


def real_graph_to_tensors(g, gate_types, canvas_scale=1000.0):
    """Converts the parser's output (from parse_real_design.py) into the
    exact tensor dict shape TimingGNN expects -- same shape as our
    synthetic to_tensors(), just fed by real data instead.

    gate_types: a SHARED list of gate type names, built once across every
    design you plan to use together (train + test). This must be the same
    list for every graph, or the model's LUT module dimensions won't match
    between designs -- this is exactly the bug caught during testing."""
    pins = g["pins"]
    N = len(pins)

    x = torch.tensor([[p["x"] / canvas_scale, p["y"] / canvas_scale,
                       float(p["kind"] == "PI"),
                       float(p["kind"] == "IN"),
                       float(p["kind"] == "OUT")] for p in pins], dtype=torch.float)

    ns = torch.tensor([d for d, _ in g["net_edges"]], dtype=torch.long)
    nd = torch.tensor([s for _, s in g["net_edges"]], dtype=torch.long)
    cs = torch.tensor([i for i, _ in g["cell_edges"]], dtype=torch.long)
    cd = torch.tensor([o for _, o in g["cell_edges"]], dtype=torch.long)

    geo = []
    for d, s in g["net_edges"]:
        dx = (pins[s]["x"] - pins[d]["x"]) / canvas_scale
        dy = (pins[s]["y"] - pins[d]["y"]) / canvas_scale
        geo.append([dx, dy, abs(dx) + abs(dy)])
    geo = torch.tensor(geo, dtype=torch.float) if geo else torch.zeros(0, 3)

    gtid = {t: i for i, t in enumerate(gate_types)}
    oh = []
    for i, o in g["cell_edges"]:
        gt = g["gate_of_out"].get(o, "UNKNOWN")
        v = [0.0] * len(gate_types)
        v[gtid.get(gt, gtid.get("UNKNOWN", 0))] = 1.0
        oh.append(v)
    gate_onehot = torch.tensor(oh, dtype=torch.float) if oh else torch.zeros(0, len(gate_types))

    # placeholder LUT tensors (real per-cell LUTs need the .lib file --
    # not parsed here; the model still learns cell delay via net_delay/
    # cell_delay labels below, just without a real physical LUT prior)
    k = 3
    lut_delay = torch.zeros(len(g["cell_edges"]), k, k)
    lut_slew = torch.zeros(len(g["cell_edges"]), k, k)

    # topological levels via Kahn's algorithm
    from collections import deque
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
    if seen < N:
        print(f"  (note: {N - seen} pins in a cycle -- given level 0)")
    maxlvl = max(level) if level else 0

    net_by_level, cell_by_level = {}, {}
    for L in range(maxlvl + 1):
        net_by_level[L] = torch.tensor(
            [e for e, (_, s) in enumerate(g["net_edges"]) if level[s] == L], dtype=torch.long)
        cell_by_level[L] = torch.tensor(
            [e for e, (_, o) in enumerate(g["cell_edges"]) if level[o] == L], dtype=torch.long)

    # ground-truth labels: derive per-pin arrival time from summed true delays
    # along the (already-known-good) real delay values, so training has a
    # real numeric target even without re-running full STA ourselves
    AT = [0.0] * N
    for lvl in range(maxlvl + 1):
        for e in net_by_level[lvl].tolist():
            d, s = g["net_edges"][e]
            delay = g["true_net_delay"].get((d, s), 0.02)
            AT[s] = max(AT[s], AT[d] + delay)
        for e in cell_by_level[lvl].tolist():
            i, o = g["cell_edges"][e]
            delay = g["true_cell_delay"].get((i, o), 0.05)
            AT[o] = max(AT[o], AT[i] + delay)

    AT_true = torch.tensor(AT, dtype=torch.float)
    SLEW_true = torch.full((N,), 0.08)
    nd_true = torch.tensor(
        [g["true_net_delay"].get((d, s), 0.02) for d, s in g["net_edges"]], dtype=torch.float
    ) if g["net_edges"] else torch.zeros(0)
    cd_true = torch.tensor(
        [g["true_cell_delay"].get((i, o), 0.05) for i, o in g["cell_edges"]], dtype=torch.float
    ) if g["cell_edges"] else torch.zeros(0)

    endpoints = [p for p in range(N) if not any(a == p for a, _ in g["net_edges"])]
    endpoints = torch.tensor(endpoints, dtype=torch.long) if endpoints else torch.arange(N)

    return {"x": x, "net_src": ns, "net_dst": nd, "cell_src": cs, "cell_dst": cd,
            "edge_geo": geo, "gate_onehot": gate_onehot,
            "lut_delay": lut_delay, "lut_slew": lut_slew,
            "levels": list(range(maxlvl + 1)),
            "net_by_level": net_by_level, "cell_by_level": cell_by_level,
            "AT_true": AT_true, "SLEW_true": SLEW_true,
            "nd_true": nd_true, "cd_true": cd_true,
            "endpoints": endpoints}


def move_to_device(g, device):
    out = {}
    for k, v in g.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {lvl: t.to(device) for lvl, t in v.items()}
        else:
            out[k] = v
    return out


def r2(pred, true):
    if len(true) == 0:
        return float("nan")
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return 0.0
    return (1 - ss_res / ss_tot).item()


if __name__ == "__main__":
    print("=" * 60)
    print(" LOADING REAL DATA: spm (train) + usb (test)")
    print("=" * 60)

    spm_dir = os.path.join(DATA_DIR, "spm")
    usb_dir = os.path.join(DATA_DIR, "usb")

    print("\nParsing spm...")
    spm_graph = build_graph(spm_dir, "spm")

    print("\nParsing usb...")
    usb_graph = build_graph(usb_dir, "usb")

    # SHARED gate-type vocabulary across BOTH designs -- built once, used
    # for both, so the model's LUT module dimensions match on both graphs
    shared_gate_types = sorted(
        set(spm_graph["gate_of_out"].values()) | set(usb_graph["gate_of_out"].values())
    ) or ["UNKNOWN"]
    if "UNKNOWN" not in shared_gate_types:
        shared_gate_types.append("UNKNOWN")
    print(f"\nShared gate-type vocabulary ({len(shared_gate_types)} types): {shared_gate_types}")

    train_g = real_graph_to_tensors(spm_graph, shared_gate_types)
    test_g = real_graph_to_tensors(usb_graph, shared_gate_types)
    n_gate_types = len(shared_gate_types)

    train_g = move_to_device(train_g, DEVICE)
    test_g = move_to_device(test_g, DEVICE)

    print("\n" + "=" * 60)
    print(" TRAINING on real spm")
    print("=" * 60)
    cfg = ModelConfig(hidden_dim=32, net_layers=3, aggregation="sum_max", lut_k=3)
    print(f"Config: {cfg.describe()}")

    model = TimingGNN(cfg, n_gates=n_gate_types).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    mse = torch.nn.MSELoss()

    epochs = 300
    for epoch in range(1, epochs + 1):
        model.train()
        out = model(train_g)
        loss = (mse(out["AT"], train_g["AT_true"]) + mse(out["SLEW"], train_g["SLEW_true"])
                + mse(out["net_delay"], train_g["nd_true"]) + mse(out["cell_delay"], train_g["cd_true"]))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                tr = r2(model(train_g)["AT"][train_g["endpoints"]],
                        train_g["AT_true"][train_g["endpoints"]])
            print(f"  epoch {epoch:4d} | loss {loss.item():8.4f} | R2 on spm (train) {tr:6.3f}")

    print("\n" + "=" * 60)
    print(" TESTING on real usb (genuinely unseen)")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        out = model(test_g)
        test_r2 = r2(out["AT"][test_g["endpoints"]], test_g["AT_true"][test_g["endpoints"]])
    print(f"  R2 on usb (unseen) = {test_r2:.4f}")
    print("\nNOTE: labels here are derived by summing the SDF's real per-edge")
    print("delays along each path (not a full independent STA re-run), and")
    print("cell LUTs are placeholders (the .lib file isn't parsed here). This")
    print("is a genuine real-data experiment, but document these two")
    print("simplifications honestly in your report.")