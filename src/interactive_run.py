"""
interactive_run.py
====================
RUN THIS FILE:  python interactive_run.py

An interactive menu that lets you CHOOSE and CHANGE the model's
architecture before training -- no code editing needed. After each run
it asks if you want to try a different configuration, and keeps a
running comparison table so you can see the effect of your choices
side by side.
"""
import torch
import numpy as np
from gen_circuits import gen_circuit
from model_configurable import ModelConfig, TimingGNN

# ----------------------------------------------------------------------
# DEVICE SETUP -- runs on GPU automatically if one is available,
# falls back to CPU otherwise. No manual switching needed.
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"GPU detected: {torch.cuda.get_device_name(0)} -- training will run on GPU.")
else:
    print("No GPU detected -- training will run on CPU (fine for small circuits).")


def move_graph_to_device(g, device):
    """Moves every tensor in a graph dict (including the per-level tensor
    dicts) onto the chosen device. Plain Python values (like the 'levels'
    list of ints) are left untouched."""
    out = {}
    for k, v in g.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {lvl: t.to(device) for lvl, t in v.items()}
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------
# Data prep (same as before -- builds tensors + topological levels)
# ----------------------------------------------------------------------
def to_tensors(c):
    pins = c["pins"]
    N = len(pins)
    x = torch.tensor([[p["x"] / 20.0, p["y"] / 20.0,
                       float(p["kind"] == "PI"),
                       float(p["kind"] == "IN"),
                       float(p["kind"] == "OUT")] for p in pins], dtype=torch.float)

    ns = torch.tensor([d for d, _ in c["net_edges"]], dtype=torch.long)
    nd = torch.tensor([s for _, s in c["net_edges"]], dtype=torch.long)
    cs = torch.tensor([i for i, _ in c["cell_edges"]], dtype=torch.long)
    cd = torch.tensor([o for _, o in c["cell_edges"]], dtype=torch.long)

    geo = []
    for d, s in c["net_edges"]:
        dx = (pins[s]["x"] - pins[d]["x"]) / 20.0
        dy = (pins[s]["y"] - pins[d]["y"]) / 20.0
        geo.append([dx, dy, abs(dx) + abs(dy)])
    geo = torch.tensor(geo, dtype=torch.float)

    from gen_circuits import GATE_TYPES, GATE_ID, CELL_LUT, SLEW_LUT
    oh, ld, ls = [], [], []
    for i, o in c["cell_edges"]:
        g = c["gate_of_out"][o]
        v = [0.0] * len(GATE_TYPES)
        v[GATE_ID[g]] = 1.0
        oh.append(v)
        ld.append(CELL_LUT[g])
        ls.append(SLEW_LUT[g])
    gate_onehot = torch.tensor(oh, dtype=torch.float)
    lut_delay = torch.tensor(np.array(ld), dtype=torch.float)
    lut_slew = torch.tensor(np.array(ls), dtype=torch.float)

    level = {p: 0 for p in range(N)}
    for p in c["order"]:
        for (d, s) in c["net_edges"]:
            if s == p:
                level[p] = max(level[p], level[d] + 1)
        for (i, o) in c["cell_edges"]:
            if o == p:
                level[p] = max(level[p], level[i] + 1)
    maxlvl = max(level.values())

    net_by_level, cell_by_level = {}, {}
    for L in range(maxlvl + 1):
        net_by_level[L] = torch.tensor(
            [e for e, (_, s) in enumerate(c["net_edges"]) if level[s] == L], dtype=torch.long)
        cell_by_level[L] = torch.tensor(
            [e for e, (_, o) in enumerate(c["cell_edges"]) if level[o] == L], dtype=torch.long)

    AT_true = torch.tensor([c["AT"][p] for p in range(N)], dtype=torch.float)
    SLEW_true = torch.tensor([c["SLEW"][p] for p in range(N)], dtype=torch.float)
    nd_true = torch.tensor([c["true_net_delay"][(d, s)] for d, s in c["net_edges"]], dtype=torch.float)
    cd_true = torch.tensor([c["true_cell_delay"].get((i, o), 0.0) for i, o in c["cell_edges"]], dtype=torch.float)

    return {"x": x, "net_src": ns, "net_dst": nd, "cell_src": cs, "cell_dst": cd,
            "edge_geo": geo, "gate_onehot": gate_onehot,
            "lut_delay": lut_delay, "lut_slew": lut_slew,
            "levels": list(range(maxlvl + 1)),
            "net_by_level": net_by_level, "cell_by_level": cell_by_level,
            "AT_true": AT_true, "SLEW_true": SLEW_true,
            "nd_true": nd_true, "cd_true": cd_true,
            "endpoints": torch.tensor(c["endpoints"], dtype=torch.long)}


def r2(pred, true):
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return 0.0
    return (1 - ss_res / ss_tot).item()


# ----------------------------------------------------------------------
# Interactive menu helpers
# ----------------------------------------------------------------------
def ask_int(prompt, default):
    raw = input(f"{prompt} [default {default}]: ").strip()
    return int(raw) if raw else default


def ask_choice(prompt, options, default):
    print(f"{prompt}")
    for i, opt in enumerate(options):
        marker = " (default)" if opt == default else ""
        print(f"    {i+1}. {opt}{marker}")
    raw = input(f"  choose 1-{len(options)} [default {default}]: ").strip()
    if not raw:
        return default
    return options[int(raw) - 1]


def ask_yesno(prompt, default=True):
    d = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{d}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def build_config_interactively():
    print("\n" + "=" * 60)
    print(" CONFIGURE YOUR MODEL")
    print("=" * 60)
    hidden_dim = ask_int("Hidden dimension (size of internal vectors)", 32)
    net_layers = ask_int("Number of Net Embedding layers (Stage 1 stacking)", 3)
    aggregation = ask_choice(
        "Aggregation method for Reduce step",
        ["sum_max", "mean", "sum", "max", "attention"], "sum_max")
    lut_k = ask_int("LUT grid resolution (k x k)", 3)
    cell_correction = ask_yesno("Use learned correction on top of LUT lookup?", True)

    cfg = ModelConfig(hidden_dim=hidden_dim, net_layers=net_layers,
                      aggregation=aggregation, lut_k=lut_k,
                      cell_correction=cell_correction)
    print(f"\nConfig set: {cfg.describe()}")
    return cfg


def build_training_config():
    print("\n" + "=" * 60)
    print(" TRAINING SETTINGS")
    print("=" * 60)
    n_train = ask_int("Number of training circuits", 14)
    n_test = ask_int("Number of test circuits (unseen)", 7)
    epochs = ask_int("Training epochs", 200)
    return n_train, n_test, epochs


# ----------------------------------------------------------------------
# Train + evaluate one configuration
# ----------------------------------------------------------------------
def run_one_config(cfg, n_train, n_test, epochs, from_gate_types):
    torch.manual_seed(0)
    train = [move_graph_to_device(to_tensors(gen_circuit(n_gates=40, n_inputs=6, seed=s)), DEVICE)
             for s in range(n_train)]
    test = [move_graph_to_device(to_tensors(gen_circuit(n_gates=40, n_inputs=6, seed=100 + s)), DEVICE)
            for s in range(n_test)]

    model = TimingGNN(cfg, n_gates=from_gate_types).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    mse = torch.nn.MSELoss()

    print(f"\nTraining with: {cfg.describe()}")
    print(f"({n_train} train circuits, {n_test} test circuits, {epochs} epochs)\n")

    for epoch in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for g in train:
            out = model(g)
            loss = (mse(out["AT"], g["AT_true"]) + mse(out["SLEW"], g["SLEW_true"])
                    + mse(out["net_delay"], g["nd_true"]) + mse(out["cell_delay"], g["cd_true"]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()

        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                tr = np.mean([r2(model(g)["AT"][g["endpoints"]], g["AT_true"][g["endpoints"]]) for g in train])
                te = np.mean([r2(model(g)["AT"][g["endpoints"]], g["AT_true"][g["endpoints"]]) for g in test])
            print(f"  epoch {epoch:4d} | loss {tot/n_train:8.4f} | "
                  f"R2 train {tr:6.3f} | R2 test(unseen) {te:6.3f}")

    model.eval()
    with torch.no_grad():
        final_r2 = np.mean([r2(model(g)["AT"][g["endpoints"]], g["AT_true"][g["endpoints"]]) for g in test])
        nd_r2 = np.mean([r2(model(g)["net_delay"], g["nd_true"]) for g in test])
    n_params = sum(p.numel() for p in model.parameters())
    return {"config": cfg.describe(), "test_r2_arrival": final_r2,
            "test_r2_netdelay": nd_r2, "n_params": n_params}


# ----------------------------------------------------------------------
# Main interactive loop
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from gen_circuits import GATE_TYPES

    print("=" * 60)
    print(" INTERACTIVE TIMING-GNN CONFIGURATOR")
    print(" (same architecture as the paper -- Net Embedding +")
    print("  Delay Propagation -- but every setting below is yours")
    print("  to change)")
    print("=" * 60)

    results_log = []

    while True:
        cfg = build_config_interactively()
        n_train, n_test, epochs = build_training_config()

        result = run_one_config(cfg, n_train, n_test, epochs, len(GATE_TYPES))
        results_log.append(result)

        print("\n" + "=" * 60)
        print(" RESULT")
        print("=" * 60)
        print(f"  Test R2 (arrival time) : {result['test_r2_arrival']:.4f}")
        print(f"  Test R2 (net delay)    : {result['test_r2_netdelay']:.4f}")
        print(f"  Trainable parameters   : {result['n_params']:,}")

        if len(results_log) > 1:
            print("\n" + "=" * 60)
            print(" COMPARISON ACROSS ALL RUNS THIS SESSION")
            print("=" * 60)
            print(f"{'#':>3} | {'config':<70} | {'R2 arrival':>10} | {'params':>8}")
            print("-" * 100)
            for i, r in enumerate(results_log):
                print(f"{i+1:>3} | {r['config']:<70} | {r['test_r2_arrival']:>10.4f} | {r['n_params']:>8,}")

        if not ask_yesno("\nTry another configuration?", default=True):
            break

    print("\nDone. Session summary saved above -- scroll up to compare all runs.")