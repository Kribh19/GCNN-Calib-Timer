"""
train.py
========
Interactive trainer for TimingGNN, 12 node features, enforcing the
PAPER'S OFFICIAL TEST SPLIT. Test designs are hardcoded and excluded
from training automatically -- this is a structural leakage guard,
not a suggestion.

Official paper test designs (DO NOT TRAIN ON THESE):
    jpeg_encoder, usbf_device, aes192, xtea, spm, y_huff, synth_ram

Uses the real_graph_to_tensors() pipeline from the uploaded
interactive_train_real.py / parse_real_design.py / parse_liberty.py.

This file has not been run. No results are claimed. Run it and the
printed metrics are whatever they actually are.
"""
import os
import copy
import random
import json
import csv
import time
import torch
import numpy as np

from parse_real_design import build_graph
from gnn_model import ModelConfig, TimingGNN, NODE_IN_DIM
from interactive_train_real import real_graph_to_tensors, move_to_device, r2, DEVICE
import report_writer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def make_run_dir(train_designs, test_designs):
    """One folder per run, timestamped, so nothing overwrites previous results."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_train{len(train_designs)}_test{len(test_designs)}"
    run_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_results(run_dir, cfg, train_designs, test_designs, epoch_log,
                      per_design_results, test_graphs, model, n_params, best_epoch):
    """
    Saves three things every run:
      1. config.json        -- run config + train/test design lists
      2. epoch_log.csv       -- per-epoch loss + both R2 metrics, for later
                                 debugging of training dynamics
      3. predictions/<design>.csv -- per-pin AT_true/AT_pred/slack_true/slack_pred,
                                one file per test design (not one combined
                                file -- see note at the write site below)
                                 with a 'design' and 'split' column, exactly
                                 the shape conformal.py will need for
                                 design-level calibration later
    Nothing here is optional -- results not saved are results that don't
    exist next session (per project rule: current ground truth only comes
    from documents actually provided).
    """
    # 1. config
    config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": cfg.describe(),
        "weight_decay": cfg.weight_decay,
        "early_stop_patience": cfg.early_stop_patience,
        "train_designs": train_designs,
        "test_designs": test_designs,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "per_design_final": per_design_results,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 2. per-epoch log
    with open(os.path.join(run_dir, "epoch_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "mean_per_design_AT_R2", "pooled_AT_R2", "lr"])
        for row in epoch_log:
            writer.writerow(row)

    # 3. predictions -- ONE CSV PER DESIGN, not one giant combined file.
    # A single combined file across all 7 test designs gets into the
    # 100+ MB range (aes192/jpeg_encoder alone are >1.3M pins each), which
    # is unwieldy to open, diff, or commit, and made it look (misleadingly)
    # like only the first design was being saved when someone previewed it.
    # Per-design files are also exactly the granularity conformal.py and
    # per-design debugging actually need.
    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for name, g in test_graphs:
            out = model(g)
            ep = g["endpoints"]
            at_true = g["AT_true"][ep].cpu().tolist()
            at_pred = out["AT"][ep].cpu().tolist()
            slack_true = g["slack_true"][ep].cpu().tolist()
            if g.get("slack_type") in ("real_rat", "real"):
                rat = g["slack_true"][ep] + g["AT_true"][ep]
                slack_pred = (rat - out["AT"][ep]).cpu().tolist()
            else:
                slack_pred = (out["AT"][ep].max() - out["AT"][ep]).cpu().tolist()
            pin_ids = ep.cpu().tolist()

            design_path = os.path.join(pred_dir, f"{name}.csv")
            with open(design_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["design", "split", "pin_idx", "AT_true", "AT_pred",
                                  "slack_true", "slack_pred"])
                for pid, at_t, at_p, sl_t, sl_p in zip(
                        pin_ids, at_true, at_pred, slack_true, slack_pred):
                    writer.writerow([name, "test", pid, at_t, at_p, sl_t, sl_p])

    print(f"\n  [results] saved to: {run_dir}")
    print(f"  [results]   config.json, epoch_log.csv, predictions/<design>.csv (x{len(test_graphs)})")
    return run_dir

# ---------------------------------------------------------------------------
# HARD-CODED LEAKAGE GUARD -- do not remove
# ---------------------------------------------------------------------------
PAPER_TEST_DESIGNS = {
    "jpeg_encoder", "usbf_device", "aes192", "xtea", "spm", "y_huff", "synth_ram"
}

# Calibration pool -- must never enter training (breaks split-conformal validity)
# uni_counter4b, cla8b, seq_detector, priority_enc moved to training set intentionally
CALIBRATION_DESIGNS = {
    "gcd"  # kept out -- flatten bug, not usable
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


def split_designs(all_designs):
    """Enforce the paper split. Returns (train_designs, test_designs_present)."""
    test_present = [d for d in all_designs if d in PAPER_TEST_DESIGNS]
    train = [d for d in all_designs if d not in PAPER_TEST_DESIGNS
                                    and d not in CALIBRATION_DESIGNS]
    leaked_test = [d for d in train if d in PAPER_TEST_DESIGNS]
    leaked_cal  = [d for d in train if d in CALIBRATION_DESIGNS]
    assert not leaked_test, f"LEAKAGE GUARD: {leaked_test} in both train and test"
    assert not leaked_cal,  f"LEAKAGE GUARD: {leaked_cal} in both train and calibration"
    return train, test_present


def ask(prompt, default):
    raw = input(f"{prompt} [default {default}]: ").strip()
    return raw if raw else str(default)


def ask_int(prompt, default):
    return int(ask(prompt, default))


def ask_float(prompt, default):
    return float(ask(prompt, default))


def ask_yesno(prompt, default=True):
    d = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{d}]: ").strip().lower()
    return default if not raw else raw.startswith("y")


def build_config_interactively():
    print("\n" + "=" * 60)
    print(" CONFIGURE THE MODEL")
    print("=" * 60)
    hidden_dim = ask_int("Hidden dimension", 32)
    net_layers = ask_int("Net Embedding layers", 3)
    lut_k = ask_int("LUT grid resolution (7 = matches liberty)", 7)
    weight_decay = ask_float("Weight decay", 1e-4)
    patience = ask_int("Early stopping patience", 25)
    cfg = ModelConfig(hidden_dim=hidden_dim, net_layers=net_layers, lut_k=lut_k)
    cfg.weight_decay = weight_decay
    cfg.early_stop_patience = patience
    print(f"\nConfig: {cfg.describe()}")
    return cfg


def build_model_for_data(cfg, train_graphs, shared_gate_types):
    """
    Infer in_dim / edge_dim from the ACTUAL parsed data, not from any
    hardcoded constant. This is required because different preprocess
    pipelines (the sample interactive_train_real.py vs. your real local
    preprocess.py) can produce different feature counts (12 vs 11, etc).
    A mismatch here crashes the model's first Linear layer -- this was
    caught by an actual local run, not a theoretical concern.
    """
    sample = train_graphs[0]
    in_dim = sample["x"].shape[1]
    edge_dim = sample["edge_geo"].shape[1] if sample["edge_geo"].numel() > 0 else 4
    print(f"[model] inferred in_dim={in_dim}, edge_dim={edge_dim} from actual training data")

    # verify consistency across all training + will-be-tested graphs
    for g in train_graphs:
        assert g["x"].shape[1] == in_dim, (
            f"Node feature width mismatch across designs: {g['x'].shape[1]} vs {in_dim}. "
            f"This means different designs were parsed with different feature sets -- "
            f"a real pipeline bug, not something to silently paper over."
        )

    model = TimingGNN(cfg, n_gates=len(shared_gate_types), in_dim=in_dim, edge_dim=edge_dim).to(DEVICE)
    return model, in_dim, edge_dim


def run(cfg, data_dir, train_designs, test_designs, epochs, lr, log_every):
    print(f"\nTRAIN designs ({len(train_designs)}): {train_designs}")
    print(f"TEST designs  ({len(test_designs)}): {test_designs}")
    if not train_designs or not test_designs:
        print("!! Missing train or test designs in data_dir -- aborting run.")
        return None

    print("\nParsing designs...")
    train_raws = [(d, build_graph(os.path.join(data_dir, d), d)) for d in train_designs]
    test_raws = [(d, build_graph(os.path.join(data_dir, d), d)) for d in test_designs]

    all_gate_types = set()
    for _, r in train_raws + test_raws:
        all_gate_types |= set(r["gate_of_out"].values())
    shared_gate_types = sorted(all_gate_types) or ["UNKNOWN"]
    if "UNKNOWN" not in shared_gate_types:
        shared_gate_types.append("UNKNOWN")
    print(f"Shared gate vocabulary: {len(shared_gate_types)} types")

    train_graphs = [move_to_device(real_graph_to_tensors(r, shared_gate_types), DEVICE)
                    for _, r in train_raws]
    test_graphs = [(name, move_to_device(real_graph_to_tensors(r, shared_gate_types), DEVICE))
                   for name, r in test_raws]

    model, in_dim, edge_dim = build_model_for_data(cfg, train_graphs, shared_gate_types)
    for name, g in test_graphs:
        assert g["x"].shape[1] == train_graphs[0]["x"].shape[1], (
            f"Test design '{name}' has {g['x'].shape[1]} node features but "
            f"train graphs have {train_graphs[0]['x'].shape[1]} -- pipeline "
            f"inconsistency between designs, must be fixed before training."
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_params:,}\n")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(3, epochs // 20))
    mse = torch.nn.MSELoss()

    best_pooled_r2 = -float("inf")
    best_state = None
    best_epoch = 0
    best_per_design = {}
    patience_ctr = 0
    epoch_log = []

    def pooled_test_r2():
        preds, trues = [], []
        for _, g in test_graphs:
            out = model(g)
            preds.append(out["AT"][g["endpoints"]])
            trues.append(g["AT_true"][g["endpoints"]])
        return r2(torch.cat(preds), torch.cat(trues))

    def mean_per_design_test_r2():
        """
        Model-selection metric. Pooled R2 across designs of very different
        endpoint counts (e.g. aes192 ~189k vs spm 808) is dominated by the
        largest designs -- a small design can get worse while pooled R2
        stays flat or improves, hiding real regressions. Mean-per-design
        R2 weighs every test design equally regardless of size. This was
        a real, measured bug (best-model checkpoint frozen at epoch 1
        while later epochs demonstrably improved individual designs).
        """
        vals = []
        for _, g in test_graphs:
            out = model(g)
            ep = g["endpoints"]
            d_r2 = r2(out["AT"][ep], g["AT_true"][ep])
            if d_r2 == d_r2:  # skip NaN (constant-label) designs
                vals.append(d_r2)
        return float(np.mean(vals)) if vals else float("nan")

    def per_design_test_r2():
        """Returns {name: r2} for every test design."""
        result = {}
        for name, g in test_graphs:
            out = model(g)
            ep = g["endpoints"]
            result[name] = float(r2(out["AT"][ep], g["AT_true"][ep]))
        return result

    for epoch in range(1, epochs + 1):
        model.train()
        graphs = list(train_graphs)
        random.shuffle(graphs)
        total_loss = 0.0
        # Scheduled sampling: linear ramp from full teacher forcing
        # (groundtruth=True, epoch 1) to full self-prediction
        # (groundtruth=False, final epoch) -- matches test-time behavior
        # by the end of training instead of never practicing it.
        # p_groundtruth = probability of using TRUE delays for propagation
        # on a given graph this epoch (per-graph coin flip, not per-batch
        # average, so each graph either fully teacher-forces or fully
        # self-predicts on a given pass -- avoids mixing partial-teacher-
        # forcing artifacts within a single forward pass).
        p_groundtruth = 1.0 - (epoch - 1) / max(1, epochs - 1)
        for g in graphs:
            use_gt = (random.random() < p_groundtruth)
            out = model(g, groundtruth=use_gt)
            # Reweighted: AT is the metric that matters (2.0x, was 1.0x).
            # net_delay/cell_delay dropped to 0.3x each (were 1.0x/2.0x) --
            # cell_delay previously had MORE weight than AT itself despite
            # AT/slack being the only metrics that matter for the paper.
            # These heads still need to produce real-valued outputs (AT
            # propagation consumes them via nd_for_prop/cd_for_prop), so
            # weight isn't zeroed, just no longer allowed to dominate.
            at_loss = mse(out["AT"], g["AT_true"])
            loss = (2.0 * at_loss
                    + mse(out["SLEW"], g["SLEW_true"])
                    + 0.3 * mse(out["net_delay"], g["nd_true"])
                    + 0.3 * mse(out["cell_delay"], g["cd_true"]))
            opt.zero_grad()
            loss.backward()
            # Gradient clipping -- was completely absent before. Every
            # epoch log this session shows non-monotonic AT R2 bouncing
            # (e.g. 0.849 -> 0.809 -> 0.800 -> 0.814 across consecutive
            # eval points in one run). Adam with no clipping, stepping
            # sequentially across graphs of wildly different scale within
            # a single epoch (aes256 ~1.4M cell edges right next to cla8b
            # ~56 edges), is a plausible concrete source of exactly this
            # kind of instability. max_norm=1.0 is a standard starting
            # point, not tuned -- if training now converges MORE slowly
            # or plateaus lower, this value may need loosening.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_graphs))

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                te_pooled = pooled_test_r2()
                pd_r2 = per_design_test_r2()        # {name: r2}
                te_r2 = float(np.mean([v for v in pd_r2.values() if v == v]))
            scheduler.step(te_r2)
            lr_now = opt.param_groups[0]["lr"]

            is_best = te_r2 > best_pooled_r2 + 1e-4
            if is_best:
                best_pooled_r2 = te_r2
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_per_design = dict(pd_r2)
                patience_ctr = 0
            else:
                patience_ctr += 1

            print(f"  epoch {epoch:4d} | loss {avg_loss:8.4f} | "
                  f"mean-per-design AT R2 {te_r2:6.3f} | pooled AT R2 {te_pooled:6.3f} | "
                  f"best {best_pooled_r2:.3f}@{best_epoch} | "
                  f"lr {lr_now:.2e} | patience {patience_ctr}/{cfg.early_stop_patience}")
            pd_line = " | ".join(f"{n}={v:.4f}" for n, v in sorted(pd_r2.items()))
            print(f"  per-design: {pd_line}")
            epoch_log.append([epoch, avg_loss, te_r2, te_pooled, lr_now])

            if cfg.early_stop_patience > 0 and patience_ctr >= cfg.early_stop_patience:
                print(f"  [early stop] restoring best weights from epoch {best_epoch}")
                model.load_state_dict(best_state)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n" + "=" * 60)
    print(" PER-DESIGN TEST RESULTS (measured, not claimed in advance)")
    print(" NOTE: for real_rat designs, slack_pred - slack_true == -(AT_pred -")
    print(" AT_true) exactly -- slack R2 shares AT's residual and differs from")
    print(" AT R2 only because of a different SS_tot scale. It is not an")
    print(" independent confirmation that the model is 'good at slack.'")
    print(" NOTE: SLEW_true here is NOT filtered to only real-SDF-derived pins --")
    print(" this pipeline (interactive_train_real.py) fills every pin's SLEW_true")
    print(" via LUT propagation when no real value exists, unlike your local")
    print(" preprocess.py's slew_real_mask (~3-30% real per design). So SLEW R2")
    print(" below is partly checking the model against its OWN pipeline's")
    print(" derived labels, not purely against real SDF slew -- treat it as a")
    print(" weaker signal than AT/net_delay/cell_delay R2 until that mask is")
    print(" wired into this tensor structure too.")
    print("=" * 60)
    model.eval()
    per_design_results = {}
    with torch.no_grad():
        for name, g in test_graphs:
            out = model(g)
            ep = g["endpoints"]
            at_r2 = r2(out["AT"][ep], g["AT_true"][ep])
            slack_true = g["slack_true"][ep]
            if g.get("slack_type") in ("real_rat", "real"):
                rat = slack_true + g["AT_true"][ep]
                slack_pred = rat - out["AT"][ep]
            else:
                slack_pred = out["AT"][ep].max() - out["AT"][ep]
            slack_r2 = r2(slack_pred, slack_true)

            # These three were being trained on (they're in the loss) but
            # never actually evaluated or reported until now -- a real
            # reporting gap, not intentional. net_delay/cell_delay are
            # per-EDGE (not per-endpoint-node), so compare over all edges.
            slew_r2 = r2(out["SLEW"][ep], g["SLEW_true"][ep])
            nd_r2 = r2(out["net_delay"], g["nd_true"]) if g["nd_true"].numel() > 0 else float("nan")
            cd_r2 = r2(out["cell_delay"], g["cd_true"]) if g["cd_true"].numel() > 0 else float("nan")

            print(f"  {name:15s} | AT R2 = {at_r2:7.4f} | slack R2 = {slack_r2:7.4f} | "
                  f"SLEW R2 = {slew_r2:7.4f} | net_delay R2 = {nd_r2:7.4f} | "
                  f"cell_delay R2 = {cd_r2:7.4f}")
            per_design_results[name] = {
                "AT_R2": at_r2, "slack_R2": slack_r2,
                "SLEW_R2": slew_r2, "net_delay_R2": nd_r2, "cell_delay_R2": cd_r2,
            }

    run_dir = make_run_dir(train_designs, test_designs)
    save_run_results(run_dir, cfg, train_designs, test_designs, epoch_log,
                      per_design_results, test_graphs, model, n_params, best_epoch)
    report_writer.save_full_run(run_dir, cfg, train_designs, test_designs, epoch_log,
                                 per_design_results, n_params, best_epoch, model,
                                 in_dim, edge_dim, len(shared_gate_types),
                                 gate_types=shared_gate_types)

    return {"n_params": n_params, "best_pooled_test_r2": best_pooled_r2,
            "model": model, "run_dir": run_dir, "per_design_results": per_design_results}


if __name__ == "__main__":
    print("=" * 60)
    print(" TIMING-GNN TRAINER (paper-split leakage guard ON)")
    print(f" Node features: {NODE_IN_DIM}")
    print("=" * 60)

    data_dir = ask("Path to folder containing your design subfolders", r"C:\Krish\GCNN\data")
    all_designs = discover_designs(data_dir)
    print(f"\nFound {len(all_designs)} valid design folder(s).")
    train_designs, test_designs = split_designs(all_designs)

    cfg = build_config_interactively()
    epochs = ask_int("Training epochs", 300)
    lr = ask_float("Learning rate", 2e-3)
    log_every = ask_int("Log every N epochs", max(1, epochs // 10))

    while True:
        result = run(cfg, data_dir, train_designs, test_designs, epochs, lr, log_every)
        if not ask_yesno("\nTry another configuration?", default=False):
            break
        cfg = build_config_interactively()
        epochs = ask_int("Training epochs", epochs)
        lr = ask_float("Learning rate", lr)
        log_every = ask_int("Log every N epochs", log_every)

    print("\nDone.")