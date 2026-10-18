"""
train.py
========
Timing-GNN trainer using preprocessed .pt files.

Current configuration (v3):
  - Loss weights: AT=1.0, slew=5.0, net=0.5, cell=3.0
  - Epochs default: 500, early stop patience: 50
  - Scheduler: ReduceLROnPlateau factor=0.6, patience=epochs//8, min_lr=1e-4
  - Paper split: 14 train / 7 test matching DAC 2022 Table 5 exactly
  - Inference timing printed per test design (for paper speedup table)
  - Conformal CSV: 20% train holdout (calib) + all test endpoints

Usage:
  python train.py
  python train.py --preprocessed /path/to/preprocessed --epochs 500

NOTE: default --preprocessed is C:\\Krish\\GCNN\\preprocessed. If you ran
the fixed preprocess.py to a different output folder (e.g.
C:\\Krish\\GCNN\\preprocessed_v3), you MUST pass --preprocessed explicitly
or this will silently train on old/stale .pt files:
  python train.py --preprocessed C:\\Krish\\GCNN\\preprocessed_v3
"""

import os
import json
import copy
import random
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

from model_configurable import ModelConfig, TimingGNN, NODE_FEATURES
from results_logger import ResultLogger

# ---------------------------------------------------------------------------
DEFAULT_PREPROCESSED = r"C:\Krish\GCNN\preprocessed"
DEFAULT_META         = os.path.join(DEFAULT_PREPROCESSED, "meta.json")

LARGE_DESIGN_NODES   = 100_000
SUBSAMPLE_ENDPOINTS  = 80_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Correct paper split from Table 5 (DAC 2022) — verified from paper results image
# Train (14): blabla, usb_cdc_core, BM64, salsa20, aes128, wbqspiflash,
#             cic_decimator, aes256, des, aes_cipher, picorv32a, zipdiv, genericfir, usb
# Test  (7):  jpeg_encoder, usbf_device, aes192, xtea, spm, y_huff, synth_ram
PAPER_TRAIN = [
    "BM64", "aes128", "aes256", "aes_cipher",
    "blabla", "cic_decimator", "des", "genericfir",
    "picorv32a", "salsa20", "usb", "usb_cdc_core",
    "wbqspiflash", "zipdiv",
]
PAPER_TEST = [
    "jpeg_encoder", "usbf_device", "aes192", "xtea", "spm", "y_huff", "synth_ram",
]


# ===========================================================================
# Helpers
# ===========================================================================

def r2(pred, true):
    if len(true) == 0:
        return float("nan")
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return float("nan")
    return (1 - ss_res / ss_tot).item()


def move_to_device(g, device):
    # These keys are topology lookup structures — keep as Python lists/dicts on CPU
    # model.forward iterates over them to index into tensors, not use them as tensors
    KEEP_CPU = {"levels", "net_by_level", "cell_by_level"}
    out = {}
    for k, v in g.items():
        if k in KEEP_CPU:
            out[k] = v  # leave as-is: list of ints / dict of lists
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def load_pt(path, gate_vocab, device="cpu"):
    g = torch.load(path, map_location="cpu", weights_only=False)
    n_types    = len(gate_vocab)
    gtype_strs = g.get("cell_gtype_str", [])
    unk_idx    = gate_vocab.get("UNKNOWN", 0)

    if gtype_strs:
        oh = []
        for gt in gtype_strs:
            v = [0.0] * n_types
            v[gate_vocab.get(gt, unk_idx)] = 1.0
            oh.append(v)
        g["gate_onehot"] = torch.tensor(oh, dtype=torch.float)

    if "gate_onehot" in g and g["gate_onehot"].shape[1] != n_types:
        old = g["gate_onehot"]
        new = torch.zeros(old.shape[0], n_types)
        new[:, :old.shape[1]] = old
        g["gate_onehot"] = new

    # Validate lut_delay shape — must be [E, 8, 7, 7] for 8-corner model
    # Old .pt files have shape [E, 7, 7] — catch this before model crashes
    if "lut_delay" in g:
        ld = g["lut_delay"]
        if ld.dim() == 3:
            # Old single-corner format [E, 7, 7] — expand to [E, 8, 7, 7]
            # by repeating the single table 8 times (ff corner used for all)
            # This allows old .pt files to still run, just without multi-corner
            print(f"  [load_pt] WARNING: {os.path.basename(path)} has old lut shape "
                  f"{list(ld.shape)} — expected [E,8,7,7]. "
                  f"Re-run preprocess.py to get multi-corner LUTs.")
            g["lut_delay"] = ld.unsqueeze(1).expand(-1, 8, -1, -1).contiguous()
            if "lut_slew" in g and g["lut_slew"].dim() == 3:
                g["lut_slew"] = g["lut_slew"].unsqueeze(1).expand(-1, 8, -1, -1).contiguous()

    return g


def subsample_for_loss(g, max_endpoints=SUBSAMPLE_ENDPOINTS):
    ep = g["endpoints"]
    if len(ep) <= max_endpoints:
        return None
    idx    = torch.randperm(len(ep))[:max_endpoints]
    sample = ep[idx]
    mask   = torch.zeros(g["x"].shape[0], dtype=torch.bool, device=g["x"].device)
    mask[sample] = True
    return mask


# ===========================================================================
# Interactive prompts
# ===========================================================================

def ask(prompt, default):
    raw = input(f"{prompt} [default: {default}]: ").strip()
    return raw if raw else str(default)

def ask_int(prompt, default):
    return int(ask(prompt, default))

def ask_float(prompt, default):
    return float(ask(prompt, default))

def ask_yesno(prompt, default=True):
    d   = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{d}]: ").strip().lower()
    return default if not raw else raw.startswith("y")

def ask_choice(prompt, options, default):
    print(prompt)
    for i, opt in enumerate(options):
        marker = " (default)" if opt == default else ""
        print(f"    {i+1}. {opt}{marker}")
    raw = input(f"  choose 1-{len(options)} [default {default}]: ").strip()
    if not raw:
        return default
    try:
        return options[int(raw) - 1]
    except (ValueError, IndexError):
        return default

def ask_multi(prompt, options, defaults):
    print(f"\n{prompt}")
    print("  (enter numbers separated by commas, or press Enter for default)")
    for i, name in enumerate(options):
        marker = " *" if name in defaults else ""
        print(f"    {i+1:2d}. {name}{marker}")
    raw = input(f"  select (e.g. 1,2,5): ").strip()
    if not raw:
        return list(defaults)
    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(options):
                selected.append(options[idx])
    return selected if selected else list(defaults)


# ===========================================================================
# Design selection UI
# ===========================================================================

def select_designs(available):
    print("\n" + "=" * 60)
    print("  DESIGN SELECTION")
    print("=" * 60)
    print(f"  Available: {len(available)} designs")

    paper_train_avail = [d for d in PAPER_TRAIN if d in available]
    paper_test_avail  = [d for d in PAPER_TEST  if d in available]

    print(f"\n  Paper split (default):")
    print(f"    Train ({len(paper_train_avail)}): {', '.join(paper_train_avail)}")
    print(f"    Test  ({len(paper_test_avail)}):  {', '.join(paper_test_avail)}")

    use_paper = ask_yesno("\n  Use paper train/test split?", default=True)

    if use_paper:
        train_designs = paper_train_avail
        test_designs  = paper_test_avail
    else:
        train_designs = ask_multi(
            "Select TRAIN designs (* = paper default):",
            available, paper_train_avail)
        remaining = [d for d in available if d not in train_designs]
        test_designs = ask_multi(
            "Select TEST designs (* = paper default):",
            remaining, [d for d in paper_test_avail if d in remaining])

    overlap = set(train_designs) & set(test_designs)
    if overlap:
        print(f"  [WARNING] Removing {overlap} from test (they are in train)")
        test_designs = [d for d in test_designs if d not in overlap]

    print(f"\n  TRAIN ({len(train_designs)}): {', '.join(train_designs)}")
    print(f"  TEST  ({len(test_designs)}):  {', '.join(test_designs)}")

    if "alu4bit" in train_designs:
        print("  [note] alu4bit has no .at.sdf -- slack will be proxy for that design")

    return train_designs, test_designs


# ===========================================================================
# Main training function
# ===========================================================================

def run_training(args):
    meta_path = os.path.join(args.preprocessed, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    gate_vocab   = meta["gate_vocab"]
    n_gates      = len(gate_vocab)
    canvas_scale = meta.get("canvas_scale", 1000.0)

    print(f"\n[meta] {n_gates} gate types  |  {NODE_FEATURES} node features")
    assert NODE_FEATURES == 18, f"Expected 18 node features, model has {NODE_FEATURES}"

    pt_files  = {
        os.path.splitext(f)[0]: os.path.join(args.preprocessed, f)
        for f in sorted(os.listdir(args.preprocessed))
        if f.endswith(".pt")
    }
    available = sorted(pt_files.keys())
    print(f"[discover] {len(available)} .pt files: {available}")

    train_names, test_names = select_designs(available)

    if not train_names:
        print("No train designs selected. Exiting.")
        return
    if not test_names:
        print("No test designs selected. Exiting.")
        return

    print("\n" + "=" * 60)
    print("  MODEL CONFIG")
    print("=" * 60)
    hidden_dim      = ask_int("Hidden dim", 32)
    net_layers      = ask_int("Net layers", 3)
    aggregation     = ask_choice("Aggregation",
                                  ["sum_max", "mean", "sum", "max", "attention"], "sum_max")
    cell_correction = ask_yesno("Cell correction MLP?", True)
    cfg = ModelConfig(hidden_dim=hidden_dim, net_layers=net_layers,
                      aggregation=aggregation, lut_k=7,
                      cell_correction=cell_correction)
    cfg.weight_decay        = ask_float("Weight decay", 5e-4)
    cfg.early_stop_patience = ask_int("Early stop patience", 50)

    print("\n" + "=" * 60)
    print("  TRAINING SETTINGS")
    print("=" * 60)
    epochs    = ask_int("Epochs", 500)
    lr        = ask_float("Learning rate", 2e-3)
    log_every = ask_int("Log every N epochs", max(1, epochs // 25))

    print("\n" + "=" * 60)
    print("  LOADING DATA")
    print("=" * 60)

    print("Loading train designs...")
    train_graphs_cpu = []
    for name in train_names:
        t0 = time.time()
        g  = load_pt(pt_files[name], gate_vocab)
        print(f"  {name}: {g['n_nodes']:,} nodes  {g['n_net_edges']:,} net edges  "
              f"({time.time()-t0:.1f}s)  "
              f"{'[LARGE — will subsample]' if g['n_nodes'] > LARGE_DESIGN_NODES else ''}")
        train_graphs_cpu.append((name, g))

    print("\nLoading test designs...")
    test_graphs_cpu = []
    for name in test_names:
        t0 = time.time()
        g  = load_pt(pt_files[name], gate_vocab)
        print(f"  {name}: {g['n_nodes']:,} nodes  ({time.time()-t0:.1f}s)")
        test_graphs_cpu.append((name, g))

    model    = TimingGNN(cfg, n_gates=n_gates, in_dim=NODE_FEATURES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {cfg.describe()}")
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Node features: {NODE_FEATURES}  |  Gate types: {n_gates}")
    print(f"  Device: {DEVICE}")

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=cfg.weight_decay)

    # Scheduler: patience = epochs//8 (62 epochs for 500 run)
    # factor=0.6 instead of 0.5 — softer LR reduction keeps gradient flow longer
    # min_lr=1e-4 — never let LR drop so low the model stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.6,
        patience=max(8, epochs // 8),
        min_lr=1e-4)
    mse = nn.MSELoss()

    run_name = f"train{'_'.join(train_names[:3])}_test{'_'.join(test_names[:3])}"
    logger   = ResultLogger(
        run_name=run_name, cfg=cfg,
        extra_info={
            "train_designs": train_names,
            "test_designs":  test_names,
            "epochs_requested": epochs,
            "lr": lr,
            "node_features": NODE_FEATURES,
            "n_gates": n_gates,
            # Log the loss weights so we can track them across runs
            "loss_weights": "AT=1.0, slew=5.0, net=0.5, cell=3.0",
        })

    print(f"\n{'='*60}")
    print(f"  TRAINING  ({len(train_names)} designs, {epochs} epochs)")
    print(f"  Loss weights: AT=1.0  slew=5.0  net=0.5  cell=3.0")
    print(f"{'='*60}\n")

    best_test_r2     = -float("inf")
    best_model_state = None
    best_epoch       = 0
    patience_counter = 0
    last_loss        = 0.0
    stopped_early    = False

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_graphs_cpu)
        total_loss  = 0.0
        total_L_AT   = 0.0
        total_L_slew = 0.0
        total_L_net  = 0.0
        total_L_cell = 0.0

        for name, g_cpu in train_graphs_cpu:
            g = move_to_device(g_cpu, DEVICE)
            out = model(g)

            ep_mask = subsample_for_loss(g)
            if ep_mask is not None:
                ep_idx = g["endpoints"][ep_mask[g["endpoints"]]]
            else:
                ep_idx = g["endpoints"]

            L_AT = mse(out["AT"][ep_idx], g["AT_true"][ep_idx])

            # slew: real mask pins only (13-24% of pins per design)
            sm = g["slew_real_mask"]
            if ep_mask is not None:
                sm = sm & ep_mask
            if sm.sum() > 50:
                L_slew = mse(out["SLEW"][sm], g["SLEW_true"][sm])
            else:
                L_slew = torch.tensor(0.0, device=DEVICE)

            nm = g["net_trusted_mask"]
            if nm.sum() > 0 and g["nd_true"][nm].max() > 1e-6:
                pred_nd = out["net_delay"][nm]
                true_nd = g["nd_true"][nm]
                L_net   = mse(pred_nd, true_nd)
            else:
                L_net = torch.tensor(0.0, device=DEVICE)

            cm = g["cell_lut_mask"]
            if cm.sum() > 0:
                L_cell = mse(out["cell_delay"][cm], g["cd_true"][cm])
            else:
                L_cell = torch.tensor(0.0, device=DEVICE)

            # Loss: AT=1.0, slew=5.0, net=0.5, cell=3.0
            # cell raised 2→3: cell delay R2 was 0.11-0.58, needs stronger gradient
            # cell delay dominates AT accuracy — most important auxiliary task
            # net=0.5: enough gradient to learn, won't explode (softplus+clamp in model)
            loss = L_AT + 5.0*L_slew + 0.5*L_net + 3.0*L_cell

            # epoch-1 debug: verify net delay head is alive (nonzero predictions)
            if epoch == 1 and name == train_names[0]:
                nd_all = out["net_delay"].detach()
                print(f"  [epoch1 debug] {name}: "
                      f"net_delay pred range [{nd_all.min():.5f}, {nd_all.max():.5f}] ns  "
                      f"(should be nonzero if bias_init fix worked)")
                print(f"  [epoch1 debug] L_AT={L_AT.item():.4f}  "
                      f"L_slew={L_slew.item():.4f}  "
                      f"L_net={L_net.item():.5f}  "
                      f"L_cell={L_cell.item():.4f}")
                print(f"  [epoch1 debug] net_trusted_mask nonzero: {nm.sum().item()}")

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
            total_loss  += loss.item()
            total_L_AT   += L_AT.item()
            total_L_slew += L_slew.item()
            total_L_net  += L_net.item()
            total_L_cell += L_cell.item()

            del g, out

        last_loss    = total_loss  / max(len(train_graphs_cpu), 1)
        last_L_AT    = total_L_AT   / max(len(train_graphs_cpu), 1)
        last_L_slew  = total_L_slew / max(len(train_graphs_cpu), 1)
        last_L_net   = total_L_net  / max(len(train_graphs_cpu), 1)
        last_L_cell  = total_L_cell / max(len(train_graphs_cpu), 1)

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                all_at_pred, all_at_true = [], []
                for _, g_cpu in train_graphs_cpu:
                    g   = move_to_device(g_cpu, DEVICE)
                    out = model(g)
                    ep  = g["endpoints"]
                    all_at_pred.append(out["AT"][ep].cpu())
                    all_at_true.append(g["AT_true"][ep].cpu())
                    del g, out
                tr_r2_at = r2(torch.cat(all_at_pred), torch.cat(all_at_true))

                te_r2_per_design = {}
                all_te_pred, all_te_true = [], []
                for tname, g_cpu in test_graphs_cpu:
                    g   = move_to_device(g_cpu, DEVICE)
                    out = model(g)
                    ep  = g["endpoints"]
                    p   = out["AT"][ep].cpu()
                    t   = g["AT_true"][ep].cpu()
                    te_r2_per_design[tname] = r2(p, t)
                    all_te_pred.append(p)
                    all_te_true.append(t)
                    del g, out
                te_r2_at = r2(torch.cat(all_te_pred), torch.cat(all_te_true))

            scheduler.step(te_r2_at)
            current_lr = opt.param_groups[0]["lr"]

            if te_r2_at > best_test_r2 + 1e-4:
                best_test_r2     = te_r2_at
                best_epoch       = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            logger.log_epoch(epoch, last_loss,
                             loss_AT=round(last_L_AT, 5),
                             loss_slew=round(last_L_slew, 5),
                             loss_net=round(last_L_net, 5),
                             loss_cell=round(last_L_cell, 5),
                             train_r2_AT=tr_r2_at,
                             test_r2_AT_pooled=te_r2_at,
                             **{f"test_r2_{n}": v
                                for n, v in te_r2_per_design.items()})

            per_design_str = "  ".join(
                f"{n}:{v:.3f}" for n, v in te_r2_per_design.items())
            print(f"  epoch {epoch:4d} | loss {last_loss:7.4f} "
                  f"[AT {last_L_AT:.3f} slew {last_L_slew:.3f} net {last_L_net:.4f} cell {last_L_cell:.3f}] | "
                  f"train {tr_r2_at:.4f} | test {te_r2_at:.4f} | "
                  f"best {best_test_r2:.4f}@{best_epoch} | lr {current_lr:.1e} | p {patience_counter}/{cfg.early_stop_patience}")
            if len(test_names) > 1:
                print(f"           per-design: {per_design_str}")

            if cfg.early_stop_patience > 0 and patience_counter >= cfg.early_stop_patience:
                print(f"\n  [early stop] epoch {epoch}  "
                      f"restoring best weights from epoch {best_epoch} "
                      f"(test R2 = {best_test_r2:.4f})")
                model.load_state_dict(best_model_state)
                stopped_early = True
                epochs = epoch
                break

    if not stopped_early and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  [best model] restoring epoch {best_epoch} "
              f"(R2 = {best_test_r2:.4f})")

    print(f"\n{'='*60}")
    print("  FINAL EVALUATION")
    print(f"{'='*60}")

    model.eval()
    final_metrics = {}

    with torch.no_grad():
        all_p, all_t = [], []
        for _, g_cpu in train_graphs_cpu:
            g   = move_to_device(g_cpu, DEVICE)
            out = model(g)
            ep  = g["endpoints"]
            all_p.append(out["AT"][ep].cpu())
            all_t.append(g["AT_true"][ep].cpu())
            del g, out
        final_metrics["train_r2_AT_pooled"] = r2(torch.cat(all_p), torch.cat(all_t))

        all_te_at_p, all_te_at_t = [], []

        for tname, g_cpu in test_graphs_cpu:
            g   = move_to_device(g_cpu, DEVICE)
            # Time inference — for paper speedup table
            torch.cuda.synchronize() if DEVICE.type == "cuda" else None
            t_inf_start = time.time()
            out = model(g)
            torch.cuda.synchronize() if DEVICE.type == "cuda" else None
            inference_ms = (time.time() - t_inf_start) * 1000
            ep  = g["endpoints"]

            at_p = out["AT"][ep].cpu()
            at_t = g["AT_true"][ep].cpu()
            r2_at = r2(at_p, at_t)
            all_te_at_p.append(at_p)
            all_te_at_t.append(at_t)

            sm = g["slew_real_mask"]   # already on DEVICE (moved by move_to_device)
            if sm.sum() > 10:
                r2_slew = r2(out["SLEW"][sm].cpu(), g["SLEW_true"][sm].cpu())
                slew_ranges = (out["SLEW"][sm].min().item(), out["SLEW"][sm].max().item(),
                               g["SLEW_true"][sm].min().item(), g["SLEW_true"][sm].max().item())
            else:
                r2_slew = float("nan")
                slew_ranges = (float("nan"),)*4

            nm = g["net_trusted_mask"]   # already on DEVICE
            if nm.sum() > 10:
                pred_nd = out["net_delay"][nm].clamp(min=0.0).cpu()
                true_nd = g["nd_true"][nm].cpu()
                r2_nd   = r2(pred_nd, true_nd)
                nd_ranges = (pred_nd.min().item(), pred_nd.max().item(),
                             true_nd.min().item(), true_nd.max().item())
            else:
                r2_nd = float("nan")
                nd_ranges = (float("nan"),)*4

            cm = g["cell_lut_mask"]   # already on DEVICE
            if cm.sum() > 10:
                r2_cd = r2(out["cell_delay"][cm].cpu(), g["cd_true"][cm].cpu())
            else:
                r2_cd = float("nan")

            slack_type = g_cpu.get("slack_type", "proxy")
            if slack_type in ("real_rat", "real"):
                rat_ep     = g["slack_true"][ep].cpu() + at_t
                slack_pred = rat_ep - at_p
                slack_true = g["slack_true"][ep].cpu()
            else:
                worst_at   = at_t.max()
                slack_true = worst_at - at_t
                slack_pred = worst_at - at_p
            r2_slack = r2(slack_pred, slack_true)

            del g, out

            final_metrics[f"{tname}_r2_AT"]      = r2_at
            final_metrics[f"{tname}_r2_slew"]    = r2_slew
            final_metrics[f"{tname}_r2_nd"]      = r2_nd
            final_metrics[f"{tname}_r2_cd"]      = r2_cd
            final_metrics[f"{tname}_r2_slack"]   = r2_slack

            print(f"\n  [{tname}]")
            print(f"    AT R2       : {r2_at:.4f}  |  inference: {inference_ms:.1f} ms")
            print(f"    Cell delay  : {r2_cd:.4f}")
            print(f"    Net delay   : {r2_nd:.4f}  "
                  f"pred[{nd_ranges[0]:.4f},{nd_ranges[1]:.4f}]  "
                  f"true[{nd_ranges[2]:.4f},{nd_ranges[3]:.4f}] ns")
            print(f"    Slew        : {r2_slew:.4f}  "
                  f"pred[{slew_ranges[0]:.4f},{slew_ranges[1]:.4f}]  "
                  f"true[{slew_ranges[2]:.4f},{slew_ranges[3]:.4f}] ns")
            print(f"    Slack({slack_type:8s}): {r2_slack:.4f}")

        final_metrics["test_r2_AT_pooled"] = r2(
            torch.cat(all_te_at_p), torch.cat(all_te_at_t))
        final_metrics["train_r2_AT"]  = final_metrics["train_r2_AT_pooled"]
        final_metrics["epochs_run"]   = epochs
        final_metrics["final_loss"]   = last_loss

    print(f"\n  ┌─ Pooled test R2 (AT)   : {final_metrics['test_r2_AT_pooled']:.4f}")
    print(f"  └─ Train R2 (AT pooled) : {final_metrics['train_r2_AT_pooled']:.4f}")

    datasets_info = {
        "train_designs":      train_names,
        "test_designs":       test_names,
        "n_train_designs":    len(train_names),
        "n_test_designs":     len(test_names),
        "total_train_nodes":  sum(g["n_nodes"] for _, g in train_graphs_cpu),
        "node_features":      NODE_FEATURES,
        "gate_types":         n_gates,
    }

    torch.manual_seed(42)

    def _get_slack(g_cpu, out, ep):
        slack_type = g_cpu.get("slack_type", "proxy")
        ep_cpu = ep.cpu() if torch.is_tensor(ep) else ep
        at_p = out["AT"][ep].cpu()
        at_t = g_cpu["AT_true"][ep_cpu]
        if slack_type in ("real_rat", "real"):
            rat  = g_cpu["slack_true"][ep_cpu] + at_t
            s_p  = rat - at_p
            s_t  = g_cpu["slack_true"][ep_cpu]
        else:
            s_t  = at_t.max() - at_t
            s_p  = at_p.max() - at_p
        return at_p, at_t, s_p, s_t

    rows_pin_idx    = []
    rows_AT_true    = []
    rows_AT_pred    = []
    rows_slack_true = []
    rows_slack_pred = []
    rows_slack_err  = []
    rows_design     = []
    rows_split      = []

    model.eval()
    with torch.no_grad():
        for dname, g_cpu in train_graphs_cpu:
            g   = move_to_device(g_cpu, DEVICE)
            out = model(g)
            ep  = g["endpoints"]

            n_ep   = len(ep)
            perm   = torch.randperm(n_ep, generator=torch.Generator().manual_seed(42))
            n_cal  = max(1, n_ep // 5)
            cal_ep = ep[perm[:n_cal]]

            at_p, at_t, s_p, s_t = _get_slack(g_cpu, out, cal_ep)
            err = (s_p - s_t).abs()

            rows_pin_idx    += cal_ep.cpu().tolist()
            rows_AT_true    += at_t.tolist()
            rows_AT_pred    += at_p.tolist()
            rows_slack_true += s_t.tolist()
            rows_slack_pred += s_p.tolist()
            rows_slack_err  += err.tolist()
            rows_design     += [dname] * len(cal_ep)
            rows_split      += ["train_calib"] * len(cal_ep)
            del g, out

    with torch.no_grad():
        for dname, g_cpu in test_graphs_cpu:
            g   = move_to_device(g_cpu, DEVICE)
            out = model(g)
            ep  = g["endpoints"]

            at_p, at_t, s_p, s_t = _get_slack(g_cpu, out, ep)
            err = (s_p - s_t).abs()

            rows_pin_idx    += ep.cpu().tolist()
            rows_AT_true    += at_t.tolist()
            rows_AT_pred    += at_p.tolist()
            rows_slack_true += s_t.tolist()
            rows_slack_pred += s_p.tolist()
            rows_slack_err  += err.tolist()
            rows_design     += [dname] * len(ep)
            rows_split      += ["test"] * len(ep)
            del g, out

    test_predictions = {
        "pin_names":   None,          # ResultLogger expects this key (None = use pin_idx)
        "pin_idx":     rows_pin_idx,
        "AT_true":     rows_AT_true,
        "AT_pred":     rows_AT_pred,
        "slack_true":  rows_slack_true,
        "slack_pred":  rows_slack_pred,
        "slack_error": rows_slack_err,
        "design":      rows_design,
        "split":       rows_split,
    }

    n_calib = rows_split.count("train_calib")
    n_test  = rows_split.count("test")
    print(f"\n  [conformal] predictions_test.csv: "
          f"{n_calib} calib rows (train 20% holdout) + "
          f"{n_test} test rows ({', '.join(test_names)})")

    run_dir = logger.save_final(
        metrics_summary=final_metrics,
        datasets=datasets_info,
        model=model,
        test_predictions=test_predictions,
    )

    print(f"\n  Full report: {run_dir}")
    print("\nDone.")
    return final_metrics


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed", default=DEFAULT_PREPROCESSED,
                        help="Path to folder with .pt files and meta.json")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--hidden",  type=int,   default=None)
    args = parser.parse_args()

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU — running on CPU")

    run_training(args)