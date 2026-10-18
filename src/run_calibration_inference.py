"""
run_calibration_inference.py
=============================
Runs a SAVED, TRAINED model (inference only -- no training, no ground
truth used in propagation) on your custom calibration-pool designs, and
writes their predictions into predictions/<design>.csv, split="calib",
inside an EXISTING run directory (the same one that already has your 7
test designs' predictions).

This exists because train_paper_split.py only ever ran inference on the
paper's 7 official TEST designs -- your 5 custom calibration designs
(uni_counter4b, cla8b, gcd, seq_detector, priority_enc) were never run
through the trained model at all, so conformal.py had nothing to
calibrate against.

USAGE:
    python run_calibration_inference.py --run_dir "results/20260803_132137_train20_test7" --data_dir "C:\\Krish\\GCNN\\data"

After this finishes, predictions/ in that run_dir will contain BOTH the
7 test-design CSVs (split=test) and the 5 calibration-design CSVs
(split=calib) -- exactly what conformal.py needs.

Then run:
    python conformal.py --predictions "results/20260803_132137_train20_test7/predictions" --alpha 0.1
"""
import argparse
import csv
import os

import torch

from gnn_model import ModelConfig, TimingGNN
from parse_real_design import build_graph
from interactive_train_real import real_graph_to_tensors, move_to_device, DEVICE


def r2(pred, true):
    """Same R2 as train_paper_split.py -- duplicated here to keep this
    script runnable standalone without importing train_paper_split."""
    if len(true) == 0:
        return float("nan")
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return float("nan")
    return (1 - ss_res / ss_tot).item()

DEFAULT_CALIB_DESIGNS = ["uni_counter4b", "cla8b", "gcd", "seq_detector", "priority_enc"]


def load_model_from_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ModelConfig(hidden_dim=ckpt["hidden_dim"], net_layers=ckpt["net_layers"],
                       lut_k=ckpt["lut_k"], cell_correction=ckpt["cell_correction"])
    model = TimingGNN(cfg, n_gates=ckpt["n_gates"],
                       in_dim=ckpt["in_dim"], edge_dim=ckpt["edge_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[calib-inference] loaded checkpoint: {ckpt_path}")
    print(f"[calib-inference]   in_dim={ckpt['in_dim']} edge_dim={ckpt['edge_dim']} "
          f"n_gates={ckpt['n_gates']} n_params={ckpt['n_params']:,} best_epoch={ckpt['best_epoch']}")
    return model, ckpt


def run(run_dir, data_dir, calib_designs):
    ckpt_path = os.path.join(run_dir, "model_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No model_checkpoint.pt found in {run_dir}. This script needs a "
            f"trained model to run inference with -- point --run_dir at a "
            f"results/<timestamp>_.../ folder that already has one."
        )
    model, ckpt = load_model_from_checkpoint(ckpt_path)

    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Use the EXACT gate vocabulary the model trained with, not a fresh
    # one rebuilt from calibration designs alone -- rebuilding from scratch
    # is what caused a real crash (mat1/mat2 shape mismatch) since
    # calibration designs almost never contain the same gate types, in
    # the same order, as the full training set.
    if ckpt.get("gate_types"):
        shared_gate_types = ckpt["gate_types"]
        print(f"\n[calib-inference] using saved training vocabulary "
              f"({len(shared_gate_types)} gate types) from checkpoint")
    else:
        print(f"\n[calib-inference] parsing {len(calib_designs)} calibration designs...")
        all_gate_types = set()
        calib_raws_tmp = []
        for d in calib_designs:
            design_dir = os.path.join(data_dir, d)
            if not os.path.isdir(design_dir):
                continue
            g = build_graph(design_dir, d)
            all_gate_types |= set(g["gate_of_out"].values())
            calib_raws_tmp.append((d, g))
        shared_gate_types = sorted(all_gate_types) or ["UNKNOWN"]
        if "UNKNOWN" not in shared_gate_types:
            shared_gate_types.append("UNKNOWN")
        print(f"\n  [WARNING] This checkpoint has NO saved gate_types (older checkpoint "
              f"format). Rebuilt a vocabulary from calibration designs alone "
              f"({len(shared_gate_types)} types) -- this is almost certainly "
              f"MISALIGNED with the {ckpt['n_gates']} gate types used at training "
              f"time. Predictions below are unreliable. Retrain and re-save with "
              f"the updated report_writer.py to fix this properly.")

    print(f"\n[calib-inference] parsing {len(calib_designs)} calibration designs...")
    calib_raws = []
    for d in calib_designs:
        design_dir = os.path.join(data_dir, d)
        if not os.path.isdir(design_dir):
            print(f"  [WARNING] design folder not found, skipping: {design_dir}")
            continue
        g = build_graph(design_dir, d)
        calib_raws.append((d, g))

    if len(shared_gate_types) != ckpt["n_gates"]:
        print(f"\n  [WARNING] Gate vocabulary size still mismatched after using "
              f"checkpoint's saved list: {len(shared_gate_types)} vs "
              f"{ckpt['n_gates']}. This should not happen if gate_types was "
              f"saved correctly -- investigate before trusting results.\n")

    print(f"\n{'design':15s} | {'n_pins':>7s} | {'AT R2':>8s} | {'slack R2':>9s} | {'slack_type':>10s}")
    print("-" * 65)
    calib_results = {}
    with torch.no_grad():
        for name, g_raw in calib_raws:
            g = move_to_device(real_graph_to_tensors(g_raw, shared_gate_types), DEVICE)
            if g["x"].shape[1] != ckpt["in_dim"]:
                print(f"  [SKIP] {name}: feature width {g['x'].shape[1]} != "
                      f"model's trained in_dim {ckpt['in_dim']} -- cannot run.")
                continue

            out = model(g)  # groundtruth=False (default) -- real inference
            ep = g["endpoints"]
            at_true_t = g["AT_true"][ep]
            at_pred_t = out["AT"][ep]
            slack_true_t = g["slack_true"][ep]
            slack_type = g.get("slack_type", "unknown")
            if slack_type in ("real_rat", "real"):
                rat_t = slack_true_t + at_true_t
                slack_pred_t = rat_t - at_pred_t
            else:
                slack_pred_t = at_pred_t.max() - at_pred_t

            at_r2 = r2(at_pred_t, at_true_t)
            slack_r2 = r2(slack_pred_t, slack_true_t)
            calib_results[name] = {"AT_R2": at_r2, "slack_R2": slack_r2, "slack_type": slack_type}
            print(f"{name:15s} | {len(ep):7d} | {at_r2:8.4f} | {slack_r2:9.4f} | {slack_type:>10s}")

            at_true = at_true_t.cpu().tolist()
            at_pred = at_pred_t.cpu().tolist()
            slack_true = slack_true_t.cpu().tolist()
            slack_pred = slack_pred_t.cpu().tolist()
            pin_ids = ep.cpu().tolist()

            design_path = os.path.join(pred_dir, f"{name}.csv")
            with open(design_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["design", "split", "pin_idx", "AT_true", "AT_pred",
                                  "slack_true", "slack_pred"])
                for pid, at_t, at_p, sl_t, sl_p in zip(
                        pin_ids, at_true, at_pred, slack_true, slack_pred):
                    writer.writerow([name, "calib", pid, at_t, at_p, sl_t, sl_p])

    print("-" * 65)
    print(f"\n  NOTE: designs whose slack_type is not 'real_rat'/'real' (i.e. "
          f"AT/RAT was derived, not from a real ARRIVALTIMES block in .at.sdf) "
          f"have slack_pred computed as a proxy (max-endpoint-AT minus this "
          f"pin's AT), which is NOT the same quantity as slack computed "
          f"against a real required-arrival-time. R2 for these designs is not "
          f"directly comparable to the paper-test-designs' slack R2 above, "
          f"and is exactly the likely cause of conformal.py's calibration "
          f"being wider than necessary (see conversation).")

    print(f"\n[calib-inference] done. predictions/ in {run_dir} now has both "
          f"test and calib splits. Run conformal.py pointing at that folder next.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                     help="Path to an existing results/<timestamp>.../ run folder with a model_checkpoint.pt")
    ap.add_argument("--data_dir", required=True, help="Path to your design data folder")
    ap.add_argument("--calib_designs", nargs="+", default=DEFAULT_CALIB_DESIGNS)
    args = ap.parse_args()
    run(args.run_dir, args.data_dir, args.calib_designs)
