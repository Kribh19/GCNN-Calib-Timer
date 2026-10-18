"""
conformal.py
============
Split conformal prediction for slack, with DESIGN-LEVEL calibration.

Consumes predictions.csv exactly as saved by train_paper_split.py:
    design, split, pin_idx, AT_true, AT_pred, slack_true, slack_pred

WHY DESIGN-LEVEL, NOT PIN-LEVEL POOLED CALIBRATION:
Pins within one design share correlated structure (same clock period,
same placement, same LUT tables) -- pooling all pins from all designs
into one calibration set would treat them as exchangeable when they're
not fully independent. Calibrating on separate held-out DESIGNS (your
custom pool: uni_counter4b, cla8b, gcd, seq_detector, priority_enc)
and testing coverage on the paper's official test designs is the
correct unit for this guarantee -- this matches your project's stated
"design-level conformal calibration (F3 fix)" plan.

METHOD: standard split conformal on the ABSOLUTE ERROR nonconformity
score (Vovk et al.):
    score_i = |slack_true_i - slack_pred_i|
    q_hat   = the ceil((n+1)*(1-alpha))/n empirical quantile of scores
              on the calibration set
    interval for a new point = [pred - q_hat, pred + q_hat]

This gives a MARGINAL coverage guarantee: P(slack_true in interval)
>= 1-alpha, averaged over the calibration+test exchangeability
assumption. It does NOT guarantee coverage on any single specific
design -- that's exactly what the "empirical coverage per design"
report below is for, so you can SEE where the guarantee holds loosely
vs tightly, not just trust it blindly.

USAGE:
    python conformal.py --predictions path/to/predictions.csv \
        --calib_designs uni_counter4b cla8b gcd seq_detector priority_enc \
        --test_designs aes192 jpeg_encoder spm synth_ram usbf_device xtea y_huff \
        --alpha 0.1

NOTE: predictions.csv as currently saved by train_paper_split.py only
contains rows for the PAPER TEST designs (split=="test"). To actually
calibrate on the custom pool, you must ALSO run train_paper_split.py
(or a small variant) with those 5 custom designs included as an
additional "calibration" split, OR run inference-only on a trained
model against those 5 designs and append rows with split=="calib".
A helper for that (append_calibration_predictions) is included below.
This script does NOT fabricate calibration data if it's missing --
it will error out loudly instead, per project rule: no code should
silently proceed on data it doesn't actually have.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict


def load_predictions(path):
    """
    Returns dict: design_name -> list of (slack_true, slack_pred) tuples,
    plus a parallel dict for AT.

    Accepts EITHER:
      - a directory containing one <design>.csv per design (current
        train_paper_split.py output format, e.g. results/<run>/predictions/)
      - a single combined CSV file with a 'design' column (older format /
        manually combined file)
    """
    if os.path.isdir(path):
        csv_paths = sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")
        )
        if not csv_paths:
            raise FileNotFoundError(
                f"No .csv files found in directory {path}. This script does not "
                f"proceed without real prediction data -- run train_paper_split.py "
                f"first and point --predictions at its predictions/ directory."
            )
    elif os.path.exists(path):
        csv_paths = [path]
    else:
        raise FileNotFoundError(
            f"predictions path not found: {path}. This script does not "
            f"proceed without real prediction data -- run train_paper_split.py "
            f"first (and/or generate calibration-pool predictions) and point "
            f"--predictions at that output (either a single CSV or the "
            f"predictions/ directory)."
        )

    slack_by_design = defaultdict(list)
    at_by_design = defaultdict(list)
    split_by_design = {}
    required_cols = {"design", "split", "slack_true", "slack_pred", "AT_true", "AT_pred"}

    for csv_path in csv_paths:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{csv_path} is missing required columns: {missing}. "
                    f"Found columns: {reader.fieldnames}. This does not match "
                    f"the format train_paper_split.py saves -- check the file."
                )
            for row in reader:
                d = row["design"]
                slack_by_design[d].append((float(row["slack_true"]), float(row["slack_pred"])))
                at_by_design[d].append((float(row["AT_true"]), float(row["AT_pred"])))
                split_by_design[d] = row["split"]

    return slack_by_design, at_by_design, split_by_design


def conformal_quantile(scores, alpha):
    """
    Standard split-conformal finite-sample-correct quantile:
        q_hat = the ceil((n+1)*(1-alpha)) / n empirical quantile of scores
    scores: list of nonneg nonconformity scores from the CALIBRATION set only.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Empty calibration score set -- cannot compute quantile.")
    scores_sorted = sorted(scores)
    # rank is 1-indexed; clamp to n (i.e. take the max score if alpha is
    # small enough that ceil pushes past the available data -- this makes
    # the interval maximally conservative in that edge case rather than
    # silently underindexing)
    rank = min(n, math.ceil((n + 1) * (1 - alpha)))
    return scores_sorted[rank - 1]


def run_conformal(predictions_path, calib_designs, test_designs, alpha=0.1,
                   out_path=None):
    slack_by_design, at_by_design, split_by_design = load_predictions(predictions_path)

    present = set(slack_by_design.keys())
    missing_calib = [d for d in calib_designs if d not in present]
    missing_test = [d for d in test_designs if d not in present]
    if missing_calib:
        raise ValueError(
            f"Calibration designs not found in predictions.csv: {missing_calib}. "
            f"Available designs: {sorted(present)}. You likely need to run "
            f"inference on these designs and append their predictions first -- "
            f"see append_calibration_predictions() in this file."
        )
    if missing_test:
        raise ValueError(
            f"Test designs not found in predictions.csv: {missing_test}. "
            f"Available designs: {sorted(present)}."
        )

    # ---- build calibration score set from calib_designs only ----
    calib_scores = []
    for d in calib_designs:
        for true_v, pred_v in slack_by_design[d]:
            calib_scores.append(abs(true_v - pred_v))

    q_hat = conformal_quantile(calib_scores, alpha)

    print(f"\nCalibration designs ({len(calib_designs)}): {calib_designs}")
    print(f"Calibration pins used: {len(calib_scores)}")
    print(f"alpha = {alpha}  (target coverage = {1 - alpha:.1%})")
    print(f"q_hat (half-interval-width) = {q_hat:.6f} ns\n")

    # ---- apply to test designs, report per-design + pooled coverage ----
    results = {"alpha": alpha, "target_coverage": 1 - alpha, "q_hat": q_hat,
               "calib_designs": calib_designs, "n_calib_pins": len(calib_scores),
               "per_design": {}}

    pooled_covered = 0
    pooled_total = 0
    pooled_widths = []

    print(f"{'design':15s} | {'n_pins':>7s} | {'coverage':>9s} | {'target':>7s} | {'mean_width':>10s}")
    print("-" * 70)
    for d in test_designs:
        covered = 0
        widths = []
        for true_v, pred_v in slack_by_design[d]:
            lo, hi = pred_v - q_hat, pred_v + q_hat
            if lo <= true_v <= hi:
                covered += 1
            widths.append(hi - lo)
        n = len(slack_by_design[d])
        coverage = covered / n if n > 0 else float("nan")
        mean_width = sum(widths) / n if n > 0 else float("nan")

        print(f"{d:15s} | {n:7d} | {coverage:8.1%} | {1-alpha:6.1%} | {mean_width:10.4f}")
        results["per_design"][d] = {"n_pins": n, "coverage": coverage, "mean_width": mean_width}

        pooled_covered += covered
        pooled_total += n
        pooled_widths.extend(widths)

    pooled_coverage = pooled_covered / pooled_total if pooled_total > 0 else float("nan")
    pooled_mean_width = sum(pooled_widths) / len(pooled_widths) if pooled_widths else float("nan")
    print("-" * 70)
    print(f"{'POOLED':15s} | {pooled_total:7d} | {pooled_coverage:8.1%} | {1-alpha:6.1%} | {pooled_mean_width:10.4f}")

    results["pooled"] = {"n_pins": pooled_total, "coverage": pooled_coverage,
                          "mean_width": pooled_mean_width}

    if pooled_coverage < (1 - alpha) - 0.02:
        print(f"\n  [WARNING] Pooled coverage ({pooled_coverage:.1%}) is meaningfully "
              f"below target ({1-alpha:.1%}). This can mean: (a) calibration and "
              f"test designs are not exchangeable enough (e.g. very different "
              f"circuit structure/size), or (b) too few calibration pins for a "
              f"stable quantile estimate. Investigate before trusting this "
              f"interval -- do not just report q_hat as-is.")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [conformal] results saved to: {out_path}")

    return results


def append_calibration_predictions(model, calib_graphs, existing_predictions_csv,
                                    out_path=None):
    """
    Helper: run a trained model (inference only, groundtruth=False) on the
    custom calibration-pool designs and APPEND their predictions to an
    existing predictions.csv (or write a new one), tagging split="calib".

    calib_graphs: list of (design_name, graph_dict) already produced by
    real_graph_to_tensors() + move_to_device(), same as test_graphs in
    train_paper_split.py.

    This is a helper, not auto-run -- call it explicitly from your own
    script once you have a trained model in memory, since conformal.py
    itself has no business loading/training a model.
    """
    import torch
    out_path = out_path or existing_predictions_csv

    rows = []
    if os.path.exists(existing_predictions_csv):
        with open(existing_predictions_csv, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
    else:
        fieldnames = ["design", "split", "pin_idx", "AT_true", "AT_pred",
                      "slack_true", "slack_pred"]

    model.eval()
    with torch.no_grad():
        for name, g in calib_graphs:
            out = model(g)  # groundtruth=False by default -- real inference
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
            for pid, at_t, at_p, sl_t, sl_p in zip(
                    pin_ids, at_true, at_pred, slack_true, slack_pred):
                rows.append({"design": name, "split": "calib", "pin_idx": pid,
                             "AT_true": at_t, "AT_pred": at_p,
                             "slack_true": sl_t, "slack_pred": sl_p})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[conformal] appended calibration predictions -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split conformal prediction for slack, design-level calibration.")
    ap.add_argument("--predictions", required=True, help="Path to predictions.csv")
    ap.add_argument("--calib_designs", nargs="+",
                     default=["uni_counter4b", "cla8b", "gcd", "seq_detector", "priority_enc"],
                     help="Held-out calibration pool designs (default: your custom pool)")
    ap.add_argument("--test_designs", nargs="+",
                     default=["aes192", "jpeg_encoder", "spm", "synth_ram",
                              "usbf_device", "xtea", "y_huff"],
                     help="Paper's official test designs (default: the 7)")
    ap.add_argument("--alpha", type=float, default=0.1,
                     help="Miscoverage rate; 0.1 = 90%% target coverage")
    ap.add_argument("--out", default=None, help="Optional path to save results JSON")
    args = ap.parse_args()

    run_conformal(args.predictions, args.calib_designs, args.test_designs,
                  alpha=args.alpha, out_path=args.out)
