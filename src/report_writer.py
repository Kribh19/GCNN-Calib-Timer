"""
report_writer.py
=================
Called automatically by train_paper_split.py at the end of every run.
Adds three things that were previously NOT being saved:

  1. model_checkpoint.pt  -- the actual trained weights (best epoch),
                              so a result can be reloaded and re-evaluated
                              later without retraining.
  2. source_snapshot/      -- a copy of gnn_model.py and train_paper_split.py
                              AS THEY EXISTED when this run happened. Per
                              project rule (current ground truth only /
                              no stale numbers surviving code changes):
                              a result without the exact code that produced
                              it is not reproducible and should not be
                              trusted months later.
  3. report.md             -- human-readable summary: config, epoch table,
                              per-design results, and an explicit diff
                              against the last known baseline numbers, so
                              regressions/improvements are visible at a
                              glance instead of requiring manual comparison.

This does NOT replace config.json / epoch_log.csv / predictions.csv --
those remain the raw machine-readable data. report.md is a rendering
of the same data for human reading.
"""
import os
import shutil
import torch

# Last known legitimate baseline, correct paper split (READ from project
# status as of this file's writing). Update this constant when a new
# run is confirmed as the new legitimate best -- do NOT let it silently
# go stale, since report.md's comparison column depends on it.
KNOWN_BASELINE_AT_R2 = {
    "xtea": 0.949,
    "spm": 0.928,
    # other designs: no previously established baseline in project status
    # at time of writing -- left out rather than guessed.
}
COMPETITOR_AT_R2 = {
    "xtea": 0.9534,   # E2ESlack
    "spm": 0.9605,    # E2ESlack
}


def save_model_checkpoint(run_dir, model, cfg, n_params, best_epoch, in_dim, edge_dim,
                           n_gates, gate_types=None):
    """
    Saves everything needed to reload this exact model later:
    weights + the architecture args that constructed it.

    gate_types: the ACTUAL list of gate type names used at training time
    (e.g. ["sky130_fd_sc_hd__inv_1", ..., "UNKNOWN"]), not just their count.
    This was a real, crash-causing gap: without the exact list, any later
    inference run (e.g. run_calibration_inference.py) rebuilds its own
    vocabulary from whatever designs it happens to see, which almost never
    matches the training vocabulary -- gate one-hot indices end up
    completely misaligned, and the width mismatch crashes the first
    Linear layer in CellLUTModule. Storing the real list makes reuse
    exact instead of reconstructed.
    """
    ckpt_path = os.path.join(run_dir, "model_checkpoint.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": cfg.hidden_dim,
        "net_layers": cfg.net_layers,
        "lut_k": cfg.lut_k,
        "cell_correction": cfg.cell_correction,
        "in_dim": in_dim,
        "edge_dim": edge_dim,
        "n_gates": n_gates,
        "gate_types": gate_types,  # the actual list -- may be None for old checkpoints
        "n_params": n_params,
        "best_epoch": best_epoch,
    }, ckpt_path)
    return ckpt_path


def snapshot_source(run_dir):
    """
    Copies the actual .py files in this directory into run_dir/source_snapshot/
    at the moment this run finished. If you later can't tell which code
    version produced a given result (a documented past failure mode of
    this project), this is the fix.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    snap_dir = os.path.join(run_dir, "source_snapshot")
    os.makedirs(snap_dir, exist_ok=True)
    copied = []
    for fname in ["gnn_model.py", "train_paper_split.py", "interactive_train_real.py",
                  "parse_real_design.py", "parse_liberty.py", "conformal.py",
                  "report_writer.py"]:
        src_path = os.path.join(src_dir, fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(snap_dir, fname))
            copied.append(fname)
    return snap_dir, copied


def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    try:
        if x != x:  # NaN
            return "NaN"
        return f"{x:.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def write_report(run_dir, cfg, train_designs, test_designs, epoch_log,
                  per_design_results, n_params, best_epoch, ckpt_path, snap_files):
    """
    Writes report.md summarizing the run, including an explicit comparison
    against KNOWN_BASELINE_AT_R2 and COMPETITOR_AT_R2 so improvements or
    regressions are visible without manually cross-referencing old logs.
    """
    lines = []
    lines.append(f"# Training Run Report")
    lines.append(f"")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append(f"")
    lines.append(f"## Config")
    lines.append(f"")
    lines.append(f"- Model: {cfg.describe()}")
    lines.append(f"- Weight decay: {cfg.weight_decay}")
    lines.append(f"- Early stop patience: {cfg.early_stop_patience}")
    lines.append(f"- Trainable parameters: {n_params:,}")
    lines.append(f"- Best epoch (by mean-per-design AT R2): {best_epoch}")
    lines.append(f"- Model checkpoint: `{os.path.basename(ckpt_path)}`")
    lines.append(f"- Source snapshot: `source_snapshot/` ({', '.join(snap_files)})")
    lines.append(f"")
    lines.append(f"## Data split")
    lines.append(f"")
    lines.append(f"- Train designs ({len(train_designs)}): {', '.join(train_designs)}")
    lines.append(f"- Test designs ({len(test_designs)}): {', '.join(test_designs)}")
    lines.append(f"")

    lines.append(f"## Training curve")
    lines.append(f"")
    lines.append(f"| epoch | loss | mean-per-design AT R2 | pooled AT R2 | lr |")
    lines.append(f"|---|---|---|---|---|")
    for row in epoch_log:
        epoch, loss, mean_r2, pooled_r2, lr = row
        lines.append(f"| {epoch} | {_fmt(loss)} | {_fmt(mean_r2)} | {_fmt(pooled_r2)} | {lr:.2e} |")
    lines.append(f"")

    # flag instability directly in the report, don't bury it
    mean_r2_vals = [row[2] for row in epoch_log]
    if len(mean_r2_vals) >= 3:
        max_drop = max(mean_r2_vals[i] - mean_r2_vals[i+1] for i in range(len(mean_r2_vals)-1))
        if max_drop > 0.2:
            lines.append(f"> **Note:** mean-per-design AT R2 dropped by as much as "
                         f"{max_drop:.3f} between consecutive logged epochs during this "
                         f"run. This is a real training-instability signal, not "
                         f"necessarily a bug -- but it means the FINAL numbers below "
                         f"came from a training process that was not monotonic, so "
                         f"re-running with a different seed could plausibly land on "
                         f"a different best epoch/result.")
            lines.append(f"")

    lines.append(f"## Per-design test results")
    lines.append(f"")
    lines.append(f"| design | AT R2 | slack R2 | SLEW R2 | net_delay R2 | cell_delay R2 | vs. baseline (AT) | vs. E2ESlack (AT) |")
    lines.append(f"|---|---|---|---|---|---|---|---|")
    for name, res in per_design_results.items():
        at_r2 = res["AT_R2"]
        slack_r2 = res["slack_R2"]
        slew_r2 = res.get("SLEW_R2")
        nd_r2 = res.get("net_delay_R2")
        cd_r2 = res.get("cell_delay_R2")
        baseline = KNOWN_BASELINE_AT_R2.get(name)
        competitor = COMPETITOR_AT_R2.get(name)
        base_str = f"{at_r2 - baseline:+.4f}" if baseline is not None else "no established baseline"
        comp_str = f"{at_r2 - competitor:+.4f}" if competitor is not None else "n/a"
        lines.append(f"| {name} | {_fmt(at_r2)} | {_fmt(slack_r2)} | {_fmt(slew_r2)} | "
                     f"{_fmt(nd_r2)} | {_fmt(cd_r2)} | {base_str} | {comp_str} |")
    lines.append(f"")
    lines.append(f"> Slack R2 shares AT's residual exactly for real_rat designs "
                 f"(see train_paper_split.py note) -- it is not independent evidence "
                 f"of slack-specific quality, only a different SS_tot scale.")
    lines.append(f"")
    lines.append(f"> SLEW R2 is checked against this pipeline's own LUT-derived SLEW_true "
                 f"(not filtered to only real-SDF slew pins) -- weaker signal than AT/"
                 f"net_delay/cell_delay R2 until a real-slew mask is wired in here too.")
    lines.append(f"")

    lines.append(f"## Reproducibility")
    lines.append(f"")
    lines.append(f"To reproduce this exact result: use the code in `source_snapshot/`, "
                 f"the same train/test design lists above, and load "
                 f"`{os.path.basename(ckpt_path)}` for the trained weights "
                 f"(architecture args are stored inside the checkpoint dict).")
    lines.append(f"")

    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def save_full_run(run_dir, cfg, train_designs, test_designs, epoch_log,
                   per_design_results, n_params, best_epoch, model,
                   in_dim, edge_dim, n_gates, gate_types=None):
    """
    Single entry point train_paper_split.py calls: does checkpoint save,
    source snapshot, and report.md generation together.
    """
    ckpt_path = save_model_checkpoint(run_dir, model, cfg, n_params, best_epoch,
                                       in_dim, edge_dim, n_gates, gate_types=gate_types)
    snap_dir, snap_files = snapshot_source(run_dir)
    report_path = write_report(run_dir, cfg, train_designs, test_designs, epoch_log,
                                per_design_results, n_params, best_epoch,
                                ckpt_path, snap_files)
    print(f"  [results]   model_checkpoint.pt, source_snapshot/, report.md")
    return {"checkpoint": ckpt_path, "source_snapshot": snap_dir, "report": report_path}
