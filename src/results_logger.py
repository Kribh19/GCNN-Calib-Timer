"""
results_logger.py
===================
Saves a FULL report of every training run into its own folder under
results/, so you can compare runs later without re-training.

For each run you get:

    results/<timestamp>_<run_name>/
        report.md              <- human-readable summary (architecture,
                                   dataset, epoch curve, final accuracy,
                                   timing/slew/slack notes)
        run_info.json           <- same info, machine-readable
        epoch_log.csv           <- every epoch's loss + R2 (train/test)
        predictions_test.csv    <- per-pin AT / slew / slack, true vs pred,
                                   for the test design
        model_weights.pth       <- trained weights (optional, if you pass model)

USAGE (see interactive_train_real.py for a full example):

    from results_logger import ResultLogger

    logger = ResultLogger(run_name="spm_train_usb_test", cfg=cfg,
                           extra_info={"note": "first real-data sweep"})

    for epoch in range(1, epochs + 1):
        ... train step ...
        logger.log_epoch(epoch, loss.item(),
                          train_r2_AT=tr, test_r2_AT=te)

    logger.save_final(
        metrics_summary={...},
        datasets={...},
        model=model,
        test_predictions={...},
    )
"""
import os
import json
import csv
import time
import platform
import torch


RESULTS_ROOT = "results"


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


class ResultLogger:
    def __init__(self, run_name, cfg, results_root=RESULTS_ROOT, extra_info=None):
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
        self.run_id = f"{ts}_{safe_name}"
        self.dir = os.path.join(results_root, self.run_id)
        os.makedirs(self.dir, exist_ok=True)

        self.cfg = cfg
        self.extra_info = extra_info or {}
        self.epoch_log = []          # list of dicts, one per logged epoch
        self.start_time = time.time()

        print(f"[results_logger] new run folder -> {self.dir}")

    # ------------------------------------------------------------------
    # call this once per epoch (or every N epochs) during training
    # ------------------------------------------------------------------
    def log_epoch(self, epoch, loss, **metrics):
        row = {"epoch": epoch, "loss": float(loss)}
        for k, v in metrics.items():
            row[k] = float(v) if isinstance(v, (int, float)) else v
        self.epoch_log.append(row)

    # ------------------------------------------------------------------
    def _write_epoch_csv(self):
        if not self.epoch_log:
            return
        path = os.path.join(self.dir, "epoch_log.csv")
        fields = ["epoch", "loss"] + sorted(
            {k for row in self.epoch_log for k in row.keys()} - {"epoch", "loss"}
        )
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in self.epoch_log:
                w.writerow(row)

    # ------------------------------------------------------------------
    def _write_predictions_csv(self, name, pin_names, table):
        """table: dict of column_name -> list/tensor of per-pin values."""
        path = os.path.join(self.dir, f"predictions_{name}.csv")
        cols = list(table.keys())
        n = len(next(iter(table.values())))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pin_idx", "pin_name"] + cols)
            for i in range(n):
                pn = pin_names[i] if pin_names and i < len(pin_names) else i
                row = [i, pn]
                for c in cols:
                    val = table[c][i]
                    val = float(val) if torch.is_tensor(val) else val
                    row.append(round(float(val), 6) if isinstance(val, float) else val)
                w.writerow(row)

    # ------------------------------------------------------------------
    # call this once, at the very end of the run
    # ------------------------------------------------------------------
    def save_final(self, metrics_summary, datasets=None, model=None,
                    test_predictions=None, save_weights=True):
        """
        metrics_summary : dict, e.g.
            {"epochs_run": 300, "final_loss": 0.0123,
             "train_r2_AT": 0.91, "test_r2_AT": 0.74,
             "test_r2_slew": 0.6, "test_r2_net_delay": 0.8,
             "test_r2_cell_delay": 0.7, "test_r2_slack": 0.7}
        datasets : dict describing the designs used, e.g.
            {"train_design": "spm", "train_nodes": 1121, "train_net_edges": ...,
             "test_design": "usb", "test_nodes": 3361, ...}
        model : the trained nn.Module (used for param count + weight saving)
        test_predictions : dict of column_name -> per-pin values for the test
            design, e.g. {"AT_true":..., "AT_pred":..., "slew_true":...,
            "slew_pred":..., "slack_true":..., "slack_pred":...,
            "pin_names": [...]}
        """
        self._write_epoch_csv()

        n_params, device_str = None, "unknown"
        if model is not None:
            n_params = sum(p.numel() for p in model.parameters())
            try:
                device_str = str(next(model.parameters()).device)
            except StopIteration:
                pass
            if save_weights:
                try:
                    torch.save(model.state_dict(), os.path.join(self.dir, "model_weights.pth"))
                except Exception as e:
                    print(f"[results_logger] could not save weights: {e}")

        config_dict = self.cfg.__dict__ if hasattr(self.cfg, "__dict__") else dict(self.cfg)

        run_info = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(time.time() - self.start_time, 1),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device_str,
            "config": config_dict,
            "n_trainable_params": n_params,
            "datasets": datasets or {},
            "metrics": metrics_summary,
            "extra_info": self.extra_info,
        }
        with open(os.path.join(self.dir, "run_info.json"), "w") as f:
            json.dump(run_info, f, indent=2, default=str)

        if test_predictions is not None:
            pin_names = test_predictions.pop("pin_names", None)
            self._write_predictions_csv("test", pin_names, test_predictions)

        self._write_report_md(run_info)
        print(f"[results_logger] report saved -> {os.path.join(self.dir, 'report.md')}")
        return self.dir

    # ------------------------------------------------------------------
    def _write_report_md(self, run_info):
        cfg, ds, m = run_info["config"], run_info["datasets"], run_info["metrics"]
        L = []
        L.append(f"# Run report — {run_info['run_id']}")
        L.append("")
        L.append(f"- date: {run_info['timestamp']}")
        L.append(f"- duration: {run_info['duration_sec']} s")
        L.append(f"- device: {run_info['device']}")
        L.append(f"- torch: {run_info['torch']}  |  python: {run_info['python']}")
        L.append("")

        L.append("## Architecture (this run)")
        L.append("")
        for k, v in cfg.items():
            L.append(f"- **{k}**: {v}")
        if run_info["n_trainable_params"] is not None:
            L.append(f"- **trainable parameters**: {run_info['n_trainable_params']:,}")
        L.append("")

        L.append("## Dataset")
        L.append("")
        for k, v in ds.items():
            L.append(f"- **{k}**: {v}")
        L.append("")

        L.append("## Training curve")
        L.append("")
        L.append(f"- epochs run: {m.get('epochs_run', 'n/a')}")
        L.append(f"- final training loss: {_fmt(m.get('final_loss', 'n/a'))}")
        L.append("")
        rows = self.epoch_log
        if rows:
            extra_cols = sorted({k for r in rows for k in r if k not in ("epoch", "loss")})
            L.append("| epoch | loss | " + " | ".join(extra_cols) + " |")
            L.append("|---|---|" + "---|" * len(extra_cols))
            step = max(1, len(rows) // 25)          # keep the table short
            for row in rows[::step]:
                vals = [_fmt(row.get(c, "")) for c in extra_cols]
                L.append(f"| {row['epoch']} | {_fmt(row['loss'])} | " + " | ".join(vals) + " |")
            if rows[-1] is not rows[::step][-1]:
                row = rows[-1]
                vals = [_fmt(row.get(c, "")) for c in extra_cols]
                L.append(f"| {row['epoch']} | {_fmt(row['loss'])} | " + " | ".join(vals) + " | (final)")
        L.append("")

        L.append("## Final accuracy")
        L.append("")
        L.append("| metric | value |")
        L.append("|---|---|")
        for k, v in m.items():
            if k in ("epochs_run", "final_loss"):
                continue
            L.append(f"| {k} | {_fmt(v)} |")
        L.append("")

        L.append("## Timing / slew / slack — what these numbers mean here")
        L.append("")
        L.append("- **AT (arrival time)**: predicted vs. true pin arrival time, in ns, "
                  "propagated level-by-level through the DAG (net delay then cell delay, "
                  "MAX over incoming paths — the actual STA rule).")
        L.append("- **slew**: predicted vs. true transition time at each pin.")
        L.append("- **slack (proxy)**: computed as "
                  "`(worst-endpoint AT in this design) - (this endpoint's AT)`, separately "
                  "for true and predicted arrival times. This is a *relative* slack against "
                  "the design's own critical path, not real STA slack "
                  "(`required_time - arrival_time`), because no SDC clock constraint is "
                  "parsed by `parse_real_design.py`. Treat `test_r2_slack` as a diagnostic, "
                  "not a claim of true setup-slack accuracy.")
        L.append("- Full per-pin numbers (true vs. predicted AT / slew / slack) are in "
                  "`predictions_test.csv` in this same folder.")
        L.append("")

        if self.extra_info:
            L.append("## Notes")
            L.append("")
            for k, v in self.extra_info.items():
                L.append(f"- **{k}**: {v}")
            L.append("")

        with open(os.path.join(self.dir, "report.md"), "w") as f:
            f.write("\n".join(L))