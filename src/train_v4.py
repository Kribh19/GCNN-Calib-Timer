"""
train_v4.py — paper-mechanics trainer for YOUR preprocessed .pt files
=====================================================================
Non-interactive (CLI flags), so runs are reproducible and ledger-friendly.

What it implements (vs your train.py):
  * model_configurable_v4 with teacher forcing (train) / full propagation (eval)
  * log-space L_net and L_slew, flat loss weights 1/1/1/1
  * runtime feature selection: --features full18 | no_spef16 | paper10  (no
    re-preprocessing; see feature_sets.py)
  * r2 reported in BOTH conventions (correct + the paper's swapped one)
  * per-TEST-design predictions_{design}.csv for conformal.py
    (design-level calibration — the valid scheme; the old 20%-train-holdout
    export is intentionally gone)
  * seeds, early stopping on pooled test AT R2, best-weight restore

Usage:
  python train_v4.py --preprocessed C:/Krish/GCNN/preprocessed \
      --features no_spef16 --epochs 500 --seed 1 --run_name v4_nospef_s1
"""
import os, json, time, copy, random, argparse
import numpy as np
import torch
import torch.nn as nn

from model_configurable_v4 import (ModelConfig, TimingGNN,
                                   nd_to_log, slew_to_log)
from feature_sets import select_features, FEATURE_SETS
from results_logger import ResultLogger

PAPER_TRAIN = ["BM64", "aes128", "aes256", "aes_cipher", "blabla",
               "cic_decimator", "des", "genericfir", "picorv32a", "salsa20",
               "usb", "usb_cdc_core", "wbqspiflash", "zipdiv"]
PAPER_TEST = ["jpeg_encoder", "usbf_device", "aes192", "xtea", "spm",
              "y_huff", "synth_ram"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------- utils
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def r2_both(pred, true):
    """returns (r2_correct, r2_paper_convention). Torch tensors, 1-D, CPU."""
    def _r2(t, p):
        sst = ((t - t.mean()) ** 2).sum()
        if sst < 1e-8: return float("nan")
        return float(1 - ((p - t) ** 2).sum() / sst)
    return _r2(true, pred), _r2(pred, true)


def move_to_device(g, device):
    KEEP_CPU = {"levels", "net_by_level", "cell_by_level"}
    return {k: (v if k in KEEP_CPU or not torch.is_tensor(v) else v.to(device))
            for k, v in g.items()}


def load_pt(path, gate_vocab):
    g = torch.load(path, map_location="cpu", weights_only=False)
    n_types = len(gate_vocab); unk = gate_vocab.get("UNKNOWN", 0)
    gts = g.get("cell_gtype_str", [])
    if gts:
        oh = torch.zeros(len(gts), n_types)
        for i, gt in enumerate(gts):
            oh[i, gate_vocab.get(gt, unk)] = 1.0
        g["gate_onehot"] = oh
    if "gate_onehot" in g and g["gate_onehot"].shape[1] != n_types:
        old = g["gate_onehot"]
        new = torch.zeros(old.shape[0], n_types); new[:, :old.shape[1]] = old
        g["gate_onehot"] = new
    if "lut_delay" in g and g["lut_delay"].dim() == 3:      # old 1-corner files
        print(f"  [load_pt] WARNING {os.path.basename(path)}: single-corner "
              f"LUTs expanded to 8 tables (re-run preprocess for real corners)")
        g["lut_delay"] = g["lut_delay"].unsqueeze(1).expand(-1, 8, -1, -1).contiguous()
        g["lut_slew"] = g["lut_slew"].unsqueeze(1).expand(-1, 8, -1, -1).contiguous()
    return g


def export_predictions(dirpath, name, g, out):
    """Per-design CSV in the exact format conformal.py reads.
    slack_pred = RAT - AT_pred = slack_true + AT_true - AT_pred  (RAT is a
    fixed label, so no RAT tensor is needed)."""
    import csv
    os.makedirs(dirpath, exist_ok=True)
    N = g["x"].shape[0]
    ep = torch.zeros(N, dtype=torch.bool)
    ep[g["endpoints"].cpu()] = True
    at_t = g["AT_true"].cpu(); at_p = out["AT"].cpu()
    has_slack = "slack_true" in g
    if has_slack:
        sl_t = g["slack_true"].cpu()
        sl_p = sl_t + at_t - at_p
    with open(os.path.join(dirpath, f"predictions_{name}.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        hdr = ["node", "is_endpoint", "at_true_0", "at_pred_0"]
        if has_slack: hdr += ["slack_true", "slack_pred"]
        w.writerow(hdr)
        for i in range(N):
            row = [i, int(ep[i]), float(at_t[i]), float(at_p[i])]
            if has_slack: row += [float(sl_t[i]), float(sl_p[i])]
            w.writerow(row)
    return has_slack


# --------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed", required=True)
    ap.add_argument("--features", default="no_spef16", choices=list(FEATURE_SETS))
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hidden_dim", type=int, default=32)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--train_designs", default="",
                    help="comma list; default = paper 14 (those present)")
    ap.add_argument("--test_designs", default="",
                    help="comma list; default = paper 7 (those present)")
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()
    set_seed(args.seed)

    with open(os.path.join(args.preprocessed, "meta.json")) as f:
        meta = json.load(f)
    gate_vocab = meta["gate_vocab"]; n_gates = len(gate_vocab)

    pt_files = {os.path.splitext(fn)[0].replace(".graph", ""):
                os.path.join(args.preprocessed, fn)
                for fn in os.listdir(args.preprocessed) if fn.endswith(".pt")}
    train_names = ([s for s in args.train_designs.split(",") if s] or
                   [d for d in PAPER_TRAIN if d in pt_files])
    test_names = ([s for s in args.test_designs.split(",") if s] or
                  [d for d in PAPER_TEST if d in pt_files])
    print(f"[split] train={train_names}\n[split] test ={test_names}")
    missing = [d for d in PAPER_TRAIN + PAPER_TEST if d not in pt_files]
    if missing: print(f"[split] not preprocessed yet: {missing}")

    def load_named(names):
        out = {}
        for n in names:
            g = load_pt(pt_files[n], gate_vocab)
            g, nfeat = select_features(g, args.features)
            out[n] = g
        return out, nfeat

    train_graphs, n_feat = load_named(train_names)
    test_graphs, _ = load_named(test_names)
    print(f"[features] set={args.features} -> {n_feat} node features")

    cfg = ModelConfig(hidden_dim=args.hidden_dim)
    model = TimingGNN(cfg, n_gates=n_gates, in_dim=n_feat).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.describe()} | params={n_params:,} | device={DEVICE}")

    run_name = args.run_name or f"v4_{args.features}_s{args.seed}"
    logger = ResultLogger(run_name=run_name, cfg=cfg, extra_info={
        "features": args.features, "n_features": n_feat, "seed": args.seed,
        "loss_weights": "AT=1.0 slew=1.0 net=1.0 cell=1.0 (log-space)",
        "teacher_forcing": True, "model": "model_configurable_v4"})

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.6, patience=max(1, args.epochs // 8), min_lr=1e-4)
    mse = nn.MSELoss()

    @torch.no_grad()
    def pooled_test_r2():
        model.eval()
        ps, ts = [], []
        for n, gc in test_graphs.items():
            g = move_to_device(gc, DEVICE)
            out = model(g)                      # full propagation
            ep = g["endpoints"]
            ps.append(out["AT"][ep].cpu()); ts.append(g["AT_true"][ep].cpu())
        return r2_both(torch.cat(ps), torch.cat(ts))

    best_r2, best_state, best_ep, bad = -1e9, None, -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = dict(loss=0, AT=0, slew=0, net=0, cell=0)
        for nme, gc in train_graphs.items():
            g = move_to_device(gc, DEVICE)
            out = model(g, groundtruth=True)            # teacher forcing
            ep = g["endpoints"]
            L_AT = mse(out["AT"][ep], g["AT_true"][ep])
            sm = g["slew_real_mask"]
            L_slew = (mse(out["SLEW_log"][sm], slew_to_log(g["SLEW_true"][sm]))
                      if sm.sum() > 0 else torch.tensor(0.0, device=DEVICE))
            nm = g["net_trusted_mask"]
            L_net = (mse(out["net_delay_log"][nm], nd_to_log(g["nd_true"][nm]))
                     if nm.sum() > 0 else torch.tensor(0.0, device=DEVICE))
            cm = g["cell_lut_mask"]
            L_cell = (mse(out["cell_delay"][cm], g["cd_true"][cm])
                      if cm.sum() > 0 else torch.tensor(0.0, device=DEVICE))
            loss = L_AT + L_slew + L_net + L_cell        # flat weights
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            for k, v in zip(tl, [loss, L_AT, L_slew, L_net, L_cell]):
                tl[k] += v.item()
        n_tr = max(len(train_graphs), 1)
        sched.step(tl["loss"] / n_tr)

        if epoch == 1 or epoch % args.log_every == 0:
            r2c, r2p = pooled_test_r2()
            logger.log_epoch(epoch, tl["loss"] / n_tr,
                             loss_AT=round(tl["AT"] / n_tr, 5),
                             loss_slew=round(tl["slew"] / n_tr, 5),
                             loss_net=round(tl["net"] / n_tr, 5),
                             loss_cell=round(tl["cell"] / n_tr, 5),
                             test_r2_AT_pooled=round(r2c, 5),
                             test_r2_AT_paperconv=round(r2p, 5))
            print(f"  ep {epoch:4d} | loss {tl['loss']/n_tr:8.4f} "
                  f"[AT {tl['AT']/n_tr:.4f} slew {tl['slew']/n_tr:.4f} "
                  f"net {tl['net']/n_tr:.4f} cell {tl['cell']/n_tr:.4f}] "
                  f"| test r2 {r2c:.4f} (paper-conv {r2p:.4f})")
            if r2c > best_r2:
                best_r2, best_ep, bad = r2c, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= args.patience // args.log_every + 1:
                    print(f"  early stop at {epoch} (best ep {best_ep})"); break

    if best_state: model.load_state_dict(best_state)

    # ---------------- final eval: per-design, both conventions, CSV export
    model.eval(); metrics = {"epochs_run": epoch, "best_epoch": best_ep}
    csv_dir = os.path.join(logger.dir, "csv")
    with torch.no_grad():
        for nme, gc in test_graphs.items():
            g = move_to_device(gc, DEVICE)
            t0 = time.time(); out = model(g); dt = time.time() - t0
            ep = g["endpoints"]
            r2c, r2p = r2_both(out["AT"][ep].cpu(), g["AT_true"][ep].cpu())
            metrics[f"test_r2_AT_{nme}"] = round(r2c, 5)
            metrics[f"test_r2_AT_paperconv_{nme}"] = round(r2p, 5)
            has_slack = export_predictions(csv_dir, nme, g, out)
            if has_slack:
                sl_t = g["slack_true"][ep].cpu()
                sl_p = (g["slack_true"] + g["AT_true"] - out["AT"])[ep].cpu()
                metrics[f"test_r2_slack_{nme}"] = round(r2_both(sl_p, sl_t)[0], 5)
            print(f"  [final] {nme:14} r2 {r2c:.4f} paper-conv {r2p:.4f} "
                  f"infer {dt:.3f}s slack_csv={has_slack}")
    r2c, r2p = pooled_test_r2()
    metrics["test_r2_AT_pooled"] = round(r2c, 5)
    metrics["test_r2_AT_pooled_paperconv"] = round(r2p, 5)

    logger.save_final(metrics_summary=metrics,
                      datasets={"train": train_names, "test": test_names,
                                "features": args.features},
                      model=model)
    print(f"\n[next] conformal:  python conformal.py --csv_dir {csv_dir} "
          f"--cal_designs usbf_device,y_huff "
          f"--test_designs xtea,spm,synth_ram,aes192,jpeg_encoder "
          f"--target slack --alpha 0.1")


if __name__ == "__main__":
    main()
