"""
train_v2.py -- CalibTimer v2: learned wire model + differentiable STA.

Before routing, the only unknowns of a timing run are routing parasitics.  The learned
model predicts, from placement only:
    wire capacitance of each net                  (label: SPEF total cap)
    wire delay of each driver->sink edge          (label: SDF INTERCONNECT)
    effective-capacitance factor kappa per arc    (label: SDF cell delay through the Liberty table)
Arrival times and slews then follow from Liberty tables + min/max propagation (sta_v2.py).

Split: the DAC'22 TimingGCN split (14 train / 7 test).  Model selection uses held-out
TRAIN designs, never the test designs.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from build_v2 import load
from liberty_v2 import load_liberty
from sta_v2 import arc_delays, prepare, propagate, r2_score, to_dev

TEST = ["jpeg_encoder", "usbf_device", "aes192", "xtea", "spm", "y_huff", "synth_ram"]
TRAIN = ["blabla", "usb_cdc_core", "BM64", "salsa20", "aes128", "wbqspiflash", "cic_decimator",
         "aes256", "des", "aes_cipher", "picorv32a", "zipdiv", "genericfir", "usb"]
CUSTOM = ["uni_counter4b", "cla8b", "seq_detector", "priority_enc", "gcd"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def features(p):
    """Per-driver net features [N,Fn], per-edge features [E,Fe], sink pin cap per edge [E] (fF)."""
    N, dev = p["n"], p["xy"].device
    s, d = p["net_src"], p["net_dst"]
    xy = p["xy"]
    dx = (xy[d, 0] - xy[s, 0]).abs()
    dy = (xy[d, 1] - xy[s, 1]).abs()
    man = dx + dy
    z = torch.zeros(N, device=dev)
    fan = p["fanout"]
    star = z.index_add(0, s, man)
    maxd = z.scatter_reduce(0, s, man, "amax", include_self=True)
    idx2 = s.unsqueeze(1).expand(-1, 2)
    lo = xy.clone().scatter_reduce(0, idx2, xy[d], "amin", include_self=True)
    hi = xy.clone().scatter_reduce(0, idx2, xy[d], "amax", include_self=True)
    bw, bh = hi[:, 0] - lo[:, 0], hi[:, 1] - lo[:, 1]
    pl = p["pin_load"][:, 2] * 1000.0                       # fF
    port_sinks = z.index_add(0, s, p["is_port"][d].float())
    sx = z.index_add(0, s, dx)
    sy = z.index_add(0, s, dy)
    lg = torch.log1p
    lt, mw, mp = p["n_tree"][:, 0], p["n_tree"][:, 1], p["n_tree"][:, 2]
    ed = p["edge_dist"]                                     # um to left/right/bottom/top die boundary
    emin = ed.min(1).values
    sink_emin = torch.full((N,), 1e4, device=dev).scatter_reduce(0, s, emin[d], "amin", include_self=True)
    fn = torch.stack([
        lg(fan), fan / 10.0, lg(bw + bh), (bw + bh) / 100.0, lg(bw), lg(bh), lg(star), star / 100.0,
        lg(maxd), lg(sx), lg(sy), lg(pl), pl / 10.0, p["is_port"].float(), port_sinks.clamp(max=1),
        lg(star / fan.clamp(min=1)),
        lg(lt), lt / 100.0, lg(mw) / 2, lg(mp) / 2, lg(lt / fan.clamp(min=1)),
        lg(ed[:, 0]), lg(ed[:, 1]), lg(ed[:, 2]), lg(ed[:, 3]), lg(emin), (emin < 20).float(),
        lg(sink_emin.clamp(max=1e4)), (sink_emin < 20).float(),
    ], 1)
    # local placement / wiring density on a 20 um grid (coupling cap and detours grow with it)
    B = 20.0
    W, H = float(xy[:, 0].max()) + B, float(xy[:, 1].max()) + B
    nx, ny = int(W // B) + 1, int(H // B) + 1
    cell = (xy[:, 0] / B).long().clamp(0, nx - 1) * ny + (xy[:, 1] / B).long().clamp(0, ny - 1)
    share = lt / (fan + 1.0)
    pin_map = torch.zeros(nx * ny, device=dev).index_add(0, cell, torch.ones(N, device=dev))
    wire_map = torch.zeros(nx * ny, device=dev).index_add(0, cell[s], share[s]).index_add(0, cell[d], share[s])
    sm = lambda m: F.avg_pool2d(m.view(1, 1, nx, ny), 3, 1, 1, count_include_pad=False).view(-1) / (B * B)
    pin_map, wire_map = sm(pin_map), sm(wire_map)
    f1 = fan.clamp(min=1)
    dens = torch.stack([
        lg(pin_map[cell] * 100), lg(wire_map[cell] * 10),
        lg(z.index_add(0, s, pin_map[cell[d]]) / f1 * 100), lg(z.index_add(0, s, wire_map[cell[d]]) / f1 * 10),
        torch.full((N,), float(N / (W * H)) * 10, device=dev),
        torch.full((N,), float(lt.sum() / (W * H)), device=dev),
    ], 1)
    fn = torch.cat([fn, dens], 1)
    cap_d = p["pincap"][d, 2] * 1000.0
    lp, ew, ep, dep = p["e_tree"].unbind(1)
    fe = torch.cat([
        torch.stack([lg(dx), lg(dy), lg(man), man / 100.0, man / (maxd[s] + 1.0),
                     cap_d / 5.0, p["is_port"][d].float(),
                     (xy[d, 0] - lo[s, 0]) / (bw[s] + 1.0), (xy[d, 1] - lo[s, 1]) / (bh[s] + 1.0),
                     (xy[s, 0] - lo[s, 0]) / (bw[s] + 1.0), (xy[s, 1] - lo[s, 1]) / (bh[s] + 1.0),
                     lg(lp), lg(ew) / 2, lg(ep) / 2, dep / 10.0, lp / (man + 1.0) / 3.0,
                     lg(ed[d, 0]), lg(ed[d, 1]), lg(ed[d, 2]), lg(ed[d, 3]), lg(emin[d]), (emin[d] < 20).float()], 1),
        fn[s],
    ], 1)
    return fn, fe, cap_d


def mlp(i, h, o, n=3):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class WireNet(nn.Module):
    def __init__(self, fn_dim, fe_dim, h=96):
        super().__init__()
        self.net = mlp(fn_dim, h, 3)          # wire-cap correction, wire-cap offset, root resistance
        self.edge = mlp(fe_dim + 3, h, 2)     # Elmore correction, sink-side resistance
        self.ceff = mlp(fn_dim + 6 + 4, h // 2, 1, n=2)
        self.slew_k = nn.Parameter(torch.full((4,), 2.0))
        self.log_c = nn.Parameter(torch.tensor(0.17).log())       # wire cap per um of tree, fF/um
        self.log_rc = nn.Parameter(torch.tensor(1.056e-4).log())  # r*c, ps/um^2
        self.log_r = nn.Parameter(torch.tensor(2.467e-3).log())   # r, kOhm/um
        self.ch_scale = nn.Parameter(torch.zeros(4))
        self.log_rport = nn.Parameter(torch.tensor([1.0, 1.0, 2.1, 2.1]).log())
        self.k_gamma = nn.Parameter(torch.ones(4))                # shielding sharpness per channel
        self.k_logx0 = nn.Parameter(torch.ones(4))                # log(R_wire/R_drive) at which kappa = 0.5

    def forward(self, p, fn, fe, cap_d):
        s = p["net_src"]
        o3 = self.net(fn)
        lt, mw, mp = p["n_tree"].unbind(1)
        # wire cap = (cap per um) x (tree length) x learned correction + small offset   [fF]
        wc_ff = self.log_c.exp() * lt * torch.exp(o3[:, 0].clamp(-2.5, 2.5)) + F.softplus(o3[:, 1] - 2.0)
        # resistance at the root of the net, kOhm: a constant for input ports (2.0-2.2 kOhm in every
        # design of this dataset), a learned non-negative value for cell drivers
        isp = p["is_port"].float()
        r_root = (1 - isp) * F.softplus(o3[:, 2] - 4.0)
        ctot = p["pin_load"][:, 2] * 1000.0 + wc_ff                       # total net cap, fF
        lw, lc = torch.log1p(wc_ff), torch.log1p(ctot)
        lr = torch.log1p(r_root * 100.0)
        o2 = self.edge(torch.cat([fe, lw[s, None], lc[s, None], lr[s, None]], 1))
        ew, ep = p["e_tree"][:, 1], p["e_tree"][:, 2]
        elmore = self.log_rc.exp() * ew + self.log_r.exp() * ep           # tree Elmore delay, ps
        r_sink = F.softplus(o2[:, 1] - 4.0)
        nd = r_root[s] * ctot[s] + elmore * torch.exp(o2[:, 0].clamp(-3.0, 3.0)) + r_sink * (cap_d + 1.0)
        nd_ps = nd[:, None] * torch.exp(self.ch_scale)[None, :]           # wire delay per channel, ps
        # input ports: the constraint's driving cell adds (drive resistance x net cap); the resistance
        # is one constant per channel in this dataset (~1.0 kOhm early corner, ~2.1 kOhm late corner)
        nd_ps = nd_ps + (isp[s] * ctot[s])[:, None] * self.log_rport.exp()[None, :]
        # effective capacitance: kappa = 1 / (1 + (R_wire / (x0 * R_drive))^gamma), small learned correction
        o, ch = p["ca_dst"], p["ca_out"]
        r_wire = r_root + (self.log_rc.exp() * mw + self.log_r.exp() * mp) / ctot.clamp(min=1.0)   # kOhm
        rd = p["ca_rd"].clamp(min=1e-3)
        lx = torch.log(r_wire[o].clamp(min=1e-4) / rd)
        cin = torch.cat([fn[o], lw[o, None], lc[o, None], torch.log(rd)[:, None], lx[:, None],
                         torch.log1p(r_wire[o] * ctot[o])[:, None], torch.log1p(rd * ctot[o])[:, None],
                         F.one_hot(ch, 4).float()], 1)
        corr = torch.tanh(self.ceff(cin).squeeze(1))
        kappa = torch.sigmoid(-self.k_gamma[ch].abs() * (lx - self.k_logx0[ch]) + corr)
        return wc_ff, nd_ps, kappa


def local_loss(p, out, slew_k):
    """Teacher-forced losses: no propagation, every label is local."""
    wc_ff, nd_ps, kappa = out
    wc_t = p["wirecap"] * 1000.0
    m = torch.isfinite(wc_t)
    l_wc = F.smooth_l1_loss(wc_ff[m], wc_t[m]) + F.mse_loss(torch.log1p(wc_ff[m]), torch.log1p(wc_t[m]))
    nd_t = (p["nd"] * 1000.0).clamp(min=0)
    l_nd = F.mse_loss(torch.log1p(nd_ps), torch.log1p(nd_t)) + F.smooth_l1_loss(nd_ps / 100.0, nd_t / 100.0)
    big = nd_t > 50.0                      # the few long wires carry most of the timing impact
    if big.any():
        l_nd = l_nd + F.mse_loss(torch.log1p(nd_ps[big]), torch.log1p(nd_t[big]))
    dl = arc_delays(p, wc_ff / 1000.0, kappa)
    cd = p["cd"][p["ca_arc"], p["ca_out"]]
    ok = torch.isfinite(cd)
    l_cd = F.mse_loss(dl[ok], cd[ok]) * 100.0
    heavy = ok & ((p["pin_load"][p["ca_dst"], p["ca_out"]] + torch.nan_to_num(p["wirecap"], nan=0.0)[p["ca_dst"]]) > 0.1)
    if heavy.any():
        l_cd = l_cd + F.mse_loss(dl[heavy], cd[heavy]) * 10.0
    # slew step along the wire, with the TRUE wire delay: trains only slew_k
    s, d = p["net_src"], p["net_dst"]
    s0, s1 = p["SLEW"][s], p["SLEW"][d]
    okk = torch.isfinite(s0) & torch.isfinite(s1)
    pred = torch.sqrt(torch.nan_to_num(s0, nan=0.0) ** 2 + (slew_k * p["nd"]) ** 2 + 1e-12)
    l_sl = F.mse_loss(pred[okk], s1[okk]) * 100.0
    return l_wc + l_nd + l_cd + l_sl, (float(l_wc.detach()), float(l_nd.detach()), float(l_cd.detach()))


def run_sta(p, model, feats=None):
    fn, fe, cap_d = feats or features(p)
    wc_ff, nd_ps, kappa = model(p, fn, fe, cap_d)
    AT, SL, _, v = propagate(p, "lut", wirecap=wc_ff / 1000.0, net_delay=nd_ps / 1000.0,
                             slew_k=model.slew_k, kappa=kappa)
    return AT, SL, v, (wc_ff, nd_ps, kappa)


def metrics(p, AT, SL, v, wc_ff=None):
    lab = torch.isfinite(p["AT"]) & v & ~p["is_src"].unsqueeze(1)
    out = {"AT_R2": r2_score(AT[lab], p["AT"][lab]),
           "AT_MAE": float((AT[lab] - p["AT"][lab]).abs().mean())}
    for c, nm in enumerate(["ER", "EF", "LR", "LF"]):
        out["AT_R2_" + nm] = r2_score(AT[:, c][lab[:, c]], p["AT"][:, c][lab[:, c]])
    sl = lab & torch.isfinite(p["SLEW"])
    out["SLEW_R2"] = r2_score(SL[sl], p["SLEW"][sl])
    if wc_ff is not None:
        m = torch.isfinite(p["wirecap"])
        out["wirecap_R2"] = r2_score(wc_ff[m] / 1000.0, p["wirecap"][m])
    # DAC'22 code convention: all pins x 4 channels, r2_score(pred, truth) argument order
    ok = torch.isfinite(p["AT"])
    t_all = torch.nan_to_num(p["AT"], nan=0.0).reshape(-1)
    q_all = torch.where(ok, AT, torch.zeros_like(AT)).reshape(-1)
    out["AT_R2_paper_conv"] = r2_score(t_all, q_all)
    # endpoints = pins that drive nothing; setup slack lives on the late channels
    drives = torch.zeros(p["n"], dtype=torch.bool, device=AT.device)
    drives[p["net_src"]] = True
    drives[p["ca_src"]] = True
    ep = (~drives).unsqueeze(1) & lab & torch.isfinite(p["RAT"])
    ep[:, :2] = False
    if ep.any():
        out["endpoint_AT_R2"] = r2_score(AT[ep], p["AT"][ep])
        out["endpoint_AT_MAE"] = float((AT[ep] - p["AT"][ep]).abs().mean())
        out["setup_slack_R2"] = r2_score(p["RAT"][ep] - AT[ep], p["RAT"][ep] - p["AT"][ep])
        out["n_endpoints"] = int(ep[:, 2].sum())
    return out


def evaluate(p, model=None):
    with torch.no_grad():
        if model is None:
            AT, SL, _, v = propagate(p, "lut")
            return metrics(p, AT, SL, v)
        AT, SL, v, (wc_ff, _, _) = run_sta(p, model)
        return metrics(p, AT, SL, v, wc_ff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--e2e_epochs", type=int, default=0)
    ap.add_argument("--e2e_max_pins", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=str, default="picorv32a,usb_cdc_core")
    ap.add_argument("--tag", type=str, default="v2")
    ap.add_argument("--custom", action="store_true")
    ap.add_argument("--holdout", type=str, default="", help="train designs left out of fitting (for conformal calibration)")
    ap.add_argument("--swap", action="store_true", help="robustness check: fit on the 7 test designs, test on the 14 train designs")
    args = ap.parse_args()
    global TRAIN, TEST
    if args.swap:
        TRAIN, TEST = TEST, TRAIN
        if args.val == "picorv32a,usb_cdc_core":
            args.val = "xtea,usbf_device"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lib = load_liberty()
    VAL = [v for v in args.val.split(",") if v]
    names = TRAIN + TEST + (CUSTOM if args.custom else [])
    P = {dn: prepare(load(dn, lib), lib, torch.device("cpu")) for dn in names}
    HOLD = [v for v in args.holdout.split(",") if v]
    fit = [d for d in TRAIN if d not in VAL and d not in HOLD]
    G = {dn: to_dev(P[dn], dev) for dn in fit + VAL}
    FE = {dn: features(G[dn]) for dn in G}
    fn_dim, fe_dim = FE[fit[0]][0].shape[1], FE[fit[0]][1].shape[1]
    model = WireNet(fn_dim, fe_dim, args.hidden).to(dev)
    print(f"fit={len(fit)} val={VAL} params={sum(q.numel() for q in model.parameters()):,}")

    def validate():
        model.eval()
        with torch.no_grad():
            r = [metrics(G[dn], *run_sta(G[dn], model, FE[dn])[:3])["AT_R2"] for dn in VAL]
        return sum(r) / len(r), r

    # ---- stage 1: teacher-forced local fitting ----
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state, hist = -1e9, None, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = fit[:]
        random.shuffle(order)
        parts = [0.0, 0.0, 0.0]
        for dn in order:
            loss, pr = local_loss(G[dn], model(G[dn], *FE[dn]), model.slew_k)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            parts = [a + b / len(order) for a, b in zip(parts, pr)]
        sched.step()
        if ep % 20 == 0 or ep == args.epochs:
            vm, vr = validate()
            hist.append({"epoch": ep, "l_wc": parts[0], "l_nd": parts[1], "l_cd": parts[2], "val_AT_R2": vm})
            print(f"  ep {ep:3d}  wc {parts[0]:.4f}  nd {parts[1]:.4f}  cd {parts[2]:.4f}  "
                  f"val AT R2 {vm:.5f} ({', '.join(f'{x:.4f}' for x in vr)})  [{time.time() - t0:.0f}s]")
            if vm > best:
                best, best_state = vm, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # ---- stage 2: end-to-end fine-tuning through the timing propagation ----
    if args.e2e_epochs:
        e2e = [d for d in fit if G[d]["n"] <= args.e2e_max_pins]
        print(f"end-to-end fine-tuning on {len(e2e)} designs: {e2e}")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
        for ep in range(1, args.e2e_epochs + 1):
            model.train()
            order = fit[:]
            random.shuffle(order)
            tot = 0.0
            for dn in order:
                p = G[dn]
                if dn in e2e:
                    AT, SL, v, out = run_sta(p, model, FE[dn])
                    lab = torch.isfinite(p["AT"]) & v & ~p["is_src"].unsqueeze(1)
                    sl = lab & torch.isfinite(p["SLEW"])
                    l_at = F.smooth_l1_loss(AT[lab], p["AT"][lab], beta=0.1)
                    l_sl = F.smooth_l1_loss(SL[sl], p["SLEW"][sl], beta=0.1)
                    loss = 3.0 * l_at + l_sl + 0.3 * local_loss(p, out, model.slew_k)[0]
                else:
                    loss = 0.3 * local_loss(p, model(p, *FE[dn]), model.slew_k)[0]
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                tot += float(loss.detach())
            vm, vr = validate()
            hist.append({"epoch": args.epochs + ep, "e2e_loss": tot / len(order), "val_AT_R2": vm})
            print(f"  e2e ep {ep:2d}  loss {tot / len(order):.4f}  val AT R2 {vm:.5f} "
                  f"({', '.join(f'{x:.4f}' for x in vr)})  [{time.time() - t0:.0f}s]")
            if vm > best:
                best, best_state = vm, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

    # ---- final evaluation ----
    del G, FE
    torch.cuda.empty_cache()
    model.eval()
    res = {}
    hdr = (f"{'design':<15}{'split':<7}{'pins':>8} | {'no-wire':>8} | {'AT R2':>7} {'ER':>7} {'EF':>7} {'LR':>7} {'LF':>7} | "
           f"{'MAE ns':>7} {'slew':>7} {'wcap':>7} {'ep AT':>7} {'slack':>7} {'paperR2':>7} {'sec':>6}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for dn in P:
        split = "test" if dn in TEST else "val" if dn in VAL else "custom" if dn in CUSTOM else "holdout" if dn in HOLD else "train"
        p = to_dev(P[dn], dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        r = evaluate(p, model)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        r["runtime_s"] = time.time() - t1
        r["nowire_AT_R2"] = evaluate(p)["AT_R2"]
        r["split"], r["pins"] = split, p["n"]
        res[dn] = r
        print(f"{dn:<15}{split:<7}{p['n']:>8} | {r['nowire_AT_R2']:>8.4f} | {r['AT_R2']:>7.4f} {r['AT_R2_ER']:>7.4f} "
              f"{r['AT_R2_EF']:>7.4f} {r['AT_R2_LR']:>7.4f} {r['AT_R2_LF']:>7.4f} | {r['AT_MAE']:>7.4f} {r['SLEW_R2']:>7.4f} "
              f"{r['wirecap_R2']:>7.4f} {r.get('endpoint_AT_R2', float('nan')):>7.4f} {r.get('setup_slack_R2', float('nan')):>7.4f} "
              f"{r['AT_R2_paper_conv']:>7.4f} {r['runtime_s']:>6.2f}")
        del p
        torch.cuda.empty_cache()
    for split in ("train", "val", "test", "custom"):
        rows = [r for r in res.values() if r["split"] == split]
        if rows:
            n = len(rows)
            print(f"mean {split:<6}: AT R2 {sum(r['AT_R2'] for r in rows) / n:.4f}  min {min(r['AT_R2'] for r in rows):.4f}  "
                  f"no-wire {sum(r['nowire_AT_R2'] for r in rows) / n:.4f}  "
                  f"endpoint AT {sum(r.get('endpoint_AT_R2', 0) for r in rows) / n:.4f}  "
                  f"slack {sum(r.get('setup_slack_R2', 0) for r in rows) / n:.4f}")
    print("slew_k =", [round(float(x), 3) for x in model.slew_k])
    os.makedirs(OUT_DIR, exist_ok=True)
    run = os.path.join(OUT_DIR, time.strftime("%Y%m%d_%H%M%S_") + args.tag)
    os.makedirs(run)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "val": VAL, "fit": fit, "holdout": HOLD,
                "fn_dim": fn_dim, "fe_dim": fe_dim}, os.path.join(run, "wirenet.pt"))
    with open(os.path.join(run, "results.json"), "w") as f:
        json.dump({"args": vars(args), "history": hist, "results": res}, f, indent=1)
    print("saved", run)


if __name__ == "__main__":
    main()
