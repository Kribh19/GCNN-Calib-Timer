"""diag_v2.py -- swap predicted wire quantities for true ones, one at a time."""
import glob
import os
import sys

import torch

from build_v2 import load
from liberty_v2 import load_liberty
from sta_v2 import prepare, propagate, r2_score
from train_v2 import OUT_DIR, WireNet, features, metrics

lib = load_liberty()
dev = torch.device("cuda")
run = sorted(glob.glob(os.path.join(OUT_DIR, "*")))[-1] if not sys.argv[1].startswith("run=") else sys.argv[1][4:]
ck = torch.load(os.path.join(run, "wirenet.pt"), weights_only=False)
model = WireNet(ck["fn_dim"], ck["fe_dim"], ck["args"]["hidden"]).to(dev)
model.load_state_dict(ck["state_dict"])
model.eval()
print("run:", run)
for dn in [a for a in sys.argv[1:] if not a.startswith("run=")]:
    p = prepare(load(dn, lib), lib, dev)
    with torch.no_grad():
        wc_ff, nd_ps, kappa = model(p, *features(p))
        wc_p, nd_p = wc_ff / 1000.0, nd_ps / 1000.0
        wc_t = torch.nan_to_num(p["wirecap"], nan=0.0)
        nd_t = p["nd"]
        k = model.slew_k
        print(f"== {dn}")
        for tag, kw in [
            ("all predicted", dict(wirecap=wc_p, net_delay=nd_p, kappa=kappa)),
            ("true wire delay", dict(wirecap=wc_p, net_delay=nd_t, kappa=kappa)),
            ("true wire cap", dict(wirecap=wc_t, net_delay=nd_p, kappa=kappa)),
            ("true cap + true delay", dict(wirecap=wc_t, net_delay=nd_t, kappa=kappa)),
            ("true cap+delay, no kappa", dict(wirecap=wc_t, net_delay=nd_t)),
        ]:
            AT, SL, _, v = propagate(p, "lut", slew_k=k, **kw)
            m = metrics(p, AT, SL, v)
            print(f"   {tag:<26} AT R2 {m['AT_R2']:.4f} [LR {m['AT_R2_LR']:.4f}]  MAE {m['AT_MAE']:.3f}  slew R2 {m['SLEW_R2']:.4f}  "
                  f"endpoint AT {m.get('endpoint_AT_R2', float('nan')):.4f}  slack {m.get('setup_slack_R2', float('nan')):.4f}")
        # wire-delay prediction quality by size
        t, q = nd_t[:, 2] * 1000, nd_ps[:, 2]
        for lo, hi in [(0, 10), (10, 100), (100, 1000), (1000, 1e9)]:
            mk = (t >= lo) & (t < hi)
            if mk.any():
                print(f"   wire delay {lo:>5}-{hi:<6.0f} ps: n={int(mk.sum()):7d}  true mean {float(t[mk].mean()):9.2f}  "
                      f"pred mean {float(q[mk].mean()):9.2f}  MAE {float((t[mk] - q[mk]).abs().mean()):9.2f}")
        s = p["net_src"]
        fan = p["fanout"][s]
        big = t > 300
        if big.any():
            drv = torch.unique(s[big])
            print(f"   nets with any wire delay > 300 ps: {drv.numel()}  fanouts: {sorted(set(int(x) for x in p['fanout'][drv].tolist()))[-8:]}  "
                  f"kappa on their drivers (pred): {[round(float(x), 2) for x in kappa[torch.isin(p['ca_dst'], drv)][:6]]}")
