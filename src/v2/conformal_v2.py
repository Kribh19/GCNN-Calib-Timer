"""
conformal_v2.py -- calibrated intervals for endpoint arrival time / setup slack (CalibTimer v2).

Calibration data: errors of FOLD models on training designs they did not see (runs tagged
cf0..cf5, each trained with --holdout on two designs).  Nothing from the test designs or the
custom designs is used to size the intervals.

Score:  |true late arrival - predicted| / sigma, with sigma one of
        "abs"   1 ns
        "rel"   0.1 + predicted arrival
        "wire"  0.01 + how far this arrival moves when every wire is 10 % heavier
                (the model's error is mostly a per-design wire-capacitance offset)
        The scale giving the narrowest intervals on the calibration designs is used.

Two interval sizes are reported:
  pin-level     q = conformal quantile of the pooled scores at 1-alpha
                (endpoints treated as exchangeable -- they are not independent within a design)
  design-level  per calibration design take its (1-alpha) score quantile, then the conformal
                quantile ACROSS designs at 1-delta.  Reading: for a new design drawn like the
                calibration designs, with probability >= 1-delta at least a (1-alpha) share of
                its endpoints fall inside the interval.

Interval:  arrival in [pred + (c - q) sigma, pred + (c + q) sigma],  slack = RAT - arrival.
"""
import glob
import json
import math
import os
import sys

import torch

from build_v2 import load
from liberty_v2 import load_liberty
from sta_v2 import prepare, propagate
from train_v2 import CUSTOM, OUT_DIR, TEST, WireNet, features

ALPHA, DELTA, EPS = 0.10, 0.20, 0.10


def latest(tag):
    r = sorted(d for d in glob.glob(os.path.join(OUT_DIR, "*_" + tag)) if os.path.isdir(d))
    if not r:
        raise FileNotFoundError(f"no run tagged {tag} in {OUT_DIR}")
    return r[-1]


def load_model(run, dev):
    ck = torch.load(os.path.join(run, "wirenet.pt"), weights_only=False, map_location=dev)
    m = WireNet(ck["fn_dim"], ck["fe_dim"], ck["args"]["hidden"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def endpoints(dn, model, lib, dev):
    """Late-channel predictions at timing endpoints: pred, true, RAT (each [n]) ."""
    p = prepare(load(dn, lib), lib, dev)
    with torch.no_grad():
        fn, fe, cap_d = features(p)
        wc_ff, nd_ps, kappa = model(p, fn, fe, cap_d)
        AT, _, _, v = propagate(p, "lut", wirecap=wc_ff / 1000.0, net_delay=nd_ps / 1000.0, slew_k=model.slew_k, kappa=kappa)
        # sensitivity: how far each arrival moves if every wire is EPS heavier (cap and delay)
        AT2, _, _, _ = propagate(p, "lut", wirecap=wc_ff * (1 + EPS) / 1000.0, net_delay=nd_ps * (1 + EPS) / 1000.0,
                                 slew_k=model.slew_k, kappa=kappa)
        sens = (AT2 - AT).abs()
    drives = torch.zeros(p["n"], dtype=torch.bool, device=dev)
    drives[p["net_src"]] = True
    drives[p["ca_src"]] = True
    ep = (~drives & ~p["is_src"]).unsqueeze(1) & torch.isfinite(p["AT"]) & v
    ep[:, :2] = False
    return AT[ep].cpu(), p["AT"][ep].cpu(), p["RAT"][ep].cpu(), sens[ep].cpu()


def sigma(pred, sens, mode):
    if mode == "abs":
        return torch.ones_like(pred)
    if mode == "rel":
        return 0.1 + pred.clamp(min=0)
    return 0.01 + sens          # "wire": ns of movement per EPS of extra wire


def cq(scores, level):
    s = torch.sort(scores).values
    k = min(len(s), math.ceil((len(s) + 1) * level))
    return float(s[k - 1])


def main():
    final_tag = sys.argv[1] if len(sys.argv) > 1 else "final"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lib = load_liberty()

    # ---- calibration residuals from fold models on their held-out designs ----
    cal = {}
    for k in range(6):
        run = latest(f"cf{k}")
        model, ck = load_model(run, dev)
        for dn in ck["holdout"]:
            q_, t_, _, s_ = endpoints(dn, model, lib, dev)
            cal[dn] = (q_, t_, s_)
    print(f"calibration designs ({len(cal)}): {sorted(cal)}")
    print(f"calibration endpoint samples: {sum(len(v[0]) for v in cal.values())}")

    best = None
    for mode in ("abs", "rel", "wire"):
        r = {d: (t - q) / sigma(q, sn, mode) for d, (q, t, sn) in cal.items()}
        pooled = torch.cat(list(r.values()))
        c = 0.0     # no centring: the sign of the per-design offset differs between designs
        q_pin = cq(pooled.abs(), 1 - ALPHA)
        per_design = {d: float(torch.quantile(x.abs(), 1 - ALPHA)) for d, x in r.items()}
        q_des = cq(torch.tensor(list(per_design.values())), 1 - DELTA)
        width = float(torch.cat([2 * q_des * sigma(q, sn, mode) for q, _, sn in cal.values()]).mean())
        print(f"  sigma={mode:<4}: q_pin {q_pin:.4f}  q_design {q_des:.4f}  mean design-level width on calibration {width:.3f} ns")
        if best is None or width < best["width"]:
            best = {"mode": mode, "c": c, "q_pin": q_pin, "q_design": q_des, "width": width, "per_design": per_design}
    mode, c = best["mode"], best["c"]
    print(f"chosen sigma = {mode} (smaller width on calibration designs)")
    print("per-calibration-design 90% score:", {d: round(v, 3) for d, v in sorted(best["per_design"].items(), key=lambda kv: kv[1])})

    # ---- apply to designs the final model never trained on ----
    model, ck = load_model(latest(final_tag), dev)
    out = {"alpha": ALPHA, "delta": DELTA, "calibration": best, "final_run": latest(final_tag), "designs": {}}
    hdr = (f"{'design':<14}{'set':<8}{'endpts':>7} | {'pin-level':^25} | {'design-level':^25} | "
           f"{'slack violations: true / pred / flagged(lower bound<0) / true ones caught':^70}")
    print("\n" + hdr)
    print(f"{'':<29} | {'coverage':>10}{'width ns':>12}    | {'coverage':>10}{'width ns':>12}    |")
    for dn in TEST + CUSTOM:
        pred, true, rat, sens = endpoints(dn, model, lib, dev)
        if len(pred) == 0:
            continue
        s = sigma(pred, sens, mode)
        row = {"endpoints": len(pred)}
        line = f"{dn:<14}{'test' if dn in TEST else 'custom':<8}{len(pred):>7} |"
        for name, q in (("pin", best["q_pin"]), ("design", best["q_design"])):
            lo, hi = pred + (c - q) * s, pred + (c + q) * s
            cov = float(((true >= lo) & (true <= hi)).float().mean())
            w = float((hi - lo).mean())
            row[name] = {"coverage": cov, "mean_width_ns": w}
            line += f" {100 * cov:>9.1f}%{w:>12.3f}    |"
        ok = torch.isfinite(rat)
        if ok.any():
            hi = pred + (c + best["q_design"]) * s
            st, sp, sl = (rat - true)[ok], (rat - pred)[ok], (rat - hi)[ok]
            nt, npd, nf = int((st < 0).sum()), int((sp < 0).sum()), int((sl < 0).sum())
            caught = int(((st < 0) & (sl < 0)).sum())
            row["slack"] = {"true_violations": nt, "pred_violations": npd, "flagged": nf, "true_caught": caught,
                            "point_pred_caught": int(((st < 0) & (sp < 0)).sum())}
            line += f" {nt:>8} / {npd:>6} / {nf:>6} / {caught:>6} (point prediction alone caught {row['slack']['point_pred_caught']})"
        else:
            line += "   no required times (unconstrained outputs)"
        out["designs"][dn] = row
        print(line)
    for grp in ("test", "custom"):
        rows = [out["designs"][d] for d in (TEST if grp == "test" else CUSTOM) if d in out["designs"]]
        if rows:
            n_ok = sum(r["design"]["coverage"] >= 1 - ALPHA for r in rows)
            print(f"{grp}: designs with >= {100 * (1 - ALPHA):.0f}% of endpoints inside the design-level interval: {n_ok} of {len(rows)} "
                  f"(target share of designs >= {100 * (1 - DELTA):.0f}%)")
    path = os.path.join(latest(final_tag), "conformal.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
