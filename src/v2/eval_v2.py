"""
eval_v2.py -- evaluate a saved run on the 7 test designs and write per-pin predictions.

    python eval_v2.py                 # latest run in runs/
    python eval_v2.py <run_folder>    # a specific run

Prints R2 (standard and DAC'22-code convention) and the true vs predicted values in ns,
and writes runs/<run>/predictions/<design>.csv plus runs/<run>/test_summary.json.
"""
import csv
import glob
import json
import os
import sys

import torch

from build_v2 import load
from liberty_v2 import load_liberty
from sta_v2 import prepare, propagate
from train_v2 import OUT_DIR, WireNet, metrics, run_sta

TEST = ["jpeg_encoder", "usbf_device", "aes192", "xtea", "spm", "y_huff", "synth_ram"]
PAPER = {"jpeg_encoder": 0.8820, "usbf_device": 0.9252, "aes192": 0.8605, "xtea": 0.9135,
         "spm": 0.8975, "y_huff": 0.9256, "synth_ram": 0.8656}      # DAC'22 Table 5, "Full"
PAPER_TRAIN = {"blabla": 0.9616, "usb_cdc_core": 0.9751, "BM64": 0.9766, "salsa20": 0.9624, "aes128": 0.8267,
               "wbqspiflash": 0.9721, "cic_decimator": 0.9840, "aes256": 0.8488, "des": 0.9922, "aes_cipher": 0.9825,
               "picorv32a": 0.9688, "zipdiv": 0.9753, "genericfir": 0.8858, "usb": 0.9784}     # DAC'22 Table 5, "Full"
CH = ["ER", "EF", "LR", "LF"]


def main():
    global TEST, PAPER
    which = "test"
    if "train" in sys.argv[1:]:          # python eval_v2.py [run] train  -> the 14 training designs
        sys.argv.remove("train")
        TEST, PAPER, which = list(PAPER_TRAIN), PAPER_TRAIN, "train"
    run = sys.argv[1] if len(sys.argv) > 1 else sorted(d for d in glob.glob(os.path.join(OUT_DIR, "*")) if os.path.isdir(d))[-1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lib = load_liberty()
    ck = torch.load(os.path.join(run, "wirenet.pt"), weights_only=False, map_location=dev)
    model = WireNet(ck["fn_dim"], ck["fe_dim"], ck["args"]["hidden"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print("run:", run, "| args:", ck["args"])
    os.makedirs(os.path.join(run, "predictions"), exist_ok=True)
    out = {}
    for dn in TEST:
        g = load(dn, lib)
        p = prepare(g, lib, dev)
        with torch.no_grad():
            # sanity check of the pipeline itself: true SDF delays must reproduce true arrival times
            A0, S0, _, v0 = propagate(p, "sdf", net_delay=p["nd"])
            oracle = metrics(p, A0, S0, v0)["AT_R2"]
            AT, SL, v, (wc, _, _) = run_sta(p, model)
        m = metrics(p, AT, SL, v, wc)
        drives = torch.zeros(p["n"], dtype=torch.bool, device=dev)
        drives[p["net_src"]] = True
        drives[p["ca_src"]] = True
        ok = torch.isfinite(p["AT"]).all(1) & v.all(1) & ~p["is_src"]
        ep = ok & ~drives & torch.isfinite(p["RAT"][:, 2:]).all(1)
        at_t, at_p = p["AT"][ep, 2:].max(1).values, AT[ep, 2:].max(1).values
        sk_t = (p["RAT"][ep, 2:] - p["AT"][ep, 2:]).min(1).values
        sk_p = (p["RAT"][ep, 2:] - AT[ep, 2:]).min(1).values
        sl_t, sl_p = p["SLEW"][ok, 2:].max(1).values, SL[ok, 2:].max(1).values
        f = float
        out[dn] = {
            "pins": p["n"], "endpoints": int(ep.sum()), "period": g["period"], "oracle_AT_R2": oracle,
            "paper_AT_R2": PAPER[dn], **{k: m[k] for k in m},
            "latest_arrival_true": f(at_t.max()), "latest_arrival_pred": f(at_p.max()),
            "mean_arrival_true": f(at_t.mean()), "mean_arrival_pred": f(at_p.mean()),
            "endpoint_arrival_MAE": f((at_p - at_t).abs().mean()),
            "WNS_true": f(sk_t.min()), "WNS_pred": f(sk_p.min()),
            "TNS_true": f(sk_t.clamp(max=0).sum()), "TNS_pred": f(sk_p.clamp(max=0).sum()),
            "violations_true": int((sk_t < 0).sum()), "violations_pred": int((sk_p < 0).sum()),
            "slack_MAE": f((sk_p - sk_t).abs().mean()),
            "mean_slew_true": f(sl_t.mean()), "mean_slew_pred": f(sl_p.mean()),
            "max_slew_true": f(sl_t.max()), "max_slew_pred": f(sl_p.max()),
            "slew_MAE": f((sl_p - sl_t).abs().mean()),
        }
        A, Ap, S, Sp, R = p["AT"].cpu(), AT.cpu(), p["SLEW"].cpu(), SL.cpu(), p["RAT"].cpu()
        e = (~drives).cpu()
        with open(os.path.join(run, "predictions", dn + ".csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pin", "is_endpoint"] + ["AT_true_" + c for c in CH] + ["AT_pred_" + c for c in CH]
                       + ["SLEW_true_" + c for c in CH] + ["SLEW_pred_" + c for c in CH] + ["RAT_true_" + c for c in CH])
            for i in range(p["n"]):
                w.writerow([g["names"][i], int(e[i])] + [f"{x:.5f}" for x in
                           A[i].tolist() + Ap[i].tolist() + S[i].tolist() + Sp[i].tolist() + R[i].tolist()])
        del p
        torch.cuda.empty_cache()
    with open(os.path.join(run, which + "_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    pc = lambda x: f"{100 * x:7.2f}%"
    print("\nR2 scores (percent)")
    print(f"{'design':<14}{'oracle':>9}{'paper':>9}{'ours(paper formula)':>21}{'AT':>9}{'ER':>9}{'EF':>9}{'LR':>9}{'LF':>9}{'slew':>9}{'endpt AT':>10}{'slack':>9}{'wire cap':>10}")
    keys = ["oracle_AT_R2", "paper_AT_R2", "AT_R2_paper_conv", "AT_R2", "AT_R2_ER", "AT_R2_EF", "AT_R2_LR", "AT_R2_LF",
            "SLEW_R2", "endpoint_AT_R2", "setup_slack_R2", "wirecap_R2"]
    wd = [9, 9, 21, 9, 9, 9, 9, 9, 9, 10, 9, 10]
    for dn in TEST:
        print(f"{dn:<14}" + "".join(pc(out[dn][k]).rjust(w) for k, w in zip(keys, wd)))
    print(f"{'AVERAGE':<14}" + "".join(pc(sum(out[d][k] for d in TEST) / len(TEST)).rjust(w) for k, w in zip(keys, wd)))

    print("\nValues in ns (late corner, worse of rise/fall), true -> predicted")
    print(f"{'design':<14}{'endpts':>7}{'latest arrival':>20}{'mean arrival':>20}{'AT err':>8}{'worst slack':>22}{'total neg slack':>24}{'violations':>14}{'slk err':>8}{'mean slew':>18}{'max slew':>16}{'slew err':>9}")
    for dn in TEST:
        o = out[dn]
        print(f"{dn:<14}{o['endpoints']:>7}"
              f"{o['latest_arrival_true']:>10.3f} ->{o['latest_arrival_pred']:>7.3f}"
              f"{o['mean_arrival_true']:>10.3f} ->{o['mean_arrival_pred']:>7.3f}{o['endpoint_arrival_MAE']:>8.3f}"
              f"{o['WNS_true']:>11.3f} ->{o['WNS_pred']:>8.3f}"
              f"{o['TNS_true']:>12.1f} ->{o['TNS_pred']:>9.1f}"
              f"{o['violations_true']:>7} ->{o['violations_pred']:>5}{o['slack_MAE']:>8.3f}"
              f"{o['mean_slew_true']:>9.4f} ->{o['mean_slew_pred']:>7.4f}"
              f"{o['max_slew_true']:>8.3f} ->{o['max_slew_pred']:>6.3f}{o['slew_MAE']:>9.4f}")
    print("\nwrote", os.path.join(run, which + "_summary.json"), "and predictions/*.csv")


if __name__ == "__main__":
    main()
