"""
sta_v2.py -- differentiable 4-channel static timing propagation.

Channels: [early-rise, early-fall, late-rise, late-fall]  (ch = corner*2 + transition).
Early channels propagate with MIN, late channels with MAX, rise/fall are linked through
each arc's timing sense -- the same recurrence OpenSTA runs.

Cell delay / output slew come from the Liberty NLDM tables (bilinear interpolation with
linear extrapolation) evaluated at (input slew, effective load).  The routing-dependent
quantities are the pluggable (learned) inputs:
    wirecap [N]   wire capacitance of the net each pin drives
    kappa   [M]   effective-capacitance factor per corner-arc (resistive shielding)
    net_delay [E,4]  wire delay per driver->sink edge
    slew_k  [4]   sink slew = sqrt(driver_slew^2 + (slew_k * wire_delay)^2)
"""
import torch

BIG = 1.0e6


def r2_score(pred, true):
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def prepare(g, lib, device):
    """Expand arcs into transition arcs x corners and bucket everything by level."""
    lvl = g["level"]
    N = g["n"]
    arc_src, arc_dst = g["arc_src"], g["arc_dst"]
    sense, launch, tid = g["arc_sense"], g["arc_launch"], g["arc_tid"]
    A = arc_src.numel()
    ar = torch.arange(A)
    parts = []
    for out_tr in (0, 1):
        o = torch.full((A,), out_tr)
        for in_tr in (0, 1):
            i = torch.full((A,), in_tr)
            by_sense = ((sense == 0) & (in_tr == out_tr)) | ((sense == 1) & (in_tr != out_tr)) | (sense == 2)
            keep = ((launch == 0) & by_sense) | ((launch == 1) & (in_tr == 0)) | ((launch == 2) & (in_tr == 1))
            parts.append(torch.stack([ar[keep], i[keep], o[keep]], 1))
    t = torch.cat(parts)                       # [T, 3] = arc, in_tr, out_tr
    rows = []
    for c in (0, 1):
        t_arc, t_in, t_out = t[:, 0], t[:, 1], t[:, 2]
        d_tid = tid[t_arc, c * 2 + t_out]
        s_tid = tid[t_arc, 4 + c * 2 + t_out]
        rows.append(torch.stack([t_arc, arc_src[t_arc], arc_dst[t_arc],
                                 c * 2 + t_in, c * 2 + t_out, d_tid, s_tid], 1))
    ca = torch.cat(rows)                       # [M, 7]
    ca = ca[ca[:, 5] >= 0]                     # drop transitions the library has no table for
    order = torch.argsort(lvl[ca[:, 2]], stable=True)
    ca = ca[order]
    ca_lvl = lvl[ca[:, 2]]
    n_order = torch.argsort(lvl[g["net_dst"]], stable=True)
    net_src, net_dst, nd = g["net_src"][n_order], g["net_dst"][n_order], g["nd"][n_order]
    net_lvl = lvl[net_dst]
    L = int(lvl.max()) + 1
    bounds = torch.arange(L + 1)
    ca_ptr = torch.searchsorted(ca_lvl, bounds)
    net_ptr = torch.searchsorted(net_lvl, bounds)
    # per level: the distinct output pins and each corner-arc's slot among them
    nodes, pos, node_ptr = [], [], [0]
    for lv in range(L):
        a, b = int(ca_ptr[lv]), int(ca_ptr[lv + 1])
        u, inv = torch.unique(ca[a:b, 2], return_inverse=True)
        nodes.append(u); pos.append(inv); node_ptr.append(node_ptr[-1] + u.numel())

    pin_load = torch.zeros(N, 4).index_add_(0, g["net_src"], g["pincap"][g["net_dst"]])
    fanout = torch.zeros(N).index_add_(0, g["net_src"], torch.ones(g["net_src"].numel()))
    sign = torch.tensor([-1.0, -1.0, 1.0, 1.0])

    # sources (level 0) start from their constraint values.  Where the label file has no entry
    # for a port (custom designs), fall back to the value at the pin it feeds.
    AT, SLEW = g["AT"].clone(), g["SLEW"].clone()
    src_at, src_slew = AT.clone(), SLEW.clone()
    miss = (lvl[g["net_src"]] == 0) & ~torch.isfinite(src_at[g["net_src"]]).all(1)
    src_at[g["net_src"][miss]] = AT[g["net_dst"][miss]]
    src_slew[g["net_src"][miss]] = SLEW[g["net_dst"][miss]]
    for c in g.get("clock_ports", []):          # unlabeled clock port: rise at 0, fall at half period
        if not torch.isfinite(AT[c]).all() and g.get("period"):
            src_at[c] = torch.tensor([0.0, g["period"] / 2, 0.0, g["period"] / 2])

    bank_a2, bank_v = torch.from_numpy(lib["bank_a2"]), torch.from_numpy(lib["bank_v"])
    dt = ca[:, 5]
    rd = (bank_v[dt, 3, 5] - bank_v[dt, 3, 1]) / (bank_a2[dt, 5] - bank_a2[dt, 1])   # ~drive resistance, kOhm

    p = {
        "n": N, "L": L, "design": g["design"],
        "ca_arc": ca[:, 0], "ca_src": ca[:, 1], "ca_dst": ca[:, 2],
        "ca_in": ca[:, 3], "ca_out": ca[:, 4], "ca_dtid": ca[:, 5], "ca_stid": ca[:, 6].clamp(min=0),
        "ca_rd": rd, "ca_ptr": ca_ptr, "net_ptr": net_ptr,
        "lvl_nodes": torch.cat(nodes), "ca_pos": torch.cat(pos), "node_ptr": torch.tensor(node_ptr),
        "net_src": net_src, "net_dst": net_dst, "nd": nd,
        "e_tree": g["e_tree"][n_order], "n_tree": g["n_tree"], "edge_dist": g["edge_dist"],
        "cd": g["cd"], "pin_load": pin_load, "fanout": fanout, "sign": sign,
        "AT": AT, "SLEW": SLEW, "RAT": g["RAT"], "src_at": src_at, "src_slew": src_slew,
        "wirecap": g["wirecap"], "level": lvl, "xy": g["xy"], "pincap": g["pincap"],
        "is_port": g["is_port"], "is_src": lvl == 0,
        "bank_a1": torch.from_numpy(lib["bank_a1"]), "bank_a2": bank_a2, "bank_v": bank_v,
    }
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in p.items()}


def to_dev(p, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in p.items()}


def lut(p, tid, slew, load):
    """Bilinear NLDM lookup with linear extrapolation. tid/slew/load: [M]."""
    a1, a2, v = p["bank_a1"][tid], p["bank_a2"][tid], p["bank_v"][tid]
    i = (torch.searchsorted(a1, slew.detach().unsqueeze(1).contiguous()).squeeze(1) - 1).clamp(0, 5)
    j = (torch.searchsorted(a2, load.detach().unsqueeze(1).contiguous()).squeeze(1) - 1).clamp(0, 5)
    m = torch.arange(tid.numel(), device=tid.device)
    x0, x1 = a1[m, i], a1[m, i + 1]
    y0, y1 = a2[m, j], a2[m, j + 1]
    tx = (slew - x0) / (x1 - x0)
    ty = (load - y0) / (y1 - y0)
    v00, v10, v01, v11 = v[m, i, j], v[m, i + 1, j], v[m, i, j + 1], v[m, i + 1, j + 1]
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11


def arc_delays(p, wirecap=None, kappa=None):
    """Cell delay of every corner-arc with the TRUE input slew (no propagation needed)."""
    o, cout = p["ca_dst"], p["ca_out"]
    load = p["pin_load"][o, cout]
    if wirecap is not None:
        load = load + wirecap[o]
    if kappa is not None:
        load = load * kappa
    s_in = torch.nan_to_num(p["src_slew"], nan=0.05)[p["ca_src"], p["ca_in"]]
    return lut(p, p["ca_dtid"], s_in, load)


def propagate(p, cell_mode="lut", wirecap=None, net_delay=None, slew_k=None, kappa=None,
              teacher_slew=False):
    """
    cell_mode : "sdf" use the SDF cell-delay labels (structure check) | "lut" Liberty tables
    Returns AT [N,4], SLEW [N,4], per-corner-arc delay [M], valid mask [N,4].
    """
    N, dev = p["n"], p["AT"].device
    sign = p["sign"]
    src = p["is_src"].unsqueeze(1)
    at0 = torch.nan_to_num(p["src_at"], nan=0.0)
    slew_true = torch.nan_to_num(p["src_slew"], nan=0.05)
    has = torch.isfinite(p["src_at"])
    S = torch.where(src & has, at0 * sign, torch.full_like(at0, -BIG))     # sign-folded arrival
    SL = torch.where(src, slew_true, torch.zeros_like(slew_true))
    dl_parts = []
    for lv in range(1, p["L"]):
        a, b = int(p["net_ptr"][lv]), int(p["net_ptr"][lv + 1])
        if b > a:
            s, d = p["net_src"][a:b], p["net_dst"][a:b]
            Ss, SLs = S[s], SL[s]
            if net_delay is not None:
                w = net_delay[a:b]
                Ss = torch.where(Ss > -BIG / 2, Ss + w * sign, Ss)
                if slew_k is not None:
                    SLs = torch.sqrt(SLs * SLs + (slew_k * w) ** 2 + 1e-12)
            S = S.index_copy(0, d, Ss)
            SL = SL.index_copy(0, d, SLs)
        a, b = int(p["ca_ptr"][lv]), int(p["ca_ptr"][lv + 1])
        if b > a:
            i, o = p["ca_src"][a:b], p["ca_dst"][a:b]
            cin, cout = p["ca_in"][a:b], p["ca_out"][a:b]
            sg = sign[cout]
            s_in = slew_true[i, cin] if teacher_slew else SL[i, cin]
            ld = p["pin_load"][o, cout]
            if wirecap is not None:
                ld = ld + wirecap[o]
            # output slew follows the TOTAL load; only the delay sees the shielded (effective) load
            so = lut(p, p["ca_stid"][a:b], s_in, ld)
            if kappa is not None:
                ld = ld * kappa[a:b]
            if cell_mode == "sdf":
                dly = p["cd"][p["ca_arc"][a:b], cout]
                ok = torch.isfinite(dly)
                dly = torch.nan_to_num(dly, nan=0.0)
            else:
                dly = lut(p, p["ca_dtid"][a:b], s_in, ld)
                ok = torch.ones_like(dly, dtype=torch.bool)
            dl_parts.append(dly)
            arr_in = S[i, cin]
            live = (arr_in > -BIG / 2) & ok
            na, nb = int(p["node_ptr"][lv]), int(p["node_ptr"][lv + 1])
            nodes = p["lvl_nodes"][na:nb]
            flat = p["ca_pos"][a:b] * 4 + cout
            n4 = (nb - na) * 4
            cand = torch.where(live, arr_in + dly * sg, torch.full_like(dly, -BIG))
            buf = torch.full((n4,), -BIG, device=dev).scatter_reduce(0, flat, cand, "amax", include_self=True)
            sl_cand = torch.where(live, so * sg, torch.full_like(so, -BIG))
            sbuf = torch.full((n4,), -BIG, device=dev).scatter_reduce(0, flat, sl_cand, "amax", include_self=True)
            sbuf = sbuf.view(-1, 4)
            S = S.index_copy(0, nodes, buf.view(-1, 4))
            SL = SL.index_copy(0, nodes, torch.where(sbuf > -BIG / 2, sbuf * sign, torch.zeros_like(sbuf)))
    valid = S > -BIG / 2
    AT = torch.where(valid, S * sign, torch.zeros_like(S))
    delays = torch.cat(dl_parts) if dl_parts else torch.zeros(0, device=dev)
    return AT, SL, delays, valid


def report(p, AT, SL, valid, tag):
    lab = torch.isfinite(p["AT"])
    both = lab & valid & ~p["is_src"].unsqueeze(1)
    t, q = p["AT"][both], AT[both]
    per = [r2_score(AT[:, c][both[:, c]], p["AT"][:, c][both[:, c]]) for c in range(4)]
    sl_ok = both & torch.isfinite(p["SLEW"])
    mism = (lab != valid).float().mean().item()
    print(f"  {tag:<28} AT R2={r2_score(q, t):.4f}  [ER {per[0]:.4f} EF {per[1]:.4f} LR {per[2]:.4f} LF {per[3]:.4f}]"
          f"  MAE={float((q - t).abs().mean()):.4f}ns  slew R2={r2_score(SL[sl_ok], p['SLEW'][sl_ok]):.4f}"
          f"  validity mismatch={mism:.4f}")
    return r2_score(q, t)


if __name__ == "__main__":
    import sys
    from liberty_v2 import load_liberty
    from build_v2 import load
    lib = load_liberty()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k2 = torch.full((4,), 2.0, device=dev)
    for dn in sys.argv[1:]:
        p = prepare(load(dn, lib), lib, dev)
        print(f"{dn}: pins={p['n']} levels={p['L']} corner-arcs={p['ca_arc'].numel()}")
        with torch.no_grad():
            wc_spef = torch.nan_to_num(p["wirecap"], nan=0.0)
            AT, SL, _, v = propagate(p, "sdf", net_delay=p["nd"], slew_k=k2)
            report(p, AT, SL, v, "A. true SDF delays")
            AT, SL, dl, v = propagate(p, "lut", wirecap=wc_spef, net_delay=p["nd"], slew_k=k2, teacher_slew=True)
            report(p, AT, SL, v, "B. LUT, true slew+SPEF cap")
            cdl = p["cd"][p["ca_arc"], p["ca_out"]][int(p["ca_ptr"][1]):]
            ok = torch.isfinite(cdl)
            print(f"     cell-delay R2 (LUT vs SDF) = {r2_score(dl[ok], cdl[ok]):.4f}")
            AT, SL, _, v = propagate(p, "lut", wirecap=wc_spef, net_delay=p["nd"], slew_k=k2)
            report(p, AT, SL, v, "C. LUT, SPEF cap, true net")
            AT, SL, _, v = propagate(p, "lut", wirecap=wc_spef)
            report(p, AT, SL, v, "E. LUT, SPEF cap only")
            AT, SL, _, v = propagate(p, "lut")
            report(p, AT, SL, v, "D. PRE-ROUTE: no wire at all")
