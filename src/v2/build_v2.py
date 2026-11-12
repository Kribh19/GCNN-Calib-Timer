"""
build_v2.py -- build the 4-channel timing graph of one design.

Channels everywhere are ordered  [early-rise, early-fall, late-rise, late-fall]
(early = ff corner / min analysis, late = ss corner / max analysis), exactly the
EL/RF convention of the DAC'22 TimingGCN dataset.

Sources (per design folder):
    <d>.at.sdf    graph structure (INTERCONNECT / IOPATH), per-channel delay labels,
                  per-pin AT / SLEW / RAT labels
    *.def         placement (pre-routing input)
    *.sdc         clock period, set_load on output ports
    *.spef        ONLY as a training label for wire capacitance (never a model input)
    liberty       pin caps, arc sense/type, NLDM table ids  (pre-routing input)
"""
import os
import re
import sys
import time

import numpy as np
import torch

from liberty_v2 import load_liberty
from tree_v2 import net_trees

DATA_DIR = r"C:\Krish\GCNN\data"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
POWER = {"VGND", "VPWR", "VNB", "VPB"}
SENSE = {"positive_unate": 0, "negative_unate": 1, "non_unate": 2}
# arc "launch" code: 0 = follows timing_sense, 1 = launched by input RISE, 2 = by input FALL,
# 3 = not propagated (async clear/preset arcs are disabled in OpenSTA by default)
LAUNCH = {"rising_edge": 1, "falling_edge": 2, "clear": 3, "preset": 3}
VAL = re.compile(r'\(([^()]*)\)')
IOPATH = re.compile(r'\(IOPATH\s+(?:\((\w+)\s+([^)\s]+)\)|(\S+))\s+(\S+)\s+\(([^()]*)\)\s+\(([^()]*)\)')
INTER = re.compile(r'\(INTERCONNECT\s+(\S+)\s+(\S+)\s+\(([^()]*)\)\s+\(([^()]*)\)')
ATL = re.compile(r'\((AT|SLEW|RAT)\s+(\S+)\s+\(([^()]*)\)\s+\(([^()]*)\)')


def mm(s):
    """'(min::max)' body -> (min, max); empty -> (nan, nan)."""
    p = [x for x in s.split(":") if x.strip()]
    if not p:
        return (float("nan"), float("nan"))
    return (float(p[0]), float(p[-1]))


def is_power(name):
    return name.rsplit("/", 1)[-1] in POWER


def norm(name):
    return name.replace("\\", "")


def parse_at_sdf(path):
    iop, inter = [], []
    at, slew, rat = {}, {}, {}
    inst_type = {}
    ctype = inst = None
    with open(path, errors="replace") as f:
        for line in f:
            s = line.lstrip()
            if s.startswith("(AT ") or s.startswith("(SLEW ") or s.startswith("(RAT "):
                m = ATL.match(s)
                if not m or is_power(m.group(2)):
                    continue
                r, fl = mm(m.group(3)), mm(m.group(4))
                v = (r[0], fl[0], r[1], fl[1])
                {"AT": at, "SLEW": slew, "RAT": rat}[m.group(1)][norm(m.group(2))] = v
            elif s.startswith("(INTERCONNECT"):
                m = INTER.match(s)
                if not m or is_power(m.group(1)) or is_power(m.group(2)):
                    continue
                r, fl = mm(m.group(3)), mm(m.group(4))
                inter.append((norm(m.group(1)), norm(m.group(2)), (r[0], fl[0], r[1], fl[1])))
            elif s.startswith("(IOPATH"):
                m = IOPATH.match(s)
                if not m:
                    continue
                a = m.group(2) or m.group(3)
                b = m.group(4)
                if is_power(a) or is_power(b):
                    continue
                r, fl = mm(m.group(5)), mm(m.group(6))
                iop.append((inst, ctype, a, b, (r[0], fl[0], r[1], fl[1])))
            elif s.startswith("(CELLTYPE"):
                ctype = s.split('"')[1]
            elif s.startswith("(INSTANCE"):
                inst = norm(s[len("(INSTANCE"):].strip().rstrip(")").strip())
                inst_type[inst] = ctype
    return iop, inter, at, slew, rat, inst_type


def parse_def(path):
    text = open(path, errors="replace").read()
    m = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', text)
    dbu = float(m.group(1)) if m else 1000.0
    die = (0.0, 0.0, 1000.0, 1000.0)
    m = re.search(r'DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', text)
    if m:
        die = tuple(int(m.group(i)) / dbu for i in range(1, 5))
    inst = {}
    a, b = text.find("\nCOMPONENTS"), text.find("END COMPONENTS")
    if a >= 0:
        for m in re.finditer(r'-\s+(\S+)\s+(\S+)\s[^;]*?\+\s+(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\w+)',
                             text[a:b]):
            inst[norm(m.group(1))] = (m.group(2), int(m.group(3)) / dbu, int(m.group(4)) / dbu, m.group(5))
    ports = {}
    a, b = text.find("\nPINS"), text.find("END PINS")
    if a >= 0:
        for m in re.finditer(r'-\s+(\S+)\s+\+\s+NET\s+\S+[^;]*?\+\s*(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)',
                             text[a:b], re.S):
            ports[norm(m.group(1))] = (int(m.group(2)) / dbu, int(m.group(3)) / dbu)
    return inst, ports, die


_LEF = None


def load_lef():
    """cell -> (width, height, {pin: (x, y) centre of the pin shapes, relative to the cell origin})."""
    global _LEF
    if _LEF is not None:
        return _LEF
    lef, macro, pin, rects = {}, None, None, []
    path = os.path.join(DATA_DIR, "techlib", "merged_unpadded.lef")
    with open(path, errors="replace") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "MACRO":
                macro = {"w": 0.0, "h": 0.0, "pins": {}}
                lef[t[1]] = macro
            elif macro is None:
                continue
            elif t[0] == "SIZE":
                macro["w"], macro["h"] = float(t[1]), float(t[3])
            elif t[0] == "PIN":
                pin, rects = t[1], []
            elif t[0] == "RECT" and pin is not None:
                rects.append([float(x) for x in t[1:5]])
            elif t[0] == "END" and len(t) > 1 and pin is not None and t[1] == pin:
                if rects:
                    r = np.array(rects)
                    area = np.maximum((r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]), 1e-9)
                    macro["pins"][pin] = (float(((r[:, 0] + r[:, 2]) / 2 * area).sum() / area.sum()),
                                          float(((r[:, 1] + r[:, 3]) / 2 * area).sum() / area.sum()))
                pin = None
    _LEF = lef
    return lef


def pin_xy(info, pn, lef):
    """Absolute pin position from the placed cell origin, its orientation and the LEF pin shape."""
    ctype, x, y, orient = info
    m = lef.get(ctype)
    if not m or pn not in m["pins"]:
        return x, y
    px, py = m["pins"][pn]
    if orient in ("S", "FN"):
        px = m["w"] - px
    if orient in ("S", "FS"):
        py = m["h"] - py
    return x + px, y + py


def parse_sdc(path):
    period, loads, clocks = None, {}, []
    if not path:
        return period, loads, clocks
    text = open(path, errors="replace").read()
    m = re.search(r'create_clock\b.*?-period\s+([\d.]+)', text)
    if m:
        period = float(m.group(1))
    for m in re.finditer(r'set_load\s+-pin_load\s+([\d.eE+-]+)\s+\[get_ports\s*\{?\s*([^\}\]]+?)\s*\}?\s*\]', text):
        loads[norm(m.group(2))] = float(m.group(1))
    for m in re.finditer(r'create_clock\b[^\n]*?\[get_ports\s*\{?\s*([^\}\]]+?)\s*\}?\s*\]', text):
        clocks.append(norm(m.group(1)))
    return period, loads, clocks


def parse_spef_wirecap(path):
    """driver pin name -> total wire cap (pF) of the net it drives. Training label only."""
    names, caps = {}, {}
    cunit = 1.0
    cur = None
    in_map = in_conn = False
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("*C_UNIT"):
                p = line.split()
                cunit = float(p[1]) * {"PF": 1.0, "FF": 1e-3}.get(p[2].upper(), 1.0)
            elif line.startswith("*NAME_MAP"):
                in_map = True
            elif line.startswith("*D_NET"):
                in_map = False
                p = line.split()
                cur = float(p[2]) * cunit
                in_conn = False
            elif in_map and line.startswith("*"):
                p = line.split()
                if len(p) == 2:
                    names[p[0]] = norm(p[1])
            elif line.startswith("*CONN"):
                in_conn = True
            elif line.startswith("*CAP") or line.startswith("*RES") or line.startswith("*END"):
                in_conn = False
            elif in_conn and cur is not None:
                p = line.split()
                if len(p) >= 3 and p[0] == "*I" and p[2] == "O":
                    i, pn = p[1].split(":")
                    caps[names.get(i, norm(i)) + "/" + pn] = cur
                elif len(p) >= 3 and p[0] == "*P" and p[2] == "I":
                    caps[names.get(p[1], norm(p[1]))] = cur
    return caps


def find(design_dir, design, suffixes):
    for f in sorted(os.listdir(design_dir)):
        for s in suffixes:
            if f.endswith(s):
                return os.path.join(design_dir, f)
    return None


def build(design, lib=None, verbose=True):
    t0 = time.time()
    lib = lib or load_liberty()
    d = os.path.join(DATA_DIR, design)
    iop, inter, at, slew, rat, inst_type = parse_at_sdf(os.path.join(d, f"{design}.at.sdf"))
    if not iop:   # custom designs keep the delay section in <design>.sdf and only AT/SLEW in .at.sdf
        iop, inter, _, _, _, inst_type = parse_at_sdf(os.path.join(d, f"{design}.sdf"))
    dinst, dports, die = parse_def(find(d, design, [".def"]))
    period, port_load, clock_ports = parse_sdc(find(d, design, [".sdc"]))
    spef = find(d, design, [".spef"])
    wirecap_by_drv = parse_spef_wirecap(spef) if spef else {}

    # ---- nodes: every pin that appears on a net edge or a cell arc ----
    names, idx = [], {}

    def nid(n):
        i = idx.get(n)
        if i is None:
            i = idx[n] = len(names)
            names.append(n)
        return i

    net_src, net_dst, nd = [], [], []
    for a, b, v in inter:
        net_src.append(nid(a)); net_dst.append(nid(b)); nd.append(v)

    cells_E, cells_L = lib["cells"]["E"], lib["cells"]["L"]
    arc_src, arc_dst, cd, arc_sense, arc_launch, arc_tid = [], [], [], [], [], []
    seen = {}
    n_nolib = 0
    for inst, ctype, a, b, v in iop:
        k = seen.get((inst, a, b), 0)
        seen[(inst, a, b)] = k + 1
        tids = [-1] * 8  # [delay E-r, E-f, L-r, L-f, trans E-r, E-f, L-r, L-f]
        sense, launch, ok = 2, 0, False
        for ci, cells in enumerate((cells_E, cells_L)):
            arcs = cells.get(ctype, {}).get("arcs", {}).get((a, b), [])
            if len(arcs) > 1:
                # OpenSTA writes the IOPATHs of one pin pair negative-unate first (measured on
                # all mux2/mux4 variants: 0.007 ns error this way vs 0.4 ns in Liberty order)
                arcs = sorted(arcs, key=lambda x: {"negative_unate": 0, "positive_unate": 1}.get(x["sense"], 2))
            if k < len(arcs):
                arc = arcs[k]
                ok = True
                sense = SENSE.get(arc["sense"], 2)
                launch = LAUNCH.get(arc["type"], 0)
                t = arc["tid"]
                tids[ci * 2 + 0] = t.get("cell_rise", -1)
                tids[ci * 2 + 1] = t.get("cell_fall", -1)
                tids[4 + ci * 2 + 0] = t.get("rise_transition", -1)
                tids[4 + ci * 2 + 1] = t.get("fall_transition", -1)
        if not ok:
            n_nolib += 1
        arc_src.append(nid(f"{inst}/{a}")); arc_dst.append(nid(f"{inst}/{b}"))
        cd.append(v); arc_sense.append(sense); arc_launch.append(launch); arc_tid.append(tids)

    N = len(names)
    lef = load_lef()
    net_src = np.array(net_src, np.int64); net_dst = np.array(net_dst, np.int64)
    arc_src = np.array(arc_src, np.int64); arc_dst = np.array(arc_dst, np.int64)

    # ---- per-pin attributes ----
    xy = np.zeros((N, 2), np.float32)
    pincap = np.zeros((N, 4), np.float32)      # input-pin cap per channel [E-r, E-f, L-r, L-f]
    is_port = np.zeros(N, bool)
    n_noxy = 0
    for i, n in enumerate(names):
        if "/" in n:
            inst, pn = n.rsplit("/", 1)
            info = dinst.get(inst)
            if info:
                xy[i] = pin_xy(info, pn, lef)
            else:
                n_noxy += 1
            ctype = inst_type.get(inst) or (info[0] if info else None)
            for ci, cells in enumerate((cells_E, cells_L)):
                p = cells.get(ctype, {}).get("pins", {}).get(pn)
                if p and p["direction"] == "input":
                    pincap[i, ci * 2 + 0] = p["rise_cap"] if p["rise_cap"] is not None else p["cap"]
                    pincap[i, ci * 2 + 1] = p["fall_cap"] if p["fall_cap"] is not None else p["cap"]
        else:
            is_port[i] = True
            if n in dports:
                xy[i] = dports[n]
            else:
                n_noxy += 1
            if n in port_load:
                pincap[i] = port_load[n]

    def lab(dct):
        out = np.full((N, 4), np.nan, np.float32)
        for i, n in enumerate(names):
            v = dct.get(n)
            if v is not None:
                out[i] = v
        out[np.abs(out) > 1e20] = np.nan
        return out

    AT, SLEW, RAT = lab(at), lab(slew), lab(rat)
    wirecap = np.full(N, np.nan, np.float32)
    for n, c in wirecap_by_drv.items():
        i = idx.get(n)
        if i is not None:
            wirecap[i] = c

    # ---- topological levels over net edges + cell arcs ----
    src_all = np.concatenate([net_src, arc_src]); dst_all = np.concatenate([net_dst, arc_dst])
    order = np.argsort(src_all, kind="stable")
    ptr = np.zeros(N + 1, np.int64)
    np.add.at(ptr, src_all + 1, 1)
    ptr = np.cumsum(ptr)
    dst_sorted = dst_all[order]
    indeg = np.bincount(dst_all, minlength=N)
    level = np.zeros(N, np.int64)
    stack = list(np.nonzero(indeg == 0)[0])
    n_seen = 0
    indeg = indeg.tolist(); lvl = level.tolist(); ptrl = ptr.tolist(); dsl = dst_sorted.tolist()
    while stack:
        u = stack.pop(); n_seen += 1
        lu = lvl[u] + 1
        for e in range(ptrl[u], ptrl[u + 1]):
            v = dsl[e]
            if lvl[v] < lu:
                lvl[v] = lu
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    level = np.array(lvl, np.int64)
    n_cyc = N - n_seen

    # ---- placement-only routing-tree estimate and distance to the die boundary ----
    e_lp, e_ew, e_ep, e_dep, n_lt, n_mw, n_mp = net_trees(N, xy, net_src, net_dst, pincap[net_dst, 2] * 1000.0)
    edge_dist = np.stack([xy[:, 0] - die[0], die[2] - xy[:, 0], xy[:, 1] - die[1], die[3] - xy[:, 1]], 1)

    g = {
        "e_tree": torch.from_numpy(np.stack([e_lp, e_ew, e_ep, e_dep], 1)),
        "n_tree": torch.from_numpy(np.stack([n_lt, n_mw, n_mp], 1)),
        "edge_dist": torch.from_numpy(edge_dist.astype(np.float32)).clamp(min=0),
        "design": design, "names": names, "n": N, "die": die, "period": period,
        "clock_ports": [idx[c] for c in clock_ports if c in idx],
        "xy": torch.from_numpy(xy), "pincap": torch.from_numpy(pincap),
        "is_port": torch.from_numpy(is_port), "level": torch.from_numpy(level),
        "AT": torch.from_numpy(AT), "SLEW": torch.from_numpy(SLEW), "RAT": torch.from_numpy(RAT),
        "wirecap": torch.from_numpy(wirecap),
        "net_src": torch.from_numpy(net_src), "net_dst": torch.from_numpy(net_dst),
        "nd": torch.tensor(nd, dtype=torch.float32).reshape(-1, 4),
        "arc_src": torch.from_numpy(arc_src), "arc_dst": torch.from_numpy(arc_dst),
        "cd": torch.tensor(cd, dtype=torch.float32).reshape(-1, 4),
        "arc_sense": torch.tensor(arc_sense, dtype=torch.long),
        "arc_launch": torch.tensor(arc_launch, dtype=torch.long),
        "arc_tid": torch.tensor(arc_tid, dtype=torch.long).reshape(-1, 8),
    }
    if verbose:
        print(f"[{design}] pins={N} net_edges={len(net_src)} cell_arcs={len(arc_src)} "
              f"levels={int(level.max()) + 1} cyc={n_cyc} no_lib_arc={n_nolib} no_xy={n_noxy} "
              f"AT_valid={np.isfinite(AT).all(1).mean():.3f} wirecap_lab={np.isfinite(wirecap).sum()} "
              f"period={period} ({time.time() - t0:.1f}s)")
    return g


def load(design, lib=None, rebuild=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, f"{design}.pt")
    if os.path.exists(p) and not rebuild:
        return torch.load(p, weights_only=False)
    g = build(design, lib)
    torch.save(g, p)
    return g


if __name__ == "__main__":
    lib = load_liberty()
    for dn in sys.argv[1:]:
        load(dn, lib, rebuild=True)
