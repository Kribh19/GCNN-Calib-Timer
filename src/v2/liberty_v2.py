"""
liberty_v2.py -- Liberty parser for the v2 (4-channel, two-corner) pipeline.

Parses BOTH corners used by the dataset's OpenSTA run:
    early ("E") = sky130_fd_sc_hd__ff_n40C_1v95.lib   (read_liberty -min)
    late  ("L") = sky130_fd_sc_hd__ss_100C_1v60.lib   (read_liberty -max)

Per cell it keeps pin direction / capacitances and every delay arc in file
order with its timing_sense, timing_type and the four NLDM tables
(cell_rise, cell_fall, rise_transition, fall_transition).

All tables are packed into one global "bank" (axes + values) so a design
only stores integer table ids.
"""
import os
import pickle
import re

import numpy as np

TECHLIB_DIR = r"C:\Krish\GCNN\data\techlib"
LIBS = {"E": "sky130_fd_sc_hd__ff_n40C_1v95.lib",
        "L": "sky130_fd_sc_hd__ss_100C_1v60.lib"}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "liberty_v2.pkl")

DELAY_TABS = ("cell_rise", "cell_fall", "rise_transition", "fall_transition")
CHECK_TABS = ("rise_constraint", "fall_constraint")
G_OPEN = re.compile(r'^(\w+)\s*\((.*)\)\s*\{$')
ATTR = re.compile(r'^(\w+)\s*:\s*(.*?)\s*;?$')
CATTR = re.compile(r'^(\w+)\s*\((.*)\)\s*;?$')
NUM = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def _nums(s):
    return [float(x) for x in NUM.findall(s)]


def parse_lib(path):
    cells = {}
    stack = []
    cell = pin = timing = table = None
    pending = None
    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if pending is not None:
                if line.endswith("\\"):
                    pending += line[:-1]
                    continue
                line = pending + line
                pending = None
            elif line.endswith("\\"):
                pending = line[:-1]
                continue
            if not line:
                continue
            if line == "}":
                if stack:
                    kind = stack.pop()
                    depth = len(stack)
                    if kind == "cell" and depth == 1:
                        cell = None
                    elif kind == "pin" and depth == 2:
                        pin = None
                    elif kind == "timing" and depth == 3:
                        if timing is not None and pin is not None:
                            if any(t in timing["tabs"] for t in DELAY_TABS[:2]):
                                for rp in timing["from"].split():
                                    arc = dict(timing)
                                    arc["from"] = rp
                                    cell["arcs"].setdefault((rp, pin["name"]), []).append(arc)
                            elif any(t in timing["tabs"] for t in CHECK_TABS):
                                for rp in timing["from"].split():
                                    chk = dict(timing)
                                    chk["from"] = rp
                                    cell["checks"].setdefault((rp, pin["name"]), []).append(chk)
                        timing = None
                    elif depth == 4:
                        table = None
                continue
            m = G_OPEN.match(line)
            if m:
                kind, arg = m.group(1), m.group(2).strip().strip('"')
                depth = len(stack)
                stack.append(kind)
                if kind == "cell" and depth == 1:
                    cell = {"pins": {}, "arcs": {}, "checks": {}}
                    cells[arg] = cell
                elif kind == "pin" and depth == 2 and cell is not None:
                    pin = {"name": arg, "direction": "input", "cap": 0.0,
                           "rise_cap": None, "fall_cap": None, "clock": False}
                    cell["pins"][arg] = pin
                elif kind == "timing" and depth == 3 and pin is not None:
                    timing = {"from": "", "sense": "non_unate", "type": "combinational",
                              "when": "", "tabs": {}}
                elif depth == 4 and timing is not None and (kind in DELAY_TABS or kind in CHECK_TABS):
                    table = {"i1": [], "i2": [], "v": []}
                    timing["tabs"][kind] = table
                continue
            if line.endswith("{"):
                stack.append("?")
                continue
            depth = len(stack)
            if table is not None and depth == 5:
                m = CATTR.match(line)
                if m:
                    k = m.group(1)
                    if k == "index_1":
                        table["i1"] = _nums(m.group(2))
                    elif k == "index_2":
                        table["i2"] = _nums(m.group(2))
                    elif k == "values":
                        table["v"] = _nums(m.group(2))
                continue
            if timing is not None and depth == 4:
                m = ATTR.match(line)
                if m:
                    k, v = m.group(1), m.group(2).strip('"')
                    if k == "related_pin":
                        timing["from"] = v
                    elif k == "timing_sense":
                        timing["sense"] = v
                    elif k == "timing_type":
                        timing["type"] = v
                    elif k == "when":
                        timing["when"] = v
                continue
            if pin is not None and depth == 3:
                m = ATTR.match(line)
                if m:
                    k, v = m.group(1), m.group(2).strip('"')
                    if k == "direction":
                        pin["direction"] = v
                    elif k == "capacitance":
                        pin["cap"] = float(v)
                    elif k == "rise_capacitance":
                        pin["rise_cap"] = float(v)
                    elif k == "fall_capacitance":
                        pin["fall_cap"] = float(v)
                    elif k == "clock":
                        pin["clock"] = (v == "true")
    return cells


def _to77(tab):
    """Return (axis1[7], axis2[7], values[7,7]); None if the table is unusable."""
    i1, i2, v = tab["i1"], tab["i2"], tab["v"]
    if not v:
        return None
    if len(i1) == 7 and len(i2) == 7 and len(v) == 49:
        return (np.array(i1, np.float32), np.array(i2, np.float32),
                np.array(v, np.float32).reshape(7, 7))
    if len(v) == 1:
        return (np.linspace(0.01, 1.5, 7, dtype=np.float32),
                np.linspace(0.0005, 0.3, 7, dtype=np.float32),
                np.full((7, 7), v[0], np.float32))
    return None


def load_liberty(force=False):
    """Returns dict with:
       cells[corner][cell] -> {"pins", "arcs", "checks"}  (arcs carry "tid": {tabname: bank id})
       bank_a1 [T,7], bank_a2 [T,7], bank_v [T,7,7]"""
    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    out = {"cells": {}}
    a1, a2, vv = [], [], []
    n_bad = 0
    for corner, fname in LIBS.items():
        path = os.path.join(TECHLIB_DIR, fname)
        print(f"[liberty_v2] parsing {path} ...")
        cells = parse_lib(path)
        for cname in sorted(cells):
            for key in sorted(cells[cname]["arcs"]):
                for arc in cells[cname]["arcs"][key]:
                    arc["tid"] = {}
                    for t in DELAY_TABS:
                        if t in arc["tabs"]:
                            r = _to77(arc["tabs"][t])
                            if r is None:
                                n_bad += 1
                                continue
                            arc["tid"][t] = len(vv)
                            a1.append(r[0]); a2.append(r[1]); vv.append(r[2])
                    del arc["tabs"]
        out["cells"][corner] = cells
        n_arcs = sum(len(v) for c in cells.values() for v in c["arcs"].values())
        print(f"[liberty_v2]   {len(cells)} cells, {n_arcs} delay arcs")
    out["bank_a1"] = np.stack(a1)
    out["bank_a2"] = np.stack(a2)
    out["bank_v"] = np.stack(vv)
    print(f"[liberty_v2] table bank: {len(vv)} tables ({n_bad} unusable skipped)")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


if __name__ == "__main__":
    lib = load_liberty(force=True)
    for corner in ("E", "L"):
        c = lib["cells"][corner]["sky130_fd_sc_hd__nand2_1"]
        print(corner, {k: (v["direction"], v["cap"], v["rise_cap"], v["fall_cap"]) for k, v in c["pins"].items()})
        for key, arcs in c["arcs"].items():
            for a in arcs:
                print("   ", key, a["sense"], a["type"], a["tid"])
        d = lib["cells"][corner]["sky130_fd_sc_hd__dfrtp_1"]
        for key, arcs in d["arcs"].items():
            for a in arcs:
                print("   dfrtp", key, a["sense"], a["type"], sorted(a["tid"]))
        x = lib["cells"][corner]["sky130_fd_sc_hd__xor2_1"]
        for key, arcs in x["arcs"].items():
            for a in arcs:
                print("   xor2", key, a["sense"], a["type"], a["when"], sorted(a["tid"]))
    t = lib["cells"]["L"]["sky130_fd_sc_hd__nand2_1"]["arcs"][("A", "Y")][0]["tid"]["cell_rise"]
    print("axis1", lib["bank_a1"][t]); print("axis2", lib["bank_a2"][t]); print(lib["bank_v"][t])
