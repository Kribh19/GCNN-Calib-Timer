"""
gen_circuits.py
================
Generates SYNTHETIC circuits that mimic the structure of a real netlist,
and computes GROUND-TRUTH timing with a real STA algorithm.

WHY: this lets you build + train the whole GNN today, without fighting
OpenLane first. Later, you swap this file for a real DEF/SPEF parser and
NOTHING ELSE in your pipeline has to change.

Graph structure matches the paper:
  - node          = a PIN
  - net edge      = wire  (driver pin -> sink pin)
  - cell edge     = gate  (input pin  -> output pin, same gate)
"""
import random
import numpy as np

# ----------------------------------------------------------------------
# Cell library (a toy stand-in for a real .lib file).
# Real .lib gives a 7x7 LUT indexed by (input_slew, output_load).
# We use a small 3x3 LUT so you can SEE the interpolation working.
# ----------------------------------------------------------------------
SLEW_IDX = np.array([0.05, 0.20, 0.60])          # input slew axis (ns)
LOAD_IDX = np.array([0.10, 0.50, 2.00])          # output load axis (pF)

# delay LUT[gate][slew_i][load_j]  (ns)
CELL_LUT = {
    "INV":  np.array([[0.04, 0.09, 0.28],
                      [0.06, 0.12, 0.33],
                      [0.11, 0.19, 0.45]]),
    "NAND": np.array([[0.07, 0.15, 0.42],
                      [0.10, 0.20, 0.50],
                      [0.17, 0.30, 0.68]]),
    "NOR":  np.array([[0.08, 0.17, 0.47],
                      [0.11, 0.23, 0.56]  ,
                      [0.19, 0.34, 0.75]]),
    "BUF":  np.array([[0.03, 0.07, 0.22],
                      [0.05, 0.10, 0.27],
                      [0.09, 0.16, 0.38]]),
}
# slew LUT: output slew as a function of the same axes
SLEW_LUT = {g: (v * 0.9 + 0.03) for g, v in CELL_LUT.items()}

GATE_TYPES = list(CELL_LUT.keys())
GATE_ID = {g: i for i, g in enumerate(GATE_TYPES)}
N_IN = {"INV": 1, "BUF": 1, "NAND": 2, "NOR": 2}

PIN_CAP = 0.06          # input-pin capacitance of every gate (pF)
WIRE_CAP_PER_UNIT = 0.05   # pF per unit of wire length
WIRE_RES_PER_UNIT = 0.10   # ohm-ish per unit


def bilinear(lut, slew, load):
    """2-D interpolation on the LUT -- exactly what a real STA engine does."""
    si = np.clip(np.searchsorted(SLEW_IDX, slew) - 1, 0, len(SLEW_IDX) - 2)
    li = np.clip(np.searchsorted(LOAD_IDX, load) - 1, 0, len(LOAD_IDX) - 2)
    s0, s1 = SLEW_IDX[si], SLEW_IDX[si + 1]
    l0, l1 = LOAD_IDX[li], LOAD_IDX[li + 1]
    ts = np.clip((slew - s0) / (s1 - s0), 0, 1)
    tl = np.clip((load - l0) / (l1 - l0), 0, 1)
    v00, v01 = lut[si, li],     lut[si, li + 1]
    v10, v11 = lut[si + 1, li], lut[si + 1, li + 1]
    return (v00 * (1 - ts) * (1 - tl) + v01 * (1 - ts) * tl +
            v10 * ts * (1 - tl)       + v11 * ts * tl)


def gen_circuit(n_gates=40, n_inputs=6, seed=0, canvas=20.0):
    """Build one random circuit + placement, then run TRUE STA on it."""
    rng = random.Random(seed)

    pins = []            # list of dicts, one per PIN (= one graph node)
    net_edges = []       # (driver_pin, sink_pin)
    cell_edges = []      # (input_pin, output_pin)  same gate
    gate_of_out = {}     # output_pin -> gate type

    def add_pin(kind):
        pins.append({"kind": kind,
                     "x": rng.uniform(0, canvas),
                     "y": rng.uniform(0, canvas)})
        return len(pins) - 1

    # primary inputs
    pi = [add_pin("PI") for _ in range(n_inputs)]
    available = list(pi)              # pins whose signal can drive something

    for _ in range(n_gates):
        g = rng.choice(GATE_TYPES)
        ins = [add_pin("IN") for _ in range(N_IN[g])]
        out = add_pin("OUT")
        gate_of_out[out] = g
        # wire each gate input to some already-available driver pin
        for ip in ins:
            drv = rng.choice(available)
            net_edges.append((drv, ip))
            cell_edges.append((ip, out))     # input -> output, through the gate
        available.append(out)

    # endpoints = pins that drive nothing (the "flip-flop D pins")
    driven = {d for d, _ in net_edges}
    endpoints = [p for p in range(len(pins))
                 if pins[p]["kind"] == "OUT" and p not in driven]

    # ---------------- TRUE STA (this is our ground-truth label) ------------
    fanout = {}
    for d, s in net_edges:
        fanout.setdefault(d, []).append(s)

    def dist(a, b):
        return abs(pins[a]["x"] - pins[b]["x"]) + abs(pins[a]["y"] - pins[b]["y"])

    # 1) net load: total cap the driver must charge = sum of wire + pin caps
    net_load = {}
    for d, sinks in fanout.items():
        wl = sum(dist(d, s) for s in sinks)
        net_load[d] = wl * WIRE_CAP_PER_UNIT + len(sinks) * PIN_CAP

    # 2) net delay (Elmore-like): depends on distance AND total net load
    true_net_delay = {}
    for d, sinks in fanout.items():
        for s in sinks:
            r = dist(d, s) * WIRE_RES_PER_UNIT
            true_net_delay[(d, s)] = r * (net_load[d] * 0.5 + PIN_CAP)

    # 3) topological order over the DAG of pins
    preds = {}
    for d, s in net_edges:
        preds.setdefault(s, []).append(("net", d))
    for i, o in cell_edges:
        preds.setdefault(o, []).append(("cell", i))

    order, seen = [], set()
    def visit(p):
        if p in seen:
            return
        seen.add(p)
        for _, q in preds.get(p, []):
            visit(q)
        order.append(p)
    for p in range(len(pins)):
        visit(p)

    # 4) propagate arrival time + slew, level by level  <-- THE STA CORE
    AT = {p: 0.0 for p in range(len(pins))}
    SLEW = {p: 0.08 for p in range(len(pins))}      # input slew at PIs
    true_cell_delay = {}

    for p in order:
        if p not in preds:
            continue
        best_at, best_slew = -1e9, 0.08
        for kind, q in preds[p]:
            if kind == "net":
                # net edge: add wire delay; slew degrades slightly
                at = AT[q] + true_net_delay[(q, p)]
                sl = SLEW[q] * 1.05
            else:
                # cell edge: LUT lookup with (input slew, output load)
                g = gate_of_out[p]
                load = net_load.get(p, PIN_CAP)      # what this gate drives
                cd = bilinear(CELL_LUT[g], SLEW[q], load)
                so = bilinear(SLEW_LUT[g], SLEW[q], load)
                true_cell_delay[(q, p)] = cd
                at = AT[q] + cd
                sl = so
            if at > best_at:                          # STA takes the MAX path
                best_at, best_slew = at, sl
        AT[p], SLEW[p] = best_at, best_slew

    return {
        "pins": pins,
        "net_edges": net_edges,
        "cell_edges": cell_edges,
        "gate_of_out": gate_of_out,
        "endpoints": endpoints,
        "net_load": net_load,
        "true_net_delay": true_net_delay,
        "true_cell_delay": true_cell_delay,
        "AT": AT,
        "SLEW": SLEW,
        "order": order,
    }


if __name__ == "__main__":
    c = gen_circuit(n_gates=40, n_inputs=6, seed=1)
    print(f"pins (nodes)   : {len(c['pins'])}")
    print(f"net edges      : {len(c['net_edges'])}")
    print(f"cell edges     : {len(c['cell_edges'])}")
    print(f"endpoints      : {len(c['endpoints'])}")
    ats = [c['AT'][e] for e in c['endpoints']]
    print(f"arrival time @ endpoints: min={min(ats):.3f}  max={max(ats):.3f} ns")
    print(f"critical path delay     : {max(ats):.3f} ns")
