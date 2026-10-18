"""
parse_real_design.py
======================
Parses REAL OpenROAD-generated design files into the same graph format
our model already understands.

Reads THREE files per design:
  1. <design>.synthesis_preroute.v   -> netlist structure (gates + connections)
  2. <design>.def (e.g. 20-<design>.def) -> real placement (x, y per pin/instance)
  3. <design>.sdf                    -> REAL timing labels (cell delay + net delay),
                                         computed by the paper's own OpenSTA run

This is built against the STANDARD DEF / structural-Verilog / SDF formats.
If it errors on your actual files, run inspect_files.py first and send me
the output -- these formats have tool-specific dialects and we may need to
adjust one regex.
"""
import re
import os


# ----------------------------------------------------------------------
# 1. Parse the gate-level Verilog netlist (structure: who connects to whom)
# ----------------------------------------------------------------------
def parse_verilog(path):
    text = open(path).read()
    text = re.sub(r'//.*', '', text)

    # module ports
    mod_match = re.search(r'module\s+\w+\s*\(([^)]*)\)\s*;', text)
    port_names = [p.strip() for p in mod_match.group(1).split(',')] if mod_match else []

    inputs = set()
    outputs = set()
    for m in re.finditer(r'^\s*input\s+([^;]+);', text, re.M):
        inputs.update(n.strip() for n in m.group(1).split(','))
    for m in re.finditer(r'^\s*output\s+([^;]+);', text, re.M):
        outputs.update(n.strip() for n in m.group(1).split(','))

    # instances: CELLTYPE instname ( .PORT(signal), .PORT(signal), ... );
    inst_re = re.compile(
        r'(sky130\w+)\s+(\S+)\s*\(([^;]+)\)\s*;', re.S)

    instances = []   # list of (cell_type, inst_name, {port_name: signal_name})
    for m in inst_re.finditer(text):
        cell_type, inst_name = m.group(1), m.group(2)
        conn_text = m.group(3)
        conns = {}
        for pm in re.finditer(r'\.(\w+)\s*\(\s*([\w\[\]]+)\s*\)', conn_text):
            conns[pm.group(1)] = pm.group(2)
        instances.append((cell_type, inst_name, conns))

    return {"inputs": inputs, "outputs": outputs, "instances": instances}


# ----------------------------------------------------------------------
# 2. Parse the DEF file (real placement: x, y for every instance + pin)
# ----------------------------------------------------------------------
def parse_def(path):
    text = open(path).read()

    units_m = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', text)
    dbu = int(units_m.group(1)) if units_m else 1000    # database units per micron

    # parse die area -- gives us the 4 boundary distances per pin
    die_x0, die_y0, die_x1, die_y1 = 0.0, 0.0, 1000.0, 1000.0
    die_m = re.search(r'DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', text)
    if die_m:
        die_x0 = int(die_m.group(1)) / dbu
        die_y0 = int(die_m.group(2)) / dbu
        die_x1 = int(die_m.group(3)) / dbu
        die_y1 = int(die_m.group(4)) / dbu

    inst_xy = {}   # instance name -> (x, y) in microns
    comp_block = re.search(r'COMPONENTS.*?END COMPONENTS', text, re.S)
    if comp_block:
        for m in re.finditer(
                r'-\s+(\S+)\s+(\S+)\s+\+\s+(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)',
                comp_block.group(0)):
            name, cell_type, x, y = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            if _is_power(name) or _is_power(cell_type):
                continue    # skip filler/tap/decap cells -- not real logic
            inst_xy[name] = (x / dbu, y / dbu)

    pin_xy = {}    # primary I/O pin name -> (x, y) in microns
    pin_dir = {}
    pins_block = re.search(r'\bPINS\b.*?END PINS', text, re.S)
    if pins_block:
        for m in re.finditer(
                r'-\s+(\S+)\s+\+\s+NET\s+\S+\s+\+\s+DIRECTION\s+(\w+).*?\+\s*(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)',
                pins_block.group(0), re.S):
            name, direction, x, y = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            if _is_power(name):
                continue    # skip VGND/VPWR -- not a real signal
            pin_xy[name] = (x / dbu, y / dbu)
            pin_dir[name] = direction

    return {"inst_xy": inst_xy, "pin_xy": pin_xy, "pin_dir": pin_dir,
            "die": (die_x0, die_y0, die_x1, die_y1)}


# ----------------------------------------------------------------------
# 3b. Parse SPEF for net capacitance -- key physical driver of slew
# ----------------------------------------------------------------------
def parse_spef_caps(path):
    """
    Parse a SPEF file and return net_name -> total_capacitance (pF).
    SPEF D_NET blocks look like:
      *D_NET net_name total_cap
      ...
    Total cap is what determines slew degradation on a wire.
    """
    net_caps = {}
    try:
        text = open(path, errors='replace').read()
        # *D_NET <net_name> <total_cap>
        for m in re.finditer(r'^\*D_NET\s+(\S+)\s+([\d.e+-]+)', text, re.M):
            net_name = m.group(1).strip('*')   # SPEF may prefix with *
            cap      = float(m.group(2))
            net_caps[net_name] = cap
    except Exception as e:
        print(f"  [spef] WARNING: could not parse {path}: {e}")
    return net_caps



def _mid(triplet):
    """SDF delay values in this dataset use (min::max) -- double colon,
    2 numbers -- not the (min:typ:max) triple some SDF dialects use.
    We return the average of min and max as a single representative value.
    Handles both dialects defensively."""
    triplet = triplet.strip()
    if "::" in triplet:
        parts = triplet.split("::")
        vals = [float(p) for p in parts if p.strip() != ""]
        return sum(vals) / len(vals) if vals else 0.0
    parts = [p for p in triplet.split(':') if p.strip() != ""]
    vals = [float(p) for p in parts]
    return sum(vals) / len(vals) if vals else 0.0


POWER_NAMES = {"VGND", "VPWR", "VNB", "VPB"}


def _is_power(name):
    """True if a signal/pin name refers to power/ground, e.g. 'VGND',
    'instname/VPWR', or a FILLER/TAP/DECAP cell (non-functional)."""
    bare = name.split('/')[-1]
    if bare in POWER_NAMES:
        return True
    if any(tag in name.upper() for tag in ("FILLER", "TAP", "DECAP", "PHY")):
        return True
    return False


def parse_sdf(path):
    text = open(path).read()

    cell_delay = {}   # (instance, in_pin, out_pin) -> delay (ns)
    net_delay = {}     # (driver "inst/pin" or bare pin, sink "inst/pin") -> delay (ns)

    for cell_m in re.finditer(r'\(CELL\s+(.*?)\)\s*\)\s*\)', text, re.S):
        block = cell_m.group(0)
        # INSTANCE may be BLANK, e.g. "(INSTANCE)" for the top-level design
        # block -- \s* (not \s+) so this still matches.
        inst_m = re.search(r'\(INSTANCE\s*([^\)]*)\)', block)
        inst_name = inst_m.group(1).strip() if inst_m else ""

        for io_m in re.finditer(
                r'\(IOPATH\s+(\S+)\s+(\S+)\s+\(([^()]+)\)\s+\(([^()]+)\)\)', block):
            in_pin, out_pin, rise, fall = io_m.groups()
            if _is_power(inst_name) or _is_power(in_pin) or _is_power(out_pin):
                continue
            delay = (_mid(rise) + _mid(fall)) / 2.0
            cell_delay[(inst_name, in_pin, out_pin)] = delay

        for ic_m in re.finditer(
                r'\(INTERCONNECT\s+(\S+)\s+(\S+)\s+\(([^()]+)\)\s+\(([^()]+)\)\)', block):
            src, dst, rise, fall = ic_m.groups()
            if _is_power(src) or _is_power(dst):
                continue
            delay = (_mid(rise) + _mid(fall)) / 2.0
            net_delay[(src, dst)] = delay

    return {"cell_delay": cell_delay, "net_delay": net_delay}


# ----------------------------------------------------------------------
# 4. Parse the .at.sdf file -- contains REAL per-pin AT, SLEW, and RAT
#    This is what the paper's authors extracted using their forked OpenSTADump.
#    Format: (ARRIVALTIMES ... (AT pin ...) (SLEW pin ...) (RAT pin ...) ...)
# ----------------------------------------------------------------------
def parse_at_sdf(path):
    text  = open(path, errors='replace').read()
    start = text.find('(ARRIVALTIMES')
    if start == -1:
        print(f"  [at.sdf] WARNING: no ARRIVALTIMES block found in {path}")
        return {"at": {}, "slew": {}, "rat": {}}

    block = text[start:]

    def mid(s):
        parts = [float(x) for x in s.split('::') if x.strip()]
        return sum(parts) / len(parts) if parts else 0.0

    at_data   = {}
    slew_data = {}
    rat_data  = {}

    pat = re.compile(
        r'\((AT|SLEW|RAT)\s+(\S+)\s+\(([-\d.e+:]+)\)\s+\(([-\d.e+:]+)\)\)')
    for m in pat.finditer(block):
        kind = m.group(1)
        pin  = m.group(2)
        if _is_power(pin):
            continue
        rise = mid(m.group(3))
        fall = mid(m.group(4))
        avg  = (rise + fall) / 2.0
        if kind == 'AT':
            at_data[pin]   = avg
        elif kind == 'SLEW':
            slew_data[pin] = avg
        elif kind == 'RAT':
            rat_data[pin]  = avg if abs(avg) < 1e20 else None

    print(f"  [at.sdf] parsed {len(at_data)} pins with real AT/SLEW/RAT")
    return {"at": at_data, "slew": slew_data, "rat": rat_data}


# ----------------------------------------------------------------------
# 5. Parse SDC for clock period AND per-port set_input_delay values.
#
#    Why this matters: real STA arrival time at any pin downstream of a
#    primary input includes that input's set_input_delay as a baseline
#    BEFORE any gate delay accumulates. SDF-delay propagation (the
#    fallback used when no real .at.sdf data exists) starts every input
#    at t=0 and has no way to know this constraint exists -- it silently
#    produces arrival times that are systematically low by roughly the
#    input delay amount, worse the closer a pin is to a primary input.
#    This was confirmed directly: uni_counter4b (uniform 2.0ns input
#    delay) and xtea (uniform 5.206ns input delay) both showed real AT
#    consistently higher than propagated AT by an amount matching their
#    SDC's set_input_delay value.
#
#    Returning this per-port so it can become an explicit model input
#    feature (rather than something the model has to blindly infer) is
#    the fix: gnn_model.py's AT initialization is no longer hardcoded to
#    zero for every node, so a primary input pin can now learn to start
#    its predicted AT near its real set_input_delay instead of being
#    structurally stuck at 0 (confirmed: PI pins are always edge drivers,
#    never sinks, so nothing in the propagation loop ever updates them).
# ----------------------------------------------------------------------
def parse_sdc(path):
    clock_period = None
    input_delay_map = {}   # bare port name -> input delay (ns)
    if not path:
        return clock_period, input_delay_map
    try:
        text = open(path, errors='replace').read()

        # FIXED: previous regex `create_clock[^-]*-period` stopped at the
        # FIRST dash it saw, which is `-name` in `create_clock -name clk
        # -period 10.0 ...` -- so it never reached -period at all and
        # clock_period silently stayed None for any SDC where -name (or
        # any other flag) precedes -period. `.*?` skips over any flags in
        # between instead of stopping at the first dash.
        cp_m = re.search(r'create_clock\b.*?-period\s+([\d.]+)', text)
        if cp_m:
            clock_period = float(cp_m.group(1))
            print(f"  clock period from SDC: {clock_period} ns")

        # set_input_delay <value> -clock [get_clocks {...}] [-add_delay]
        #     [get_ports {port_name}]
        # Port name may or may not be wrapped in braces, may include bus
        # notation like d_in[0] -- captured as-is to match pin_idx keys
        # built from DEF/Verilog port names later in build_graph().
        for m in re.finditer(
                r'set_input_delay\s+([\d.]+)[^\[]*\[get_ports\s*\{?\s*([^\}\]]+?)\s*\}?\s*\]',
                text):
            delay_val = float(m.group(1))
            port_name = m.group(2).strip()
            # last write wins if multiple set_input_delay lines target the
            # same port (e.g. -add_delay variants) -- acceptable for a
            # single representative baseline value
            input_delay_map[port_name] = delay_val
        if input_delay_map:
            print(f"  parsed {len(input_delay_map)} set_input_delay entries from SDC")
    except Exception as e:
        print(f"  [sdc] WARNING: could not parse {path}: {e}")
    return clock_period, input_delay_map


def build_graph(design_dir, design_name, def_filename=None):
    v_path   = os.path.join(design_dir, f"{design_name}.synthesis_preroute.v")
    def_path = def_filename or _find_def(design_dir)
    sdf_path = os.path.join(design_dir, f"{design_name}.sdf")

    # .at.sdf contains REAL per-pin AT, SLEW, and RAT -- use it if available
    at_sdf_path = os.path.join(design_dir, f"{design_name}.at.sdf")

    # also try to find SDC for real slack computation + input delays
    sdc_path = None
    for sdc_name in [f"{design_name}.cts_1.sdc", f"{design_name}.sdc"]:
        candidate = os.path.join(design_dir, sdc_name)
        if os.path.exists(candidate):
            sdc_path = candidate
            break

    print(f"  reading: {v_path}")
    net = parse_verilog(v_path)
    print(f"  reading: {def_path}")
    placement = parse_def(def_path)
    print(f"  reading: {sdf_path}")
    timing = parse_sdf(sdf_path)

    # parse SPEF for net capacitance (physical driver of slew degradation)
    spef_path = None
    for spef_name in [f"20-{design_name}.spef", f"{design_name}.spef"]:
        candidate = os.path.join(design_dir, spef_name)
        if os.path.exists(candidate):
            spef_path = candidate
            break
    net_caps = {}
    if spef_path:
        net_caps = parse_spef_caps(spef_path)
        print(f"  [spef] loaded {len(net_caps)} net capacitances from {spef_path}")
    at_timing = {}
    if os.path.exists(at_sdf_path):
        print(f"  reading: {at_sdf_path}  (real AT/SLEW/RAT labels)")
        at_timing = parse_at_sdf(at_sdf_path)
    else:
        print(f"  [at.sdf] not found -- will derive AT from SDF delays, SLEW from LUT")

    # parse SDC: clock period (existing) + per-port input delay (new)
    clock_period, input_delay_map = parse_sdc(sdc_path)

    die_x0, die_y0, die_x1, die_y1 = placement["die"]
    die_w = max(die_x1 - die_x0, 1.0)
    die_h = max(die_y1 - die_y0, 1.0)

    # ---- build pin-level nodes ----
    pins = {}
    net_edges  = []
    cell_edges = []
    signal_driver = {}
    signal_sinks  = {}

    OUTPUT_PORTS = {"Y", "X", "Q", "Z"}

    # Ports that structurally indicate a sequential element (flip-flop/latch),
    # regardless of cell-library naming convention. A cell with a CLK pin is
    # clocked: its output (Q) is NOT combinationally dependent on D within the
    # same evaluation -- Q depends on the value D held at the last clock edge.
    # Wiring D->Q as a plain cell_edge (as if it were e.g. an AND gate's A->Y)
    # fabricates an artificial combinational cycle whenever the Q net feeds
    # back, directly or indirectly, into that same D pin's fan-in cone.
    SEQUENTIAL_CLOCK_PORTS = {"CLK", "CLK_N", "GCLK"}

    for cell_type, inst_name, conns in net["instances"]:
        ix, iy = placement["inst_xy"].get(inst_name, (0.0, 0.0))
        in_keys = []
        out_key = None
        is_sequential = any(p in SEQUENTIAL_CLOCK_PORTS for p in conns.keys())
        for port, signal in conns.items():
            key = f"{inst_name}/{port}"
            pins[key] = {
                "x": ix, "y": iy,
                "kind": "OUT" if port in OUTPUT_PORTS else "IN",
                # die-boundary distances (paper Table 2, feature 3)
                "dist_left":   ix - die_x0,
                "dist_right":  die_x1 - ix,
                "dist_bottom": iy - die_y0,
                "dist_top":    die_y1 - iy,
                "die_w": die_w, "die_h": die_h,
            }
            if port in OUTPUT_PORTS:
                out_key = key
                signal_driver[signal] = key
            elif not _is_power(port):   # skip VGND/VPWR/VNB/VPB -- no timing arc
                in_keys.append(key)
                signal_sinks.setdefault(signal, []).append(key)
        if is_sequential:
            # Sequential cell: pins (D, CLK, SET/RESET, Q, ...) are all still
            # created as graph nodes above, but we deliberately create NO
            # cell_edge from any input to Q. The register acts as a level-0
            # source for downstream combinational logic, matching real timing
            # semantics (Q's arrival time is clock-relative, not derived by
            # propagating D's arrival through this cell in the same graph pass).
            continue
        for ik in in_keys:
            if out_key:
                cell_edges.append((ik, out_key, cell_type))

    for pname, (px, py) in placement["pin_xy"].items():
        key = pname
        direction = placement["pin_dir"].get(pname, "INPUT")
        pins[key] = {
            "x": px, "y": py,
            "kind": "PI" if direction == "INPUT" else "OUT",
            "dist_left":   px - die_x0,
            "dist_right":  die_x1 - px,
            "dist_bottom": py - die_y0,
            "dist_top":    die_y1 - py,
            "die_w": die_w, "die_h": die_h,
        }
        if direction == "INPUT":
            signal_driver[pname] = key
        else:
            signal_sinks.setdefault(pname, []).append(key)

    for signal, driver_key in signal_driver.items():
        for sink_key in signal_sinks.get(signal, []):
            net_edges.append((driver_key, sink_key))

    # ---- attach true delay labels ----
    pin_list = list(pins.keys())
    pin_idx  = {k: i for i, k in enumerate(pin_list)}

    true_cell_delay = {}
    for (inst, in_pin, out_pin), d in timing["cell_delay"].items():
        ik, ok = f"{inst}/{in_pin}", f"{inst}/{out_pin}"
        if ik in pin_idx and ok in pin_idx:
            true_cell_delay[(pin_idx[ik], pin_idx[ok])] = d

    true_net_delay = {}
    for (src, dst), d in timing["net_delay"].items():
        if src in pin_idx and dst in pin_idx:
            true_net_delay[(pin_idx[src], pin_idx[dst])] = d

    net_edges_idx  = [(pin_idx[d], pin_idx[s]) for d, s in net_edges
                      if d in pin_idx and s in pin_idx]
    cell_edges_idx = [(pin_idx[i], pin_idx[o]) for i, o, _ in cell_edges
                      if i in pin_idx and o in pin_idx]
    gate_of_out    = {pin_idx[o]: ct for i, o, ct in cell_edges if o in pin_idx}

    # cell_arc_info: (cell_type, in_port, out_port) per cell edge for LUT lookup
    cell_arc_info = []
    for i, o, ct in cell_edges:
        if i in pin_idx and o in pin_idx:
            in_port  = i.split("/")[-1] if "/" in i else i
            out_port = o.split("/")[-1] if "/" in o else o
            cell_arc_info.append((ct, in_port, out_port))

    pins_list = [pins[k] for k in pin_list]

    # endpoint detection: pins with no outgoing net edges AND no outgoing
    # cell edges. Previously only net-edge drivers were excluded, which
    # left every combinational gate's input pin (e.g. an AND gate's A pin,
    # which only drives a cell edge A->Y within the same instance, never a
    # net edge) misclassified as a "timing endpoint" -- confirmed on xtea:
    # 37,860/40,810 pins (92.7%) were being counted as endpoints, which is
    # not plausible for real STA endpoints (register D-pins + primary
    # outputs are a small fraction of total pins). Fixed by excluding any
    # pin that is a source of EITHER a net_edge OR a cell_edge.
    net_drivers = {d for d, _ in net_edges_idx}
    cell_drivers = {i for i, _ in cell_edges_idx}
    endpoint_set = {i for i in range(len(pins_list))
                     if i not in net_drivers and i not in cell_drivers}

    # map input_delay_map (bare port names from SDC) to pin indices.
    # Primary I/O pins use the bare port name as their pin key directly
    # (see the pin_xy loop above), so this is a direct lookup, no bare/
    # full-name fallback needed like the at.sdf mapping below.
    true_input_delay = {}   # pin_idx -> set_input_delay value (ns) from SDC
    for port_name, delay_val in input_delay_map.items():
        if port_name in pin_idx:
            true_input_delay[pin_idx[port_name]] = delay_val

    # map at_timing pin names to pin indices
    # at.sdf uses bare pin names (e.g. "p", "x[31]") for PI/PO
    # and "inst/port" style for internal pins
    true_at_slew   = {}   # pin_idx -> real slew from at.sdf
    true_at_rat    = {}   # pin_idx -> real RAT from at.sdf
    true_at_arrival= {}   # pin_idx -> real AT from at.sdf
    if at_timing:
        at_map   = at_timing.get("at",   {})
        slew_map = at_timing.get("slew", {})
        rat_map  = at_timing.get("rat",  {})
        for pin_name, idx in pin_idx.items():
            # try full name first, then bare port name
            bare = pin_name.split("/")[-1] if "/" in pin_name else pin_name
            for key in [pin_name, bare]:
                if key in at_map:
                    true_at_arrival[idx] = at_map[key]
                if key in slew_map:
                    true_at_slew[idx]    = slew_map[key]
                if key in rat_map and rat_map[key] is not None:
                    true_at_rat[idx]     = rat_map[key]

        print(f"  mapped: {len(true_at_arrival)} real AT, "
              f"{len(true_at_slew)} real SLEW, "
              f"{len(true_at_rat)} real RAT to pin indices")

    if true_input_delay:
        print(f"  mapped: {len(true_input_delay)} set_input_delay values to pin indices")

    print(f"  parsed: {len(pins_list)} pins, {len(net_edges_idx)} net edges, "
          f"{len(cell_edges_idx)} cell edges, {len(endpoint_set)} endpoints")
    print(f"  timing labels: {len(true_net_delay)} net delays, "
          f"{len(true_cell_delay)} cell delays")
    if clock_period:
        print(f"  clock period: {clock_period} ns")

    return {
        "pins":             pins_list,
        "net_edges":        net_edges_idx,
        "cell_edges":       cell_edges_idx,
        "gate_of_out":      gate_of_out,
        "cell_arc_info":    cell_arc_info,
        "true_net_delay":   true_net_delay,
        "true_cell_delay":  true_cell_delay,
        "true_at_slew":     true_at_slew,
        "true_at_arrival":  true_at_arrival,
        "true_at_rat":      true_at_rat,
        "true_input_delay": true_input_delay,  # pin_idx -> set_input_delay (ns)
        "net_caps":         net_caps,        # signal_name -> total_cap (pF) from SPEF
        "pin_names":        pin_list,
        "endpoint_set":     endpoint_set,
        "clock_period":     clock_period,
        "die":              (die_x0, die_y0, die_x1, die_y1),
    }


def _find_def(design_dir):
    for f in os.listdir(design_dir):
        if f.endswith(".def"):
            return os.path.join(design_dir, f)
    raise FileNotFoundError(f"no .def file found in {design_dir}")


if __name__ == "__main__":
    # smoke test against the mock files
    g = build_graph("mock_spm", "spm")
    print("\nSample pins:", g["pin_names"][:5])
    print("Sample net delay entries:", list(g["true_net_delay"].items())[:3])
    print("Sample cell delay entries:", list(g["true_cell_delay"].items())[:3])