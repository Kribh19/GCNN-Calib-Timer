"""
preprocess.py
=============
Parses all 22 designs and saves .pt tensor files.

*** FIXES APPLIED (this version) ***

  Bug #1 — cd_true (cell delay labels) were being assigned by raw list
  position from parse_sdf()'s flat `cd` list, with no guarantee that
  list-position ei in cell_edges corresponded to the ei-th IOPATH entry
  encountered in the SDF file. Verified against a real spm.sdf sample:
  each SDF CELL block explicitly carries (INSTANCE name), which is
  exactly the identity information needed to match correctly — the same
  pattern already used (correctly) for nd_true (net delay).

  Fix: parse_sdf() now returns cell_delays as a dict keyed by
  (in_key, out_key) = ("instance/in_pin", "instance/out_pin"), built by
  scanning each (CELL ... (INSTANCE ...) ... (IOPATH ...)) block with a
  paren-depth scanner (so blocks are never misaligned). build_design()
  now matches cd_true by this identity key, not by position.
  STATUS: applied AND verified — re-run on all 27 designs, cd_hits now
  tracks lut_hits almost 1:1 across every design (was previously 2.4x
  mismatched for spm: 636 vs 262). Confirmed working.

  Bug #2 — Liberty LUT arc match rate was only ~20-40% across all 27
  designs. Root cause found via diagnose_lut_match.py (read-only
  diagnostic, run against real spm data): power/ground pins (VPWR,
  VGND, VPB, VNB) were being treated as real timing-arc inputs, e.g.
  arc_key="VGND→X" was being looked up in Liberty and counted as a
  "miss" even though power/ground pins never have timing arcs in any
  Liberty file — Liberty correctly has no cell_rise/cell_fall for them.
  Confirmed by exact arithmetic: sky130_fd_sc_hd__clkbuf_1 has 1 real
  signal input (A) + 4 power/ground pins = 5 "in_ports" per instance;
  445 total attempts / 5 = 89 instances; 89 hits (all from A→X) exactly
  matches instance count. Same exact-match pattern held for and2_1 and
  a22o_1. This was not a naming-format bug — it was counting arcs that
  structurally could never exist.

  Fix: POWER_GROUND_PORTS = {'VPWR','VGND','VPB','VNB'} added; these
  ports are now excluded from in_keys / cell_edges construction in
  build_design(), so they never generate a fake cell_edges timing arc.
  Node entries for these pins are left in `pins` dict untouched (does
  not affect indexing elsewhere) — only cell_edges construction changed.
  STATUS: applied, NOT yet re-run. Re-run on spm first to verify the
  match rate improves before running all 27 designs again.

  Everything else in this file is UNCHANGED from the original version —
  per project rule "do not touch working code", only parse_sdf(), the
  cd_true assignment block, and the in_keys/cell_edges construction loop
  in build_design() were modified.

Node features (18):
  0-1:   x,y norm (DEF)
  2-4:   is_PI, is_IN, is_OUT (Verilog)
  5-8:   dist to 4 die boundaries (DEF)
  9:     is_endpoint
  10:    fanout_norm
  11:    net_cap_norm (SPEF)
  12:    logic_depth_norm
  13:    net_res_norm (SPEF)
  14-15: pin_cap_rise_ff, pin_cap_fall_ff (Liberty ff)
  16-17: pin_cap_rise_ss, pin_cap_fall_ss (Liberty ss)

Cell edge LUT: [E, 8, 7, 7]
  Tables 0-3: ff corner cell_rise, cell_fall, rise_transition, fall_transition
  Tables 4-7: ss corner same 4

Liberty file format (sky130hd):
  cell ("sky130_fd_sc_hd__and2_1") {
      pin ("A") {
          capacitance : 0.0024;
          fall_capacitance : 0.0023;
          rise_capacitance : 0.0025;
          timing () {
              related_pin : "A1";
              cell_rise ("del_1_7_7") {
                  index_1("...7 values...");
                  index_2("...7 values...");
                  values("v00,...,v06", \
                         "v10,...,v16", ...7 rows...);
              }
          }
      }
  }

SDF format:
  (INTERCONNECT src_inst/port dst_inst/port (min::max))
  (CELL
   (CELLTYPE "sky130_fd_sc_hd__inv_2")
   (INSTANCE _182_)
   (DELAY
    (ABSOLUTE
     (IOPATH A Y (rise_min::rise_max) (fall_min::fall_max))
    )
   )
  )
  -- verified from real spm.sdf, July 2026 debugging session
"""

import os, re, json, argparse, time, collections
import numpy as np
import torch

# ── PATHS ────────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Krish\GCNN\data"
TECHLIB_DIR = r"C:\Krish\GCNN\data\techlib"
OUT_DIR     = r"C:\Krish\GCNN\preprocessed"

LIB_FF = os.path.join(TECHLIB_DIR, "sky130_fd_sc_hd__ff_n40C_1v95.lib")
LIB_SS = os.path.join(TECHLIB_DIR, "sky130_fd_sc_hd__ss_100C_1v60.lib")

NODE_FEATURES  = 18
NET_EDGE_FEATS = 4
CAP_SCALE      = 1.0      # pF
RES_SCALE      = 1000.0   # Ω → kΩ
CANVAS_SCALE   = 1000.0   # µm
NON_DESIGNS    = {'techlib', 'data', 'preprocessed', 'src', 'results'}

# ── LIBERTY PARSER ───────────────────────────────────────────────────────────
# UNCHANGED from prior version — verified against real .lib text (inv_2 cell,
# July 2026): cell_rise("del_1_7_7"){...} and related_pin:"A"; format matches
# this parser's regex correctly. Not modified.

def _read_table_values(text, start):
    """
    Given text and start position of 'values(' read all the
    comma/backslash-continued numbers until the closing );
    Returns list of floats.
    """
    pos = text.index('(', start)
    depth = 1
    pos += 1
    buf = []
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        pos += 1
    raw = ''.join(buf)
    nums = re.findall(r'-?[\d]+\.[\d]+(?:[eE][+\-]?\d+)?', raw)
    return [float(n) for n in nums]


def _find_block_end(text, open_brace_pos):
    """Find position after matching closing brace (for { } blocks)."""
    depth = 1
    i = open_brace_pos + 1
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return i


def parse_liberty(path):
    """
    Returns:
      luts[cell][arc_key] = {cell_rise, cell_fall, rise_transition, fall_transition}
        where arc_key = "related_pin→output_pin"  e.g. "A→Y"
      caps[cell][pin_name] = {'rise': float_pF, 'fall': float_pF}
    """
    print(f"    parsing {os.path.basename(path)} ...", end=' ', flush=True)
    text = open(path, errors='replace').read()

    luts = {}
    caps = {}

    for cm in re.finditer(r'\bcell\s*\(\s*"([^"]+)"\s*\)\s*\{', text):
        cname = cm.group(1)
        cstart = cm.end() - 1
        cend = _find_block_end(text, cstart)
        cell_body = text[cstart+1:cend-1]

        luts[cname] = {}
        caps[cname] = {}

        for pm in re.finditer(r'\bpin\s*\(\s*"([^"]+)"\s*\)\s*\{', cell_body):
            pname = pm.group(1)
            pstart = pm.end() - 1
            pend = _find_block_end(cell_body, pstart)
            pin_body = cell_body[pstart+1:pend-1]

            cap_base = 0.0
            m = re.search(r'\bcapacitance\s*:\s*([\d.eE+\-]+)', pin_body)
            if m:
                cap_base = float(m.group(1))
            rise_cap = cap_base
            fall_cap = cap_base
            m = re.search(r'\brise_capacitance\s*:\s*([\d.eE+\-]+)', pin_body)
            if m:
                rise_cap = float(m.group(1))
            m = re.search(r'\bfall_capacitance\s*:\s*([\d.eE+\-]+)', pin_body)
            if m:
                fall_cap = float(m.group(1))
            if cap_base > 0 or rise_cap > 0 or fall_cap > 0:
                caps[cname][pname] = {'rise': rise_cap, 'fall': fall_cap}

            for tm in re.finditer(r'\btiming\s*\(\s*\)\s*\{', pin_body):
                tstart = tm.end() - 1
                tend = _find_block_end(pin_body, tstart)
                timing_body = pin_body[tstart+1:tend-1]

                rm = re.search(r'\brelated_pin\s*:\s*"([^"]+)"', timing_body)
                if not rm:
                    continue
                rel_pin = rm.group(1)
                arc_key = f"{rel_pin}→{pname}"

                def _get_table(ttype, tbody):
                    m2 = re.search(
                        rf'\b{ttype}\s*\(\s*"\w+"\s*\)\s*\{{', tbody)
                    if not m2:
                        return None
                    tb_open = m2.end() - 1
                    tb_end = _find_block_end(tbody, tb_open)
                    tblock = tbody[tb_open+1:tb_end-1]
                    vm = re.search(r'\bvalues\s*\(', tblock)
                    if not vm:
                        return None
                    nums = _read_table_values(tblock, vm.start())
                    if len(nums) < 49:
                        return None
                    return np.array(nums[:49], dtype=np.float32).reshape(7, 7)

                ZERO = np.zeros((7, 7), np.float32)
                cr = _get_table('cell_rise',       timing_body)
                cf = _get_table('cell_fall',       timing_body)
                rt = _get_table('rise_transition', timing_body)
                ft = _get_table('fall_transition', timing_body)

                if any(t is not None for t in [cr, cf, rt, ft]):
                    luts[cname][arc_key] = {
                        'cell_rise':       cr if cr is not None else ZERO,
                        'cell_fall':       cf if cf is not None else ZERO,
                        'rise_transition': rt if rt is not None else ZERO,
                        'fall_transition': ft if ft is not None else ZERO,
                    }

    n_arcs = sum(len(v) for v in luts.values())
    print(f"{len(luts)} cells, {n_arcs} arcs")
    return luts, caps


# ── SDF PARSER ───────────────────────────────────────────────────────────────
# *** MODIFIED — Bug #1 fix ***
# cell_delays is now a dict keyed by (in_key, out_key) = "instance/pin" pairs,
# matched by real identity from (INSTANCE ...) inside each CELL block —
# instead of a flat positional list. Net-delay (INTERCONNECT) parsing is
# unchanged; it was already matched by identity and verified working.

def _find_paren_block_end(text, open_paren_pos):
    """text[open_paren_pos] must be '('. Returns index just after the matching ')'."""
    depth = 0
    i = open_paren_pos
    n = len(text)
    while i < n:
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def parse_sdf(path):
    """
    Returns:
      net_delays  {(src, dst): float_ns}   src/dst = "inst/port"  (unchanged)
      cell_delays {(in_key, out_key): float_ns}
                  in_key/out_key = "instance/pin", matched by identity
                  from (CELL (CELLTYPE ...)(INSTANCE ...)(DELAY(ABSOLUTE(
                  IOPATH in out ...)))) blocks — verified structure from
                  real spm.sdf.

    SDF INTERCONNECT: (INTERCONNECT src dst (min::max))
    SDF CELL block:   one block per instance; may contain multiple IOPATH
                       entries (one per timing arc of that instance).
    """
    text = open(path, errors='replace').read()
    nd = {}
    cd = {}

    # INTERCONNECT — unchanged, already matched by identity, verified working
    for m in re.finditer(
        r'\(INTERCONNECT\s+(\S+)\s+(\S+)\s+'
        r'\(([\d.]+)::([\d.]+)\)\s+'
        r'\(([\d.]+)::([\d.]+)\)\s*\)', text):
        src, dst = m.group(1), m.group(2)
        rise = (float(m.group(3)) + float(m.group(4))) / 2.0
        fall = (float(m.group(5)) + float(m.group(6))) / 2.0
        nd[(src, dst)] = (rise + fall) / 2.0

    # CELL blocks — one block per instance, paren-depth scanned so IOPATH
    # entries are never misaligned across instance boundaries (this
    # misalignment was Bug #1: cd_true assigned by raw list position).
    for m in re.finditer(r'\(CELL\b', text):
        start = m.start()
        end   = _find_paren_block_end(text, start)
        block = text[start:end]

        im = re.search(r'\(INSTANCE\s+(\S*?)\)', block)
        if not im or not im.group(1):
            # Top-level design-wrapper CELL block has "(INSTANCE)" empty —
            # this is the whole-design CELL at the top of the file, not a
            # real gate instance. Skip it.
            continue

        instance = im.group(1)

        for iop in re.finditer(
            r'\(IOPATH\s+(\S+)\s+(\S+)\s+'
            r'\(([\d.]+)::([\d.]+)\)\s+'
            r'\(([\d.]+)::([\d.]+)\)\s*\)', block):
            in_pin, out_pin = iop.group(1), iop.group(2)
            rise = (float(iop.group(3)) + float(iop.group(4))) / 2.0
            fall = (float(iop.group(5)) + float(iop.group(6))) / 2.0
            val  = (rise + fall) / 2.0
            in_key  = f"{instance}/{in_pin}"
            out_key = f"{instance}/{out_pin}"
            cd[(in_key, out_key)] = val

    return nd, cd


# ── at.sdf PARSER ────────────────────────────────────────────────────────────
# UNCHANGED

def parse_at_sdf(path):
    """Returns {pin: float} for AT, SLEW, RAT."""
    text = open(path, errors='replace').read()
    at, slew, rat = {}, {}, {}
    pat = re.compile(
        r'\((AT|SLEW|RAT)\s+(\S+)\s+'
        r'\(([-\d.eE+:]+)\)\s+\(([-\d.eE+:]+)\)\)')
    for m in pat.finditer(text):
        kind, pin = m.group(1), m.group(2)
        def _mid(s):
            parts = [float(x) for x in s.split('::') if x.strip()]
            return sum(parts) / len(parts) if parts else 0.0
        val = (_mid(m.group(3)) + _mid(m.group(4))) / 2.0
        if kind == 'AT':
            at[pin] = val
        elif kind == 'SLEW':
            slew[pin] = val
        elif kind == 'RAT' and abs(val) < 1e19:
            rat[pin] = val
    return at, slew, rat


# ── DEF PARSER ───────────────────────────────────────────────────────────────
# UNCHANGED

def parse_def(path):
    text = open(path, errors='replace').read()
    um = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', text)
    dbu = int(um.group(1)) if um else 1000
    die = (0.0, 0.0, 1000.0, 1000.0)
    dm = re.search(
        r'DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', text)
    if dm:
        die = (int(dm.group(1))/dbu, int(dm.group(2))/dbu,
               int(dm.group(3))/dbu, int(dm.group(4))/dbu)
    inst_xy = {}
    cb = re.search(r'COMPONENTS\s+\d+\s*;(.*?)END\s+COMPONENTS', text, re.S)
    if cb:
        for m in re.finditer(
            r'-\s+(\S+)\s+\S+\s+.*?PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)',
                cb.group(1), re.S):
            inst_xy[m.group(1)] = (int(m.group(2))/dbu, int(m.group(3))/dbu)
    pi_xy = {}
    pb = re.search(r'PINS\s+\d+\s*;(.*?)END\s+PINS', text, re.S)
    if pb:
        for m in re.finditer(
            r'-\s+(\S+)\s+.*?PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)',
                pb.group(1), re.S):
            pi_xy[m.group(1)] = (int(m.group(2))/dbu, int(m.group(3))/dbu)
    return inst_xy, pi_xy, die


# ── VERILOG PARSER ───────────────────────────────────────────────────────────
# UNCHANGED

def parse_verilog(path):
    text = open(path, errors='replace').read()
    text = re.sub(r'//[^\n]*', '', text)
    inputs, outputs = set(), set()
    for m in re.finditer(r'^\s*input\s+([^;]+);', text, re.M):
        inputs.update(x.strip() for x in m.group(1).split(','))
    for m in re.finditer(r'^\s*output\s+([^;]+);', text, re.M):
        outputs.update(x.strip() for x in m.group(1).split(','))
    instances = []
    for m in re.finditer(r'(sky130\w+)\s+(\S+)\s*\(([^;]+)\)\s*;', text, re.S):
        ct, inst = m.group(1), m.group(2)
        conns = {}
        for pm in re.finditer(r'\.(\w+)\s*\(\s*([\w\[\]]+)\s*\)', m.group(3)):
            conns[pm.group(1)] = pm.group(2)
        instances.append((ct, inst, conns))
    return inputs, outputs, instances


# ── SPEF PARSER ──────────────────────────────────────────────────────────────
# UNCHANGED

def parse_spef(path):
    text = open(path, errors='replace').read()
    net_caps, net_res = {}, {}
    for sec in re.split(r'\*D_NET\s+', text)[1:]:
        nm = re.match(r'(\S+)', sec)
        if not nm:
            continue
        net = nm.group(1).strip('"\\')
        cm = re.search(r'\*C\s+([\d.eE+\-]+)', sec)
        if cm:
            net_caps[net] = float(cm.group(1))
        total_r = sum(float(r) for r in
                      re.findall(r'\*RES\s+\d+\s+\S+\s+\S+\s+([\d.eE+\-]+)', sec))
        if total_r > 0:
            net_res[net] = total_r
    return net_caps, net_res


# ── SDC PARSER ───────────────────────────────────────────────────────────────
# UNCHANGED

def parse_sdc(path):
    try:
        text = open(path, errors='replace').read()
        m = re.search(r'create_clock[^-]*-period\s+([\d.]+)', text)
        return float(m.group(1)) if m else None
    except:
        return None


# ── BUILD ONE DESIGN ─────────────────────────────────────────────────────────
# *** MODIFIED — Bug #1 fix in cd_true assignment block only ***
# Everything else in build_design() is UNCHANGED.

OUTPUT_PORTS = {'Y', 'X', 'Q', 'Z', 'Q_N', 'COUT'}

# Bug #2 fix: power/ground pins are never real timing-arc inputs.
# Liberty correctly has no cell_rise/cell_fall entries for arcs like
# "VGND→X" — these are power rails, not logic signals. Treating them
# as candidate arc inputs was silently generating ~78% fake "misses"
# in the Liberty LUT match rate (confirmed via diagnose_lut_match.py
# against real spm data — see module docstring for the exact arithmetic
# that confirmed this).
POWER_GROUND_PORTS = {'VPWR', 'VGND', 'VPB', 'VNB'}


def build_design(design_dir, design_name,
                 luts_ff, caps_ff, luts_ss, caps_ss,
                 gate_vocab):
    def _find(*names):
        for n in names:
            p = os.path.join(design_dir, n)
            if os.path.exists(p):
                return p
        return None

    v_path    = _find(f"{design_name}.synthesis_preroute.v")
    def_path  = _find(f"20-{design_name}.def", f"{design_name}.def")
    sdf_path  = _find(f"{design_name}.sdf")
    at_path   = _find(f"{design_name}.at.sdf")
    spef_path = _find(f"20-{design_name}.spef", f"{design_name}.spef")
    sdc_path  = _find(f"{design_name}.cts_1.sdc", f"{design_name}.sdc")

    assert v_path,   f"Missing verilog: {design_name}"
    assert def_path, f"Missing DEF: {design_name}"
    assert sdf_path, f"Missing SDF: {design_name}"

    inputs, outputs, instances = parse_verilog(v_path)
    inst_xy, pi_xy, die        = parse_def(def_path)
    sdf_nd, sdf_cd             = parse_sdf(sdf_path)   # sdf_cd now a dict, not a list
    at_map, slew_map, rat_map  = parse_at_sdf(at_path) if at_path else ({}, {}, {})
    net_caps, net_res          = parse_spef(spef_path) if spef_path else ({}, {})
    clock_period               = parse_sdc(sdc_path) if sdc_path else None

    die_x0, die_y0, die_x1, die_y1 = die
    die_w = max(die_x1 - die_x0, 1.0)
    die_h = max(die_y1 - die_y0, 1.0)

    # ── build pin graph ───────────────────────────────────────────────────────
    pins          = {}
    net_edges     = []
    cell_edges    = []
    signal_driver = {}
    signal_sinks  = collections.defaultdict(list)

    for sig in inputs:
        x, y = pi_xy.get(sig, (die_x0, die_y0))
        pins[sig] = {'x': x, 'y': y, 'kind': 'PI',
                     'sig': sig, 'ct': None, 'port': None}
        signal_driver[sig] = sig

    for ct, inst, conns in instances:
        ix, iy = inst_xy.get(inst, (0.0, 0.0))
        out_key = None
        in_keys = []
        for port, sig in conns.items():
            key    = f"{inst}/{port}"
            is_out = port in OUTPUT_PORTS
            pins[key] = {'x': ix, 'y': iy,
                         'kind': 'OUT' if is_out else 'IN',
                         'sig': sig, 'ct': ct, 'port': port}
            if is_out:
                out_key = key
                signal_driver[sig] = key
            elif port in POWER_GROUND_PORTS:
                # Bug #2 fix: power/ground pins (VPWR/VGND/VPB/VNB) are
                # never real timing-arc inputs — skip them so they don't
                # generate a fake cell_edges entry that can never match
                # a Liberty LUT. The pin node itself stays in `pins`
                # (harmless, doesn't affect indexing elsewhere); only
                # cell_edges construction is affected.
                continue
            else:
                in_keys.append((key, port, sig))
                signal_sinks[sig].append(key)
        if out_key:
            for (ik, rel_pin, _) in in_keys:
                cell_edges.append(
                    (ik, out_key, ct, rel_pin, pins[out_key]['port']))

    for sig in outputs:
        x, y = pi_xy.get(sig, (die_x1, die_y1))
        pins[sig] = {'x': x, 'y': y, 'kind': 'PO',
                     'sig': sig, 'ct': None, 'port': None}
        if sig in signal_driver:
            signal_sinks[sig].append(sig)

    for sig, drv in signal_driver.items():
        for snk in signal_sinks[sig]:
            if drv != snk:
                net_edges.append((drv, snk, sig))

    pin_keys = list(pins.keys())
    pin_idx  = {k: i for i, k in enumerate(pin_keys)}
    N        = len(pin_keys)

    # ── topological sort ──────────────────────────────────────────────────────
    in_deg = collections.defaultdict(int)
    adj    = collections.defaultdict(list)
    for s, d, _ in net_edges:
        si, di = pin_idx[s], pin_idx[d]
        adj[si].append(di)
        in_deg[di] += 1
    for ik, ok, *_ in cell_edges:
        ii, oi = pin_idx[ik], pin_idx[ok]
        adj[ii].append(oi)
        in_deg[oi] += 1

    queue = collections.deque(i for i in range(N) if in_deg[i] == 0)
    level = [0] * N
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            level[v] = max(level[v], level[u] + 1)
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    max_level = max(level) if level else 0

    net_src_list, net_dst_list, net_sig_list = [], [], []
    for s, d, sig in net_edges:
        net_src_list.append(pin_idx[s])
        net_dst_list.append(pin_idx[d])
        net_sig_list.append(sig)

    cell_src_list, cell_dst_list = [], []
    cell_type_list, cell_rel_list, cell_out_list = [], [], []
    for ik, ok, ct, rp, op in cell_edges:
        cell_src_list.append(pin_idx[ik])
        cell_dst_list.append(pin_idx[ok])
        cell_type_list.append(ct)
        cell_rel_list.append(rp)
        cell_out_list.append(op)

    E_net  = len(net_src_list)
    E_cell = len(cell_src_list)

    net_by_level  = collections.defaultdict(list)
    cell_by_level = collections.defaultdict(list)
    for ei, (s, _) in enumerate(zip(net_src_list, net_dst_list)):
        net_by_level[level[s]].append(ei)
    for ei, (s, _) in enumerate(zip(cell_src_list, cell_dst_list)):
        cell_by_level[level[s]].append(ei)
    levels = sorted(set(list(net_by_level.keys()) +
                        list(cell_by_level.keys())))

    # endpoints
    ep_set = set()
    for pname in at_map:
        if pname in pin_idx:
            ep_set.add(pin_idx[pname])
        parts = pname.split('/')
        if len(parts) >= 2:
            short = '/'.join(parts[-2:])
            if short in pin_idx:
                ep_set.add(pin_idx[short])
    for k, p in pins.items():
        if p['kind'] == 'PO':
            ep_set.add(pin_idx[k])
    endpoints = torch.tensor(sorted(ep_set), dtype=torch.long)

    # SPEF lookup helpers
    def _cap(sig):
        for k in [sig, f"\\{sig}"]:
            if k in net_caps:
                return net_caps[k]
        return 0.0

    def _res(sig):
        for k in [sig, f"\\{sig}"]:
            if k in net_res:
                return net_res[k]
        return 0.0

    fanout_of = {sig: len(sinks) for sig, sinks in signal_sinks.items()}

    # ── node features [N, 18] ─────────────────────────────────────────────────
    x_feat = torch.zeros(N, NODE_FEATURES, dtype=torch.float32)
    for i, key in enumerate(pin_keys):
        p  = pins[key]
        px, py = p['x'], p['y']
        sig = p['sig'] or ''
        ct  = p['ct']  or ''
        port= p['port'] or ''
        kind= p['kind']

        x_feat[i, 0] = px / CANVAS_SCALE
        x_feat[i, 1] = py / CANVAS_SCALE
        x_feat[i, 2] = 1.0 if kind == 'PI' else 0.0
        x_feat[i, 3] = 1.0 if kind == 'IN' else 0.0
        x_feat[i, 4] = 1.0 if kind == 'OUT' else 0.0
        x_feat[i, 5] = (px - die_x0) / die_w
        x_feat[i, 6] = (die_x1 - px) / die_w
        x_feat[i, 7] = (py - die_y0) / die_h
        x_feat[i, 8] = (die_y1 - py) / die_h
        x_feat[i, 9] = 1.0 if i in ep_set else 0.0
        x_feat[i,10] = min(fanout_of.get(sig, 0), 50) / 50.0
        x_feat[i,11] = min(_cap(sig), 10.0) / CAP_SCALE
        x_feat[i,12] = level[i] / max(max_level, 1)
        x_feat[i,13] = min(_res(sig), 10000.0) / RES_SCALE
        if ct and port:
            cf = caps_ff.get(ct, {}).get(port, {})
            cs = caps_ss.get(ct, {}).get(port, {})
            x_feat[i,14] = cf.get('rise', 0.0)
            x_feat[i,15] = cf.get('fall', 0.0)
            x_feat[i,16] = cs.get('rise', 0.0)
            x_feat[i,17] = cs.get('fall', 0.0)

    # ── net edge features [E_net, 4] ──────────────────────────────────────────
    edge_geo = torch.zeros(E_net, NET_EDGE_FEATS, dtype=torch.float32)
    for ei, (s, d, sig) in enumerate(
            zip(net_src_list, net_dst_list, net_sig_list)):
        px_s = pins[pin_keys[s]]['x']
        py_s = pins[pin_keys[s]]['y']
        px_d = pins[pin_keys[d]]['x']
        py_d = pins[pin_keys[d]]['y']
        dx  = (px_d - px_s) / CANVAS_SCALE
        dy  = (py_d - py_s) / CANVAS_SCALE
        man = (abs(px_d - px_s) + abs(py_d - py_s)) / CANVAS_SCALE
        fo  = min(fanout_of.get(sig, 1), 50) / 50.0
        edge_geo[ei] = torch.tensor([dx, dy, man, fo])

    # ── cell edge LUT [E_cell, 8, 7, 7] ──────────────────────────────────────
    ZERO77     = np.zeros((7, 7), np.float32)
    TABLE_KEYS = ['cell_rise', 'cell_fall', 'rise_transition', 'fall_transition']

    lut_delay     = torch.zeros(E_cell, 8, 7, 7, dtype=torch.float32)
    lut_slew      = torch.zeros(E_cell, 8, 7, 7, dtype=torch.float32)
    gate_oh_list  = []
    cell_lut_mask = []
    n_types = len(gate_vocab)
    unk_idx = gate_vocab.get('UNKNOWN', 0)

    lut_hits = 0
    for ei, (ik, ok, ct, rel_pin, out_port) in enumerate(
            zip(cell_src_list, cell_dst_list,
                cell_type_list, cell_rel_list, cell_out_list)):
        oh = [0.0] * n_types
        oh[gate_vocab.get(ct, unk_idx)] = 1.0
        gate_oh_list.append(oh)

        arc_key = f"{rel_pin}→{out_port}"
        ff_t = luts_ff.get(ct, {}).get(arc_key, {})
        ss_t = luts_ss.get(ct, {}).get(arc_key, {})

        for ti, tk in enumerate(TABLE_KEYS):
            arr_ff = ff_t.get(tk, ZERO77)
            arr_ss = ss_t.get(tk, ZERO77)
            lut_delay[ei, ti]     = torch.from_numpy(arr_ff)
            lut_delay[ei, 4 + ti] = torch.from_numpy(arr_ss)
            lut_slew[ei, ti]      = torch.from_numpy(arr_ff)
            lut_slew[ei, 4 + ti]  = torch.from_numpy(arr_ss)

        has = bool(ff_t or ss_t)
        cell_lut_mask.append(has)
        if has:
            lut_hits += 1

    gate_onehot  = torch.tensor(gate_oh_list, dtype=torch.float32)
    cell_lut_mask_t = torch.tensor(cell_lut_mask, dtype=torch.bool)

    # ── labels ────────────────────────────────────────────────────────────────
    AT_true        = torch.zeros(N, dtype=torch.float32)
    SLEW_true      = torch.zeros(N, dtype=torch.float32)
    slack_true     = torch.zeros(N, dtype=torch.float32)
    slew_real_mask = torch.zeros(N, dtype=torch.bool)

    def _match(pname):
        if pname in pin_idx:
            return pin_idx[pname]
        parts = pname.split('/')
        if len(parts) >= 2:
            short = '/'.join(parts[-2:])
            if short in pin_idx:
                return pin_idx[short]
        return None

    at_found = 0
    for pname, val in at_map.items():
        idx = _match(pname)
        if idx is not None:
            AT_true[idx] = float(val)
            at_found += 1

    slew_found = 0
    for pname, val in slew_map.items():
        idx = _match(pname)
        if idx is not None and val > 0:
            SLEW_true[idx] = float(val)
            slew_real_mask[idx] = True
            slew_found += 1

    slack_type = 'proxy'
    if rat_map:
        rat_found = 0
        for pname, val in rat_map.items():
            if val is None:
                continue
            idx = _match(pname)
            if idx is not None:
                slack_true[idx] = float(val) - AT_true[idx]
                rat_found += 1
        if rat_found > 0:
            slack_type = 'real_rat'
    elif clock_period is not None:
        for i in sorted(ep_set):
            slack_true[i] = clock_period - AT_true[i]
        slack_type = 'real'

    # net delay labels — SDF: (INTERCONNECT src dst (min::max))  — UNCHANGED,
    # already matched by identity
    nd_true          = torch.zeros(E_net, dtype=torch.float32)
    net_trusted_mask = torch.zeros(E_net, dtype=torch.bool)
    nd_hits = 0
    for ei, (s, d, _) in enumerate(
            zip(net_src_list, net_dst_list, net_sig_list)):
        sk = pin_keys[s]
        dk = pin_keys[d]
        if (sk, dk) in sdf_nd:
            nd_true[ei] = sdf_nd[(sk, dk)]
            net_trusted_mask[ei] = True
            nd_hits += 1
        elif (dk, sk) in sdf_nd:
            nd_true[ei] = sdf_nd[(dk, sk)]
            net_trusted_mask[ei] = True
            nd_hits += 1

    # *** MODIFIED — Bug #1 fix ***
    # cell delay labels — matched by (in_key, out_key) identity, same
    # pattern as nd_true above. sdf_cd is now a dict, not a positional list.
    cd_true = torch.zeros(E_cell, dtype=torch.float32)
    cd_hits = 0
    for ei, (ik, ok) in enumerate(zip(cell_src_list, cell_dst_list)):
        key = (pin_keys[ik], pin_keys[ok])
        if key in sdf_cd:
            cd_true[ei] = sdf_cd[key]
            cd_hits += 1

    print(f"      N={N} net_edges={E_net} cell_edges={E_cell}")
    print(f"      AT={at_found} SLEW={slew_found} "
          f"nd_mask={nd_hits} lut_hits={lut_hits}/{E_cell} "
          f"cd_hits={cd_hits}/{E_cell} slack={slack_type}")

    return {
        'n_nodes':       N,
        'n_net_edges':   E_net,
        'n_cell_edges':  E_cell,
        'net_src':       torch.tensor(net_src_list,  dtype=torch.long),
        'net_dst':       torch.tensor(net_dst_list,  dtype=torch.long),
        'cell_src':      torch.tensor(cell_src_list, dtype=torch.long),
        'cell_dst':      torch.tensor(cell_dst_list, dtype=torch.long),
        'x':             x_feat,
        'edge_geo':      edge_geo,
        'gate_onehot':   gate_onehot,
        'lut_delay':     lut_delay,
        'lut_slew':      lut_slew,
        'cell_lut_mask': cell_lut_mask_t,
        'AT_true':       AT_true,
        'SLEW_true':     SLEW_true,
        'nd_true':       nd_true,
        'cd_true':       cd_true,
        'slack_true':    slack_true,
        'slack_type':    slack_type,
        'slew_real_mask':slew_real_mask,
        'net_trusted_mask': net_trusted_mask,
        'endpoints':     endpoints,
        'levels':        levels,
        'net_by_level':  dict(net_by_level),
        'cell_by_level': dict(cell_by_level),
        'cell_gtype_str':cell_type_list,
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────
# UNCHANGED

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',    default=DATA_DIR)
    ap.add_argument('--techlib', default=TECHLIB_DIR)
    ap.add_argument('--out',     default=OUT_DIR)
    ap.add_argument('--designs', nargs='*', default=None)
    args = ap.parse_args()

    lib_ff = os.path.join(args.techlib, 'sky130_fd_sc_hd__ff_n40C_1v95.lib')
    lib_ss = os.path.join(args.techlib, 'sky130_fd_sc_hd__ss_100C_1v60.lib')
    assert os.path.exists(lib_ff), f"FF liberty not found: {lib_ff}"
    assert os.path.exists(lib_ss), f"SS liberty not found: {lib_ss}"

    print("Loading Liberty files ...")
    luts_ff, caps_ff = parse_liberty(lib_ff)
    luts_ss, caps_ss = parse_liberty(lib_ss)

    os.makedirs(args.out, exist_ok=True)

    if args.designs:
        designs = args.designs
    else:
        designs = sorted(
            d for d in os.listdir(args.data)
            if os.path.isdir(os.path.join(args.data, d))
            and d not in NON_DESIGNS)
    print(f"\nDesigns to process ({len(designs)}): {designs}\n")

    print("Building gate vocabulary ...")
    all_types = {'UNKNOWN'}
    for dn in designs:
        vp = os.path.join(args.data, dn, f"{dn}.synthesis_preroute.v")
        if not os.path.exists(vp):
            continue
        text = open(vp, errors='replace').read()
        for m in re.finditer(r'(sky130\w+)\s+\S+\s*\(', text):
            all_types.add(m.group(1))
    gate_vocab = {g: i for i, g in enumerate(sorted(all_types))}
    print(f"  {len(gate_vocab)} gate types\n")

    meta = {
        'gate_vocab':    gate_vocab,
        'n_gates':       len(gate_vocab),
        'NODE_FEATURES': NODE_FEATURES,
        'canvas_scale':  CANVAS_SCALE,
        'lib_ff':        os.path.basename(lib_ff),
        'lib_ss':        os.path.basename(lib_ss),
        'lut_shape':     'E x 8 x 7 x 7',
        'node_feature_names': [
            'x_norm', 'y_norm', 'is_PI', 'is_IN', 'is_OUT',
            'dist_left', 'dist_right', 'dist_bottom', 'dist_top',
            'is_endpoint', 'fanout_norm', 'net_cap_norm',
            'logic_depth_norm', 'net_res_norm',
            'pin_cap_rise_ff', 'pin_cap_fall_ff',
            'pin_cap_rise_ss', 'pin_cap_fall_ss',
        ],
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print("meta.json saved\n")

    ok, failed = [], []
    for dn in designs:
        design_dir = os.path.join(args.data, dn)
        if not os.path.isdir(design_dir):
            print(f"  SKIP {dn} — not a directory")
            continue
        print(f"  [{dn}]")
        t0 = time.time()
        try:
            g = build_design(design_dir, dn,
                             luts_ff, caps_ff, luts_ss, caps_ss,
                             gate_vocab)
            out_path = os.path.join(args.out, f"{dn}.pt")
            torch.save(g, out_path)
            print(f"      saved {out_path}  ({time.time()-t0:.1f}s)")
            ok.append(dn)
        except Exception as e:
            import traceback
            print(f"      ERROR: {e}")
            traceback.print_exc()
            failed.append(dn)

    print(f"\n{'='*50}")
    print(f"Done: {len(ok)} OK  {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")


if __name__ == '__main__':
    main()