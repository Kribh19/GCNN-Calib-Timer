"""
parse_liberty.py
=================
Parses sky130hd Liberty (.lib) files and extracts the 7x7 delay/slew
LUT tables for every cell arc.

Returns a dict:
    lut_db[(cell_type, in_pin, out_pin)] = {
        "cell_rise":        np.array shape (7,7),   # delay, rise transition
        "cell_fall":        np.array shape (7,7),   # delay, fall transition
        "rise_transition":  np.array shape (7,7),   # output slew, rise
        "fall_transition":  np.array shape (7,7),   # output slew, fall
        "index_slew":       np.array shape (7,),    # input slew axis
        "index_load":       np.array shape (7,),    # output load axis
    }

For the model we use:
    lut_delay[e] = (cell_rise + cell_fall) / 2      shape (7,7)
    lut_slew[e]  = (rise_transition + fall_transition) / 2  shape (7,7)

averaged over rise/fall to get a single representative value per arc,
matching the paper's 2-corner (early/late) x 2-transition (rise/fall)
setup simplified to 1 effective LUT per arc.

Usage:
    from parse_liberty import load_liberty_db, get_lut_for_arc
    db = load_liberty_db("path/to/sky130_fd_sc_hd__ff_n40C_1v95.lib")
    delay_lut, slew_lut = get_lut_for_arc(db, "sky130_fd_sc_hd__inv_1", "A", "Y")
"""
import re
import numpy as np
from functools import lru_cache


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _parse_values_block(text):
    """Extract all numbers from a values(...) block -- handles multi-line."""
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    return [float(n) for n in nums]


def _parse_index(line):
    """Parse index_1/index_2 line -> list of floats."""
    m = re.search(r'index_\d\s*\(\s*"([^"]+)"', line)
    if not m:
        return []
    return [float(x.strip()) for x in m.group(1).split(',') if x.strip()]


def _extract_table(block_text, table_name):
    """
    Find a named sub-block like cell_rise(...) { ... } inside block_text
    and return (index_slew, index_load, 7x7 array).
    Returns None if not found.
    """
    # find the table block
    pat = re.compile(
        table_name + r'\s*\([^)]*\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        re.S)
    m = pat.search(block_text)
    if not m:
        return None
    inner = m.group(1)

    # parse index axes
    idx1, idx2 = [], []
    for line in inner.split('\n'):
        if 'index_1' in line:
            idx1 = _parse_index(line)
        elif 'index_2' in line:
            idx2 = _parse_index(line)

    # parse values block
    vm = re.search(r'values\s*\(([^;]+)\)', inner, re.S)
    if not vm:
        return None
    nums = _parse_values_block(vm.group(1))

    if not idx1 or not idx2 or not nums:
        return None

    n1, n2 = len(idx1), len(idx2)
    if len(nums) < n1 * n2:
        return None

    arr = np.array(nums[:n1 * n2], dtype=np.float32).reshape(n1, n2)
    return np.array(idx1, dtype=np.float32), np.array(idx2, dtype=np.float32), arr


# ---------------------------------------------------------------------------
# main parser
# ---------------------------------------------------------------------------
def load_liberty_db(lib_path, verbose=False):
    """
    Parse a Liberty file and return:
        db[(cell_type, in_pin, out_pin)] = {
            "cell_rise", "cell_fall",
            "rise_transition", "fall_transition",
            "index_slew", "index_load"
        }
    """
    with open(lib_path, errors='replace') as f:
        text = f.read()

    db = {}
    n_arcs = 0

    # find every cell block
    cell_pat = re.compile(r'\bcell\s*\(\s*"?(\S+?)"?\s*\)\s*\{', re.S)
    cell_starts = [(m.start(), m.group(1)) for m in cell_pat.finditer(text)]

    for ci, (cstart, cell_name) in enumerate(cell_starts):
        # find the matching closing brace for this cell
        next_start = cell_starts[ci + 1][0] if ci + 1 < len(cell_starts) else len(text)
        cell_block = text[cstart:next_start]

        # find output pin blocks
        pin_pat = re.compile(r'\bpin\s*\(\s*"?(\S+?)"?\s*\)\s*\{', re.S)
        pin_starts = [(m.start(), m.group(1)) for m in pin_pat.finditer(cell_block)]

        for pi, (pstart, pin_name) in enumerate(pin_starts):
            next_pstart = pin_starts[pi + 1][0] if pi + 1 < len(pin_starts) else len(cell_block)
            pin_block = cell_block[pstart:next_pstart]

            # only output pins have timing arcs with cell_rise
            if 'cell_rise' not in pin_block:
                continue

            # find timing blocks within this pin
            timing_pat = re.compile(r'\btiming\s*\(\s*\)\s*\{', re.S)
            timing_starts = [m.start() for m in timing_pat.finditer(pin_block)]

            for ti, tstart in enumerate(timing_starts):
                next_tstart = timing_starts[ti + 1] if ti + 1 < len(timing_starts) else len(pin_block)
                timing_block = pin_block[tstart:next_tstart]

                # get related_pin (input pin)
                rp = re.search(r'related_pin\s*:\s*"?(\S+?)"?\s*;', timing_block)
                if not rp:
                    continue
                in_pin = rp.group(1).strip('"')

                # skip constraint arcs (setup/hold) - we only want propagation arcs
                ts = re.search(r'timing_type\s*:\s*"?(\S+?)"?\s*;', timing_block)
                if ts and ts.group(1) in ('setup_rising', 'hold_rising',
                                           'setup_falling', 'hold_falling',
                                           'recovery_rising', 'removal_rising'):
                    continue

                # extract the 4 LUT tables
                rise_result = _extract_table(timing_block, 'cell_rise')
                fall_result = _extract_table(timing_block, 'cell_fall')
                rt_result   = _extract_table(timing_block, 'rise_transition')
                ft_result   = _extract_table(timing_block, 'fall_transition')

                if rise_result is None or fall_result is None:
                    continue

                idx_slew, idx_load, cell_rise = rise_result
                _,        _,        cell_fall = fall_result
                _, _, rise_trans = rt_result if rt_result else (idx_slew, idx_load, cell_rise * 0.9)
                _, _, fall_trans = ft_result if ft_result else (idx_slew, idx_load, cell_fall * 0.9)

                key = (cell_name, in_pin, pin_name)
                db[key] = {
                    "cell_rise":       cell_rise,
                    "cell_fall":       cell_fall,
                    "rise_transition": rise_trans,
                    "fall_transition": fall_trans,
                    "index_slew":      idx_slew,
                    "index_load":      idx_load,
                }
                n_arcs += 1

    if verbose:
        print(f"[parse_liberty] parsed {len(db)} arcs from {n_arcs} timing blocks "
              f"across cells in {lib_path}")
    return db


def get_lut_for_arc(db, cell_type, in_pin, out_pin, k=7):
    """Return (lut_delay, lut_slew) each shape (k,k). Zeros if arc not found."""
    entry = db.get((cell_type, in_pin, out_pin))
    if entry is None:
        return np.zeros((k, k), dtype=np.float32), np.zeros((k, k), dtype=np.float32)
    lut_delay = (entry["cell_rise"] + entry["cell_fall"]) / 2.0
    lut_slew  = (entry["rise_transition"] + entry["fall_transition"]) / 2.0
    def resize(arr, k):
        if arr.shape == (k, k): return arr
        from scipy.ndimage import zoom
        return zoom(arr, (k/arr.shape[0], k/arr.shape[1])).astype(np.float32)
    try:
        lut_delay = resize(lut_delay, k)
        lut_slew  = resize(lut_slew,  k)
    except Exception:
        lut_delay = np.zeros((k, k), dtype=np.float32)
        lut_slew  = np.zeros((k, k), dtype=np.float32)
    return lut_delay, lut_slew


# Default sky130hd ff-corner LUT axes (used as fallback when arc not found)
_DEFAULT_SLEW_AXIS = np.array([0.01,0.023,0.053,0.123,0.283,0.65,1.5],   dtype=np.float32)
_DEFAULT_LOAD_AXIS = np.array([0.0005,0.001,0.002,0.004,0.008,0.016,0.032], dtype=np.float32)


def get_lut_for_arc_with_axes(db, cell_type, in_pin, out_pin, k=7):
    """
    Return (lut_delay, lut_slew, lut_valid, idx_slew_norm, idx_load_norm).

    lut_delay, lut_slew : shape (k,k) -- LUT values averaged rise+fall
    lut_valid           : float 1.0 if arc found in liberty, 0.0 otherwise
                          (matches paper Table 3 'LUT is valid or not' feature)
    idx_slew_norm       : shape (k,)  -- slew axis values normalised to [0,1]
    idx_load_norm       : shape (k,)  -- load axis values normalised to [0,1]
                          (matches paper Table 3 'LUT indices 8x(7+7)' feature;
                           we have 1 corner x (7+7)=14 values per arc)
    """
    lut_delay, lut_slew = get_lut_for_arc(db, cell_type, in_pin, out_pin, k)
    entry = db.get((cell_type, in_pin, out_pin))
    lut_valid = 1.0 if entry is not None else 0.0

    def _norm(arr, k):
        if arr is None or len(arr) == 0:
            return np.linspace(0.0, 1.0, k, dtype=np.float32)
        arr = np.array(arr, dtype=np.float32)
        if len(arr) != k:
            xo = np.linspace(0, 1, len(arr))
            xn = np.linspace(0, 1, k)
            arr = np.interp(xn, xo, arr).astype(np.float32)
        mn, mx = arr.min(), arr.max()
        return ((arr - mn) / (mx - mn + 1e-12)).astype(np.float32)

    if entry is None:
        idx_slew = _norm(_DEFAULT_SLEW_AXIS, k)
        idx_load = _norm(_DEFAULT_LOAD_AXIS, k)
    else:
        idx_slew = _norm(entry.get("index_slew", _DEFAULT_SLEW_AXIS), k)
        idx_load = _norm(entry.get("index_load", _DEFAULT_LOAD_AXIS), k)

    return lut_delay, lut_slew, lut_valid, idx_slew, idx_load


# ---------------------------------------------------------------------------
# cached loader -- call this once, reuse across all designs in a session
# ---------------------------------------------------------------------------
_DB_CACHE = {}

def get_liberty_db(techlib_dir, corner="ff"):
    """
    Load and cache the liberty DB for a given corner.
    corner: "ff" (fast/early) or "ss" (slow/late)
    """
    import os
    if corner in _DB_CACHE:
        return _DB_CACHE[corner]

    lib_map = {
        "ff": "sky130_fd_sc_hd__ff_n40C_1v95.lib",
        "ss": "sky130_fd_sc_hd__ss_100C_1v60",   # your file is missing .lib ext
    }
    fname = lib_map.get(corner, lib_map["ff"])
    path = os.path.join(techlib_dir, fname)

    if not os.path.exists(path):
        # try adding .lib extension
        path2 = path + ".lib"
        if os.path.exists(path2):
            path = path2
        else:
            print(f"[parse_liberty] WARNING: lib file not found: {path} -- using zeros")
            return {}

    print(f"[parse_liberty] loading liberty: {path} ...")
    db = load_liberty_db(path, verbose=True)
    _DB_CACHE[corner] = db
    return db


if __name__ == "__main__":
    import sys, os
    path = sys.argv[1] if len(sys.argv) > 1 else \
           r"C:\Krish\GCNN\data\techlib\sky130_fd_sc_hd__ff_n40C_1v95.lib"
    db = load_liberty_db(path, verbose=True)
    print(f"\nTotal arcs parsed: {len(db)}")
    print("\nSample entries:")
    for i, (key, val) in enumerate(list(db.items())[:5]):
        print(f"  {key}  ->  delay shape {val['cell_rise'].shape}  "
              f"slew shape {val['rise_transition'].shape}")
        print(f"    delay range: {val['cell_rise'].min():.4f} - {val['cell_rise'].max():.4f} ns")

# ---------------------------------------------------------------------------
# Pin capacitance extraction -- matches paper Table 2: "capacitance 4x EL/RF"
# We approximate with 1x (single corner) since we only have ff lib
# ---------------------------------------------------------------------------
def load_pin_caps(lib_path):
    """
    Parse Liberty and return {(cell_type, pin_name): capacitance_pF} for INPUT pins.
    The paper uses this as a node feature (4x EL/RF; we use 1x for simplicity).
    """
    pin_caps = {}
    try:
        with open(lib_path, errors='replace') as f:
            text = f.read()
        cell_pat = re.compile(r'\bcell\s*\(\s*"?(\S+?)"?\s*\)\s*\{', re.S)
        cell_starts = [(m.start(), m.group(1)) for m in cell_pat.finditer(text)]
        for ci, (cstart, cell_name) in enumerate(cell_starts):
            next_start = cell_starts[ci+1][0] if ci+1 < len(cell_starts) else len(text)
            cell_block = text[cstart:next_start]
            pin_pat = re.compile(r'\bpin\s*\(\s*"?(\S+?)"?\s*\)\s*\{', re.S)
            pin_starts = [(m.start(), m.group(1)) for m in pin_pat.finditer(cell_block)]
            for pi, (pstart, pin_name) in enumerate(pin_starts):
                next_pstart = pin_starts[pi+1][0] if pi+1 < len(pin_starts) else len(cell_block)
                pin_block = cell_block[pstart:next_pstart]
                if 'cell_rise' in pin_block:
                    continue   # output pin -- skip
                cap_m = re.search(r'\bcapacitance\s*:\s*([\d.eE+-]+)', pin_block)
                if cap_m:
                    pin_caps[(cell_name, pin_name)] = float(cap_m.group(1))
    except Exception as e:
        print(f"[parse_liberty] WARNING: pin cap extraction failed: {e}")
    return pin_caps


_PINCAP_CACHE = {}

def get_pin_caps(techlib_dir, corner="ff"):
    """Load and cache per-pin input capacitances from liberty."""
    import os
    if corner in _PINCAP_CACHE:
        return _PINCAP_CACHE[corner]
    lib_map = {"ff": "sky130_fd_sc_hd__ff_n40C_1v95.lib",
               "ss": "sky130_fd_sc_hd__ss_100C_1v60.lib"}
    path = os.path.join(techlib_dir, lib_map.get(corner, lib_map["ff"]))
    if not os.path.exists(path):
        return {}
    caps = load_pin_caps(path)
    _PINCAP_CACHE[corner] = caps
    print(f"[parse_liberty] loaded {len(caps)} pin input caps from liberty")
    return caps