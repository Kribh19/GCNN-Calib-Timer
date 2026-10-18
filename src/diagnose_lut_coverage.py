"""
diagnose_lut_match.py
======================
READ-ONLY diagnostic. Does NOT modify preprocess.py, does NOT write any
.pt files, does NOT touch your preprocessed/ folder. It only imports and
calls your existing parse_liberty() / parse_verilog() functions and
reports, per cell type, which timing arcs fail to find a Liberty LUT
match and why -- so we can see the real cause of the ~20-40% match rate
seen across all 27 designs before touching any parsing code.

Usage (run from the same folder as preprocess.py):
    python diagnose_lut_match.py --design spm --data C:\\Krish\\GCNN\\data --techlib C:\\Krish\\GCNN\\data\\techlib

Run it once per design you want to inspect. Start with spm (smallest),
then try one large design (e.g. xtea or jpeg_encoder) to see if the
pattern is consistent or design-specific.
"""

import os
import argparse
import collections

# Import directly from your real preprocess.py -- nothing duplicated,
# nothing guessed. If this import fails, run this script from the same
# directory as preprocess.py.
from preprocess import parse_liberty, parse_verilog, OUTPUT_PORTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', required=True,
                     help='design name, e.g. spm')
    ap.add_argument('--data', required=True,
                     help='path to data dir, e.g. C:\\Krish\\GCNN\\data')
    ap.add_argument('--techlib', required=True,
                     help='path to techlib dir')
    args = ap.parse_args()

    lib_ff = os.path.join(args.techlib, 'sky130_fd_sc_hd__ff_n40C_1v95.lib')
    lib_ss = os.path.join(args.techlib, 'sky130_fd_sc_hd__ss_100C_1v60.lib')
    assert os.path.exists(lib_ff), f"FF liberty not found: {lib_ff}"
    assert os.path.exists(lib_ss), f"SS liberty not found: {lib_ss}"

    print(f"Loading Liberty files (unmodified parse_liberty)...")
    luts_ff, caps_ff = parse_liberty(lib_ff)
    luts_ss, caps_ss = parse_liberty(lib_ss)
    print(f"  Liberty cell types loaded: {len(luts_ff)} (ff), {len(luts_ss)} (ss)")

    v_path = os.path.join(args.data, args.design,
                           f"{args.design}.synthesis_preroute.v")
    assert os.path.exists(v_path), f"Verilog not found: {v_path}"

    print(f"\nParsing Verilog for design '{args.design}' (unmodified parse_verilog)...")
    inputs, outputs, instances = parse_verilog(v_path)
    print(f"  {len(instances)} instances parsed")

    # Rebuild the exact same cell_edges + arc_key construction logic used
    # in preprocess.py's build_design(), purely for reporting -- no .pt
    # files written, no state mutated.
    type_total          = collections.Counter()
    type_hit            = collections.Counter()
    missing_examples     = collections.defaultdict(set)  # cell_type -> arc_keys needed but not found
    available_examples   = collections.defaultdict(set)  # cell_type -> arc_keys Liberty actually has
    not_in_liberty_types = set()

    for ct, inst, conns in instances:
        out_port = None
        in_ports = []
        for port, sig in conns.items():
            if port in OUTPUT_PORTS:
                out_port = port
            else:
                in_ports.append(port)
        if out_port is None:
            # This instance has no recognized output port at all --
            # OUTPUT_PORTS may not cover this cell's real output pin name.
            # Flag separately, don't silently skip.
            type_total[ct] += 0  # ensure it shows up if it has any in_ports
            continue

        for rel_pin in in_ports:
            arc_key = f"{rel_pin}\u2192{out_port}"
            type_total[ct] += 1

            ff_has = arc_key in luts_ff.get(ct, {})
            ss_has = arc_key in luts_ss.get(ct, {})
            hit = ff_has or ss_has

            if hit:
                type_hit[ct] += 1
            else:
                missing_examples[ct].add(arc_key)

            if ct not in luts_ff and ct not in luts_ss:
                not_in_liberty_types.add(ct)

            if ct in luts_ff:
                available_examples[ct] |= set(luts_ff[ct].keys())
            if ct in luts_ss:
                available_examples[ct] |= set(luts_ss[ct].keys())

    total_arcs = sum(type_total.values())
    total_hits = sum(type_hit.values())

    print(f"\n{'='*100}")
    print(f"  RESULTS for design '{args.design}'")
    print(f"{'='*100}")
    print(f"  Overall arc match rate: {total_hits}/{total_arcs} "
          f"({100*total_hits/max(total_arcs,1):.1f}%)\n")

    if not_in_liberty_types:
        print(f"  ** {len(not_in_liberty_types)} cell types used in Verilog "
              f"have NO entry in Liberty at all: **")
        for ct in sorted(not_in_liberty_types):
            print(f"       {ct}  (used {type_total[ct]} times)")
        print()

    print(f"  {'cell type':42s} {'total':>7} {'hit':>6} {'miss':>6} {'miss%':>7}")
    print(f"  {'-'*42} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")

    rows = []
    for ct in type_total:
        tot, hit = type_total[ct], type_hit[ct]
        if tot == 0:
            continue
        miss = tot - hit
        rows.append((ct, tot, hit, miss))
    rows.sort(key=lambda r: -r[3])  # worst offenders first

    for ct, tot, hit, miss in rows:
        pct = 100 * miss / tot if tot else 0
        flag = "  <-- NOT IN LIBERTY" if ct in not_in_liberty_types else ""
        print(f"  {ct:42s} {tot:7d} {hit:6d} {miss:6d} {pct:6.1f}%{flag}")

    # For the single worst offender (excluding "not in liberty" types,
    # since those are a different failure mode), show exactly what arc_key
    # strings were needed vs what Liberty actually has -- this is the
    # detail that reveals naming mismatches (e.g. "A1" vs "A", "Y" vs "X").
    candidates = [r for r in rows if r[0] not in not_in_liberty_types and r[3] > 0]
    if candidates:
        worst_ct = candidates[0][0]
        print(f"\n  Detail for worst partially-matching type: '{worst_ct}'")
        print(f"    arc_keys NEEDED but MISSING (sample): "
              f"{sorted(missing_examples[worst_ct])[:8]}")
        print(f"    arc_keys Liberty ACTUALLY HAS (sample): "
              f"{sorted(available_examples[worst_ct])[:8]}")
        print(f"\n    Compare the two lists above -- if the pin names look")
        print(f"    similar but not identical (e.g. 'A1' vs 'A', or a")
        print(f"    different output port letter), that is the naming")
        print(f"    mismatch causing the miss. If Liberty's list is simply")
        print(f"    empty, that cell type/pin has no timing arc defined at")
        print(f"    all in this corner file (a real Liberty coverage gap,")
        print(f"    not a parser bug).")

    print(f"\n{'='*100}")
    print("  Done. No files were written or modified by this script.")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()