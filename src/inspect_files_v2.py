"""
inspect_files_v2.py
=====================
More targeted than inspect_files.py -- searches PAST the boilerplate
(ROW definitions, power/filler nets) to find the exact patterns we need:
  - a real gate instance in DEF COMPONENTS (not a FILLER/TAP cell)
  - a real signal pin in DEF PINS
  - a real NETS entry
  - a real IOPATH line in the SDF (actual cell delay -- not power/filler)
  - a real INTERCONNECT line that is NOT a power net

Usage:
    python inspect_files_v2.py C:\\Krish\\GCNN\\data\\spm
"""
import sys
import os
import re

POWER_NAMES = {"VGND", "VPWR", "VNB", "VPB"}


def find_lines(path, pattern, max_matches=5, exclude_patterns=None, context=""):
    exclude_patterns = exclude_patterns or []
    found = []
    print(f"\n--- searching {os.path.basename(path)} for: {context} ---")
    if not os.path.exists(path):
        print("  !! file not found")
        return []
    with open(path, errors="replace") as f:
        for line in f:
            if pattern.search(line):
                if any(ex in line for ex in exclude_patterns):
                    continue
                found.append(line.rstrip())
                if len(found) >= max_matches:
                    break
    if found:
        for l in found:
            print(f"  {l}")
    else:
        print("  (no matches found)")
    return found


if __name__ == "__main__":
    folder = sys.argv[1]
    design = os.path.basename(folder.rstrip("/\\"))

    v_path = os.path.join(folder, f"{design}.synthesis_preroute.v")
    def_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".def")]
    def_path = def_paths[0] if def_paths else None
    sdf_path = os.path.join(folder, f"{design}.sdf")          # try the PLAIN sdf first
    at_sdf_path = os.path.join(folder, f"{design}.at.sdf")

    print("=" * 70)
    print(" 1. A REAL (non-filler, non-tap) instance line in the Verilog netlist")
    print("=" * 70)
    find_lines(v_path, re.compile(r'sky130_fd_sc_hd__\w+\s+_\d+_\s*\('),
              exclude_patterns=["FILLER", "TAP", "DECAP", "PHY"],
              context="a real gate instantiation")

    print("\n" + "=" * 70)
    print(" 2. DEF: the COMPONENTS section header + first few REAL instances")
    print("=" * 70)
    if def_path:
        with open(def_path, errors="replace") as f:
            text = f.read()
        m = re.search(r'COMPONENTS\s+\d+\s*;', text)
        if m:
            start = m.start()
            snippet = text[start:start + 4000]
            lines = snippet.split("\n")
            shown = 0
            for line in lines:
                if any(ex in line for ex in ["FILLER", "TAP", "DECAP", "PHY"]):
                    continue
                print(f"  {line}")
                shown += 1
                if shown >= 8:
                    break
        else:
            print("  !! COMPONENTS section not found")

    print("\n" + "=" * 70)
    print(" 3. DEF: the PINS section (first few entries)")
    print("=" * 70)
    if def_path:
        m = re.search(r'\bPINS\s+\d+\s*;', text)
        if m:
            snippet = text[m.start():m.start() + 2000]
            for line in snippet.split("\n")[:12]:
                print(f"  {line}")
        else:
            print("  !! PINS section not found")

    print("\n" + "=" * 70)
    print(" 4. DEF: the NETS section (first few entries, non-power)")
    print("=" * 70)
    if def_path:
        m = re.search(r'\bNETS\s+\d+\s*;', text)
        if m:
            snippet = text[m.start():m.start() + 3000]
            shown = 0
            for line in snippet.split("\n"):
                if any(p in line for p in POWER_NAMES) or "FILLER" in line:
                    continue
                print(f"  {line}")
                shown += 1
                if shown >= 8:
                    break
        else:
            print("  !! NETS section not found")

    print("\n" + "=" * 70)
    print(" 5. SDF: a real IOPATH line (actual cell delay) -- checking BOTH files")
    print("=" * 70)
    for path, label in [(sdf_path, "spm.sdf"), (at_sdf_path, "spm.at.sdf")]:
        find_lines(path, re.compile(r'IOPATH'), max_matches=4,
                  context=f"IOPATH lines in {label}")

    print("\n" + "=" * 70)
    print(" 6. SDF: a real INTERCONNECT line that is NOT power/filler")
    print("=" * 70)
    for path, label in [(sdf_path, "spm.sdf"), (at_sdf_path, "spm.at.sdf")]:
        find_lines(path, re.compile(r'INTERCONNECT'), max_matches=4,
                  exclude_patterns=list(POWER_NAMES) + ["FILLER"],
                  context=f"non-power INTERCONNECT lines in {label}")

    print("\n" + "=" * 70)
    print(" 7. SDF: what CELLTYPE values appear (first 10 unique, non-filler)")
    print("=" * 70)
    for path, label in [(sdf_path, "spm.sdf"), (at_sdf_path, "spm.at.sdf")]:
        if not os.path.exists(path):
            continue
        seen = set()
        with open(path, errors="replace") as f:
            for line in f:
                m = re.search(r'CELLTYPE\s+"?([^")\s]+)"?', line)
                if m and "FILLER" not in m.group(1) and "TAP" not in m.group(1):
                    seen.add(m.group(1))
                if len(seen) >= 10:
                    break
        print(f"  {label}: {sorted(seen)}")

    print("\n" + "=" * 70)
    print(" DONE -- copy ALL of the above and share it back")
    print("=" * 70)
