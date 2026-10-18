"""
inspect_files.py
==================
RUN THIS FIRST, before parse_real_design.py, on your REAL spm folder.

It prints the first lines of each key file so we can confirm the real
format matches what parse_real_design.py expects. If anything looks
different from a standard DEF / structural-Verilog / SDF file, paste
the output back and I'll adjust the parser to match exactly.

Usage:
    python inspect_files.py C:\\Krish\\GCNN\\data\\spm
"""
import sys
import os
import glob


def show_head(path, n=15, label=""):
    print(f"\n{'='*70}")
    print(f" {label}: {path}")
    print(f"{'='*70}")
    if not os.path.exists(path):
        print("  !! FILE NOT FOUND")
        return
    with open(path, errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                print("  ...")
                break
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_files.py <path-to-design-folder>")
        sys.exit(1)

    folder = sys.argv[1]
    design = os.path.basename(folder.rstrip("/\\"))

    print(f"Inspecting design folder: {folder}")
    print(f"Files found:")
    for f in sorted(os.listdir(folder)):
        full = os.path.join(folder, f)
        size = os.path.getsize(full)
        print(f"  {f}  ({size:,} bytes)")

    v_files = glob.glob(os.path.join(folder, "*.v"))
    def_files = glob.glob(os.path.join(folder, "*.def"))
    sdf_files = glob.glob(os.path.join(folder, "*.sdf"))

    if v_files:
        show_head(v_files[0], 25, "VERILOG (netlist structure)")
    if def_files:
        show_head(def_files[0], 30, "DEF (placement)")
    if sdf_files:
        show_head(sdf_files[0], 30, "SDF (timing labels)")

    print("\n" + "=" * 70)
    print(" NEXT STEP")
    print("=" * 70)
    print("Copy everything printed above and share it back -- I'll confirm")
    print("parse_real_design.py's patterns match, or fix any mismatches,")
    print("before you run the real parse.")
