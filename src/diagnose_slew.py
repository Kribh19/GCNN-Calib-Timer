"""
Run this from C:\Krish\GCNN\src:
    python diagnose_slew.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from parse_real_design import build_graph
from parse_liberty import get_liberty_db, get_lut_for_arc

TECHLIB = r"C:\Krish\GCNN\data\techlib"
DESIGN  = r"C:\Krish\GCNN\data\xtea"

print("Loading liberty...")
db = get_liberty_db(TECHLIB, corner="ff")
print(f"  {len(db)} arcs loaded")

print("\nParsing xtea...")
g = build_graph(DESIGN, "xtea")

# --- 1. What fraction of cell edges have real lut_slew? ---
lut_k = 7
zero_lut = 0
nonzero_lut = 0
slew_vals_nonzero = []

cell_arc_info = g.get("cell_arc_info", [])
for idx, (ct, in_port, out_port) in enumerate(cell_arc_info):
    ld, ls = get_lut_for_arc(db, ct, in_port, out_port, k=lut_k)
    if ls.max() > 0:
        nonzero_lut += 1
        slew_vals_nonzero.extend(ls.flatten().tolist())
    else:
        zero_lut += 1

total = zero_lut + nonzero_lut
print(f"\n--- LUT slew coverage ---")
print(f"  Cell edges with nonzero lut_slew: {nonzero_lut}/{total} ({100*nonzero_lut/total:.1f}%)")
print(f"  Cell edges with ZERO lut_slew:    {zero_lut}/{total} ({100*zero_lut/total:.1f}%)")
if slew_vals_nonzero:
    arr = np.array(slew_vals_nonzero)
    print(f"  lut_slew value range (nonzero): [{arr.min():.4f}, {arr.max():.4f}] ns")
    print(f"  lut_slew mean: {arr.mean():.4f} ns")

# --- 2. Gradient sparsity through index_reduce(amax) ---
from collections import Counter
net_edges = g["net_edges"]
dst_count = Counter(dst for _, dst in net_edges)
fanin1  = sum(1 for v in dst_count.values() if v == 1)
fanin_gt1 = sum(1 for v in dst_count.values() if v > 1)
total_dst = len(dst_count)
print(f"\n--- Net edge fanin (gradient sparsity) ---")
print(f"  Pins with fanin=1:  {fanin1}/{total_dst} ({100*fanin1/total_dst:.1f}%)")
print(f"  Pins with fanin>1:  {fanin_gt1}/{total_dst} ({100*fanin_gt1/total_dst:.1f}%)")
print(f"  For fanin>1 pins, only 1 edge gets gradient (amax is not differentiable elsewhere)")

# --- 3. What does true_at_slew look like vs LUT propagated? ---
true_slew = g.get("true_at_slew", {})
if true_slew:
    vals = [v for v in true_slew.values() if v > 0]
    print(f"\n--- True slew from at.sdf ---")
    print(f"  Count: {len(vals)}")
    print(f"  Range: [{min(vals):.4f}, {max(vals):.4f}] ns")
    print(f"  Mean: {np.mean(vals):.4f}  Std: {np.std(vals):.4f}")

print("\nDone.")
