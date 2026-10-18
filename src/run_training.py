"""
Non-interactive runner for train_paper_split.py.
Calls run() directly with fixed config — no stdin prompts.

Cycle 2 change (2026-09-25): cell delay computation moved inside the
level-by-level propagation loop. The propagated SLEW at each gate input
is now passed explicitly to the LUT coef_mlp as the slew-axis query
coordinate. Previously the LUT was queried once before propagation with
SLEW=0.15 everywhere; the coef_mlp had to guess the liberty table row
from h[cell_src] alone. Now it receives the actual physical slew at each
level, matching the original paper's SignalProp architecture. Expected
to improve synth_ram cell_delay R2 (was 0.53) by giving the LUT the
correct query point for high-slew deep-path nodes.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_paper_split import (
    discover_designs, split_designs, run, ModelConfig
)

DATA_DIR = r"C:\Krish\GCNN\data"

EPOCHS    = 500
LR        = 2e-3
LOG_EVERY = 10

cfg = ModelConfig(hidden_dim=32, net_layers=3, lut_k=7)
cfg.weight_decay        = 1e-4
cfg.early_stop_patience = 25

all_designs = discover_designs(DATA_DIR)
train_designs, test_designs = split_designs(all_designs)

print(f"Train ({len(train_designs)}): {sorted(train_designs)}")
print(f"Test  ({len(test_designs)}):  {sorted(test_designs)}")

result = run(cfg, DATA_DIR, train_designs, test_designs,
             epochs=EPOCHS, lr=LR, log_every=LOG_EVERY)

if result:
    print(f"\nRun saved to: {result['run_dir']}")
    print(f"Best mean-per-design AT R2: {result['best_pooled_test_r2']:.4f}")
