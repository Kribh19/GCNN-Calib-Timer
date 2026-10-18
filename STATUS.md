# CalibTimer — Ground Truth Status
**Date verified:** 2026-09-25  
**Verified by:** reading actual files this session (all claims labeled READ/MEASURED/ASSUMED)

---

## What is ACTUALLY TRUE right now

### 1. There are TWO separate training pipelines — they are NOT the same thing

| | `train.py` | `train_paper_split.py` |
|---|---|---|
| Model | `model_configurable.py` (v3, 8-table LUT) | `gnn_model.py` (VN + level enc + at_init_mlp) |
| Data | Preprocessed `.pt` files (`C:\Krish\GCNN\preprocessed`) | Raw DEF/SDF/SDC via `parse_real_design` |
| Node features | **18** (line 249: `assert NODE_FEATURES == 18`) | **12** (`interactive_train_real.py:NODE_FEATURES=12`) |
| LUT shape | `[E, 8, 7, 7]` (8-corner, multi-table) | `[E, 7, 7]` (1 averaged table) |
| Early stop metric | **POOLED** test R² (line 463: `scheduler.step(te_r2_at)`) | **mean-per-design** R² |
| Leakage guard | ❌ NO `CALIBRATION_DESIGNS` guard | ✅ Added this session (2026-09-25) |

READ: `train.py:35` `from model_configurable import ModelConfig, TimingGNN, NODE_FEATURES`  
READ: `train.py:249` `assert NODE_FEATURES == 18`  
READ: `train_paper_split.py:28` `from gnn_model import ModelConfig, TimingGNN, NODE_IN_DIM`  

---

### 2. Claimed leakage fix status

**PARTIAL — not in train.py.**

- `train_paper_split.py`: ✅ `CALIBRATION_DESIGNS` set + `split_designs()` exclusion added 2026-09-25 this session.
- `train.py`: ❌ Still has NO `CALIBRATION_DESIGNS` guard. Lines 51-59 define `PAPER_TRAIN` (14) and `PAPER_TEST` (7) but no calibration exclusion. If `uni_counter4b`, `cla8b`, `gcd`, `seq_detector`, `priority_enc` exist in the preprocessed folder, they will silently enter training when using train.py.

READ: `train.py:51-59` — no CALIBRATION_DESIGNS anywhere in the file.

---

### 3. Results: NO clean post-fix run exists

MEASURED: `ls C:\Krish\GCNN\src\results\` — 33 run directories, newest `20260819_133902_train20_test7`.  
Every run is named `train20_test7` or earlier single-design experiments. **Not a single run exists with a clean 15-design split.**  
The 0.9023 mean-per-design R² (from `20260819_133902_train20_test7`) is STALE and must not be cited.

---

### 4. conformal.py — actual state

READ: `conformal.py:1-48` — Design-level split conformal, correct `ceil((n+1)(1-α)/n)` quantile formula.  
READ: `conformal.py:40-48` — Explicit note: "predictions.csv as currently saved by train_paper_split.py only contains rows for the PAPER TEST designs. To calibrate on the custom pool, you must ALSO run inference on those 5 custom designs."  

**conformal.py is correctly implemented BUT cannot be run yet** because:
1. No post-fix training run exists, so there are no predictions CSVs to feed it.
2. Calibration-pool inference (uni_counter4b, cla8b, gcd, seq_detector, priority_enc) has never been run.
3. The `append_calibration_predictions()` helper exists in conformal.py but has not been called.

---

### 5. Feature count (12 vs 18)

The "12 features" description applies ONLY to the raw pipeline (`train_paper_split.py` + `gnn_model.py`):  
READ: `interactive_train_real.py:36` `NODE_FEATURES = 12`  

The preprocessed `.pt` pipeline (`train.py` + `model_configurable.py`) uses 18:  
READ: `model_configurable.py:76` `NODE_FEATURES = 18`  

These are not interchangeable. The two pipelines produce tensors of different shapes and use different model architectures.

---

### 6. Sanity counter gaps (Rule 5)

No training run has been executed this session, so no runtime sanity counters (liberty cell count, arc count, lut_hits, nd_mask size, SLEW_true.std()) can be reported as MEASURED. These must be collected when the next training run is executed.

---

## Summary: What needs to happen next (in order)

1. **Run `train_paper_split.py` on the clean 15-design split** (leakage fix now in place) to get the first valid post-fix AT R² numbers. This is the top priority.
2. **Add CALIBRATION_DESIGNS guard to `train.py`** (the preprocessed-pipeline trainer) — currently unguarded.
3. **After step 1**, run `append_calibration_predictions()` on uni_counter4b, cla8b, seq_detector, priority_enc (gcd excluded pending flatten re-synth) against the trained model, then run `conformal.py`.
4. **Verify SDF format** of calibration designs (Rule 2 gate) before preprocessing them.

---

## Numbers that are VALID to cite

None currently. All AT R² numbers (including 0.9023) come from runs with training-set leakage and are invalidated.

## Numbers that must NEVER be cited

- R²=0.9975/0.9976 (leaky split, old pipeline)
- R²=0.9023 (leaky 20-design split, pre-fix)
- Any xtea/spm/synth_ram number from `20260803_135146` or earlier (AT_true label bug)
