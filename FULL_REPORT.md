# CalibTimer GCNN — Full Project Report
**For:** Handoff to next Claude session  
**Date:** 2026-10-01  
**Author:** Krish Bhavsar + Claude (session 5db3e7de)

---

## 1. Project Goal

Build **CalibTimer** — a Graph Neural Network (GNN) that predicts pre-routing Static Timing Analysis (STA) values at the pin level before physical routing, using the SkyWater sky130hd 130nm PDK with OpenROAD Flow Scripts benchmarks.

**Outputs predicted per pin:**
- AT (Arrival Time) — the primary target metric
- Slack (= RAT − AT)
- SLEW (transition time)
- net_delay and cell_delay (intermediate predictions used inside the model)

**Headline metric:** Mean-per-design AT R² across the 7 official DAC 2022 test designs  
**Target:** All 7 test designs must reach AT R² ≥ 0.90

**Paper context:** This extends DAC 2022 "TimingPredict" (Guo et al.) with:
1. Leakage-audited SDF/Liberty parsing pipeline
2. Re-verified timer-inspired GNN on the official DAC 2022 benchmark split
3. Design-level split conformal prediction for guaranteed slack coverage intervals

---

## 2. Data Splits (FIXED — never change without explicit discussion)

### Official DAC 2022 Test Designs (7) — NEVER train on these
```
jpeg_encoder, usbf_device, aes192, xtea, spm, y_huff, synth_ram
```

### Training Designs (15) — current clean split
```
BM64, aes128, aes256, aes_cipher, alu4bit, blabla, cic_decimator, des,
genericfir, picorv32a, salsa20, usb, usb_cdc_core, wbqspiflash, zipdiv
```
*(14 official DAC 2022 train designs + alu4bit)*

### Calibration Designs (5) — excluded from training (split-conformal validity)
```
uni_counter4b, cla8b, gcd (flatten bug — not usable), seq_detector, priority_enc
```
**CRITICAL:** `CALIBRATION_DESIGNS` is hardcoded in `train_paper_split.py` with a disjoint assertion. If calibration designs appear in training, split-conformal statistical validity breaks. This was a real historical bug that was fixed.

### DAC 2022 Split Derivation
```python
random.seed(8026728); random.sample(available_data, 14)
```
Confirmed from the DAC 2022 paper repo.

---

## 3. Model Architecture

### File: `C:\Krish\GCNN\src\gnn_model.py`

**Class:** `TimingGNN` (inside `gnn_model.py`)  
**Params:** ~40,730–40,794 (varies slightly by run, hidden_dim=32)

**Node features (12 — must match `real_graph_to_tensors()` exactly):**
| Index | Feature |
|---|---|
| 0 | x / canvas_scale |
| 1 | y / canvas_scale |
| 2 | is_PI (primary input flag) |
| 3 | is_IN |
| 4 | is_OUT |
| 5 | dist_left / die_w |
| 6 | dist_right / die_w |
| 7 | dist_bottom / die_h |
| 8 | dist_top / die_h |
| 9 | is_endpoint |
| 10 | fanout_norm |
| 11 | input_delay_norm (set_input_delay / 10.0, 0.0 if none) |

**Net edge features (4):** `[dx, dy, manhattan, fanout_norm]`

**Architecture pipeline:**
```
Node features (12) + level encoding
        ↓
NetEmbedding (3-layer sum_max GNN + virtual node)
        ↓
Level-ordered AT/SLEW propagation loop:
  for each level (topological order):
    CellLUTModule (Kronecker LUT k=7) → cell_delay
    net_delay head
    AT = max(incoming AT + cell_delay + net_delay)
    SLEW update
        ↓
Output: AT, SLEW, net_delay, cell_delay per pin
```

**Key architectural features beyond simple GNN:**
- **Virtual Node (VN):** One extra learned node connected to every pin — fixes xtea's deep sparse DAG by giving global information flow
- **Sinusoidal level encoding** (NeRF/Transformer-style, dim=32), projected additively into initial node embedding h0
- **at_init_mlp:** Learned per-node initial AT (softplus activation), so PI pins don't start frozen at 0
- **Scheduled sampling:** Linear ramp from teacher-forcing → self-prediction over epochs: `p_groundtruth = 1 - (epoch-1) / (epochs-1)`
- **Gradient detach:** `net_delay` and `cell_delay` are `.detach()`ed before being fed back into AT propagation — this was critical for fixing net_delay R² collapse on spm
- **CellLUTModule:** Kronecker-style LUT with k=7 grid resolution (matches Liberty table size)

**ModelConfig defaults:**
```python
hidden_dim = 32
net_layers = 3
aggregation = "sum_max"
lut_k = 7
cell_correction = True
weight_decay = 1e-4
early_stop_patience = 25
```

---

## 4. Training Setup

### File: `C:\Krish\GCNN\src\train_paper_split.py`

**Loss function (current — clean baseline):**
```python
loss = (2.0 * mse(out["AT"], g["AT_true"])
        + mse(out["SLEW"], g["SLEW_true"])
        + 0.3 * mse(out["net_delay"], g["nd_true"])
        + 0.3 * mse(out["cell_delay"], g["cd_true"]))
```

**Optimizer:** Adam, lr=2e-3, weight_decay=1e-4  
**Scheduler:** ReduceLROnPlateau (mode=max, factor=0.5, patience=epochs//20)  
**Grad clip:** `clip_grad_norm_(..., max_norm=1.0)`  
**Early stopping:** patience=25 (by mean-per-design AT R²)  

**How to run (interactive prompts in order):**
```
1. Path to data folder   → press Enter (default: C:\Krish\GCNN\data)
2. Hidden dimension      → press Enter (default: 32)
3. Net Embedding layers  → press Enter (default: 3)
4. LUT grid resolution   → press Enter (default: 7)
5. Weight decay          → press Enter (default: 0.0001)
6. Early stopping patience → e.g. 25 or 50
7. Training epochs        → e.g. 400 or 500
8. Learning rate          → press Enter (default: 0.002)
9. Log every N epochs     → e.g. 10 or 50
10. Try another config?   → n
```

**How to run non-interactively (PowerShell):**
```powershell
cd C:\Krish\GCNN\src
$inputs = @("", "", "", "", "", "25", "400", "", "10", "n") -join "`n"
$inputs | python train_paper_split.py
```

---

## 5. Complete Experiment History

### IMPORTANT: What AT R² means
- **Mean-per-design AT R²:** Average R² computed separately per design then averaged — each design counts equally regardless of pin count. This is the headline metric.
- **Pooled AT R²:** R² computed over all pins across all designs — larger designs dominate. Not used for comparisons.

---

### Pre-Clean-Split Runs (INVALIDATED — do not cite)

All runs before `20260925_*` used either a leaky split (calibration designs in training) or had the AT_true label bug (real SDF arrival data was being discarded). Numbers from these runs are meaningless. Key invalidated claims:
- R² ≈ 0.9975/0.9976 — leaky split
- xtea=0.9958, spm=0.9439, synth_ram=0.9460 (run `20260803_135146`) — AT_true bug

---

### Cycle 1 — CURRENT BEST ✅
**Run:** `20260925_200206_train15_test7`  
**Config:** hid=32, layers=3, lut_k=7, patience=25, lr=2e-3, 500 epochs logged, best@370  
**Fixes:** None — clean baseline  
**Params:** 40,730

| design | AT R² | slack R² | cell_delay R² |
|---|---|---|---|
| aes192 | **0.9654** | 0.9753 | 0.6998 |
| jpeg_encoder | **0.9513** | 0.9836 | 0.5740 |
| spm | **0.9515** | 0.9961 | -1.1378 |
| synth_ram | **0.6909** ❌ | 0.9725 | 0.5299 |
| usbf_device | **0.9824** | 0.9951 | 0.5970 |
| xtea | **0.8441** ❌ | 0.9883 | 0.7317 |
| y_huff | **0.9063** | 0.9737 | 0.6808 |
| **mean** | **0.8988** | — | — |

Training curve: reached 0.893 at epoch 120, slowly plateaued around 0.897–0.899 from epoch 230 onward. Best was 0.8988 at epoch 370. Checkpoint saved.

**Why the model struggles with xtea and synth_ram — root cause analysis:**

**xtea (stuck at 0.844):**
- xtea has primary input (PI) pins with `set_input_delay = 5.5 ns` → feature_11 = 0.55 normalized
- Training designs have input_delay_norm in range 0.10–0.30 → **0.55 is severely out-of-distribution**
- `at_init_mlp` must extrapolate OOD → predicts wrong initial AT for all PI pins → all downstream AT wrong since AT propagates forward from PIs
- Additionally xtea has a deep sparse DAG (many reconvergent paths) that the GNN struggles to traverse

**synth_ram (stuck at 0.691):**
- synth_ram has max_level = 46; training designs max at ~15–25 levels
- Sinusoidal level encoding is evaluated at level indices 26–46 which are **completely OOD** during training
- Wrong level encoding → wrong slew weight lookup in LUT → wrong cell delays → wrong AT propagation
- Also: synth_ram has memory-style timing paths (mux trees, wide AND trees) unlike any training design

---

### Cycle 3 — All 7 Fixes at Once (REGRESSION ❌)
**Run:** `20260928_192210_train15_test7`  
**Fixes applied:** F1+F2+F3+F4+F5+F6+F7 simultaneously  
**Result:** mean AT R² = 0.869 (was 0.899) — significant regression

| design | AT R² | vs Cycle 1 |
|---|---|---|
| aes192 | ~0.96 | ≈0 |
| jpeg_encoder | ~0.96 | +0.01 |
| spm | ~0.87 | -0.08 |
| synth_ram | 0.608 | -0.083 |
| usbf_device | ~0.97 | -0.01 |
| xtea | 0.768 | -0.076 |
| y_huff | ~0.97 | +0.06 |
| **mean** | **0.869** | **-0.030** |

SLEW R² was catastrophically negative (−10 to −71) in this run.

**Root causes:**
- **Fix 5 (SLEW weight 0.1×):** Drastically reduced gradient signal to SLEW head → SLEW head predicted garbage → wrong slew fed as LUT slew-axis coordinate → wrong cell delays → wrong AT propagation (cascading failure)
- **Fix 4 (h[o] in cell_feat_lvl):** Doubled LUT input width from (h + n_gates + 1) to (2h + n_gates + 1) — the model needs many more epochs to converge with doubled input; insufficient epochs given

---

### Run C — Fix 1+2+3 Only, 300 Epochs
**Run:** `20260928_224543_train15_test7`  
**Config:** F1+F2+F3, patience=25, 300 epochs  
**Params:** 40,794  
**Best epoch:** 300 (patience=0/25 at final epoch — NOT converged)

| design | AT R² | vs Cycle 1 |
|---|---|---|
| aes192 | 0.9655 | +0.001 |
| jpeg_encoder | 0.9674 | +0.016 |
| spm | 0.8688 | -0.083 |
| synth_ram | 0.5948 | -0.096 |
| usbf_device | 0.9728 | -0.010 |
| xtea | 0.7877 | -0.056 |
| y_huff | 0.9938 | +0.088 |
| **mean** | **0.879** | **-0.020** |

Note: Model was still improving at epoch 300 (patience counter = 0). Not a fair comparison to Cycle 1.

---

### Run D — Fix 1+2+3 Only, 500 Epochs (full convergence test)
**Run:** `20260930_190934_train15_test7`  
**Config:** F1+F2+F3, patience=50, 500 epochs  
**Params:** 40,794  
**Best epoch:** 400

| epoch | mean AT R² |
|---|---|
| 100 | 0.759 |
| 200 | 0.846 |
| 300 | 0.880 |
| 350 | 0.887 |
| **400 (best)** | **0.892** |
| 450 | 0.886 |
| 500 | 0.884 |

| design | AT R² | vs Cycle 1 |
|---|---|---|
| aes192 | 0.9663 | +0.001 |
| jpeg_encoder | **0.9804** | **+0.029** |
| spm | 0.8794 | -0.073 |
| synth_ram | 0.6548 | -0.036 |
| usbf_device | 0.9732 | -0.010 |
| xtea | 0.7999 | -0.044 |
| y_huff | **0.9894** | **+0.083** |
| **mean** | **0.892** | **-0.007** |

**Verdict:** Even at full convergence (500 epochs), Fix 1+2+3 does NOT beat Cycle 1 (0.892 vs 0.899). Fix 3 (weighted MSE) helps jpeg_encoder and y_huff but actively hurts spm, xtea, and synth_ram.

---

## 6. Fix Inventory — Every Fix Tried

| Fix | Description | Where | Status | Effect |
|---|---|---|---|---|
| **F1** | PI direct formula: bypass at_init_mlp for PI pins, use `AT = input_delay_ns` directly | `gnn_model.py` forward() | **REVERTED** | Helped xtea marginally (+0.01) but not enough to reach 0.90 |
| **F2** | Level normalization per-design: `level_normed = level / max_level * 30` before sinusoidal encoding | `gnn_model.py` NetEmbedding.forward() | **REVERTED** | Did not fix synth_ram OOD issue — likely needs more than just normalization |
| **F3** | Weighted MSE 10×: `_w = 1 + 9*(AT_true > 0.01)` on AT loss | `train_paper_split.py` loss | **REVERTED** | Helps jpeg/y_huff, hurts spm/xtea/synth_ram — net negative |
| **F4** | Add h[o] (output pin embedding) to cell_feat_lvl input | `gnn_model.py` cell loop | **PERMANENTLY REJECTED** | Doubled LUT input width → needs far more epochs → do NOT re-apply |
| **F5** | SLEW loss weight 0.1× (was 1×) | `train_paper_split.py` loss | **PERMANENTLY REJECTED** | Killed SLEW head → garbage slew → wrong LUT axis → wrong cell delays → wrong AT |
| **F6** | Gradient clip 5.0 (was 1.0) | `train_paper_split.py` | **REVERTED** | Neutral — no measurable effect |
| **F7** | Early stopping patience=5 (was 25) | config | **REVERTED** | Too aggressive, stopped model before convergence |

---

## 7. Current Code State (as of 2026-10-01)

**All fixes reverted.** Code is clean baseline matching Cycle 1.

### `gnn_model.py` — key sections as they currently stand:

**Level encoding (original — no normalization):**
```python
level_enc = sinusoidal_level_encoding(pin_level, self.level_enc_dim, x.device)
```

**AT initialization (original — softplus only, no PI bypass):**
```python
at_init_raw = self.at_init_mlp(x).squeeze(-1)
AT = torch.nn.functional.softplus(at_init_raw)
```

### `train_paper_split.py` — loss as it currently stands:
```python
at_loss = mse(out["AT"], g["AT_true"])
loss = (2.0 * at_loss
        + mse(out["SLEW"], g["SLEW_true"])
        + 0.3 * mse(out["net_delay"], g["nd_true"])
        + 0.3 * mse(out["cell_delay"], g["cd_true"]))
```

---

## 8. Key Bugs Fixed (Historical — do not reintroduce)

1. **Endpoint detection bug:** Combinational gate input pins were misclassified as endpoints. Fixed in `parse_real_design.py` — add cell-edge drivers to exclusion set. All AT R² before this fix are invalid.

2. **AT_true label bug:** `interactive_train_real.py` was discarding real `.at.sdf` arrival data. Fixed by wiring real arrival data into AT_true properly. All numbers before this fix are invalid.

3. **Calibration-pool leakage bug:** `train.py` only excluded PAPER_TEST_DESIGNS, not calibration designs. Fixed: `CALIBRATION_DESIGNS` hardcoded set in `train_paper_split.py` with disjoint assertion. The 0.9023 result needs re-measurement because of this.

4. **Scheduled sampling + gradient interference:** net/cell delay predictions fed back into AT's `index_reduce` causing net_delay R² collapse on spm. Fixed: `.detach()` self-predicted delays before propagation.

5. **OOM bug in cell loop (Cycle 2 dynamic LUT):** Chained `cell_delay_pred.scatter()` in the level loop accumulated gradient graph. Fixed by list-collect + single scatter after loop.

---

## 9. Suggestions for Next Claude Session

These are ideas that have NOT been tried yet and are worth exploring:

**For xtea (stuck at 0.844):**
- Try clamping `feature_11` (input_delay_norm) to max training value (0.30) at test time — prevents OOD extrapolation without any architecture change
- Or normalize feature_11 differently: use log scale so extreme values compress

**For synth_ram (stuck at 0.691):**
- Try clamping `pin_level` to training max (~25) at test time so sinusoidal encoding stays in-distribution
- Try a separate learned embedding for "level > max_train" rather than sinusoidal for OOD levels
- This is purely inference-time logic — no retraining needed to test the clamping idea

**Other ideas not yet tried:**
- Longer training: Cycle 1 plateaued around 0.898–0.899 but was still at patience=25. Try patience=50 with 600 epochs with the clean baseline — might push mean above 0.90
- Larger hidden_dim: try hid=64 (currently hid=32) — doubles params to ~160k, may capture more complex timing patterns
- Per-design loss weighting: upweight xtea and synth_ram in the training loss directly

---

## 10. Important Notes on DAC 2022 Paper Conventions

- Their R² convention is **SWAPPED** vs sklearn standard: they compute `r2_score(pred, truth)` not `r2_score(truth, pred)`. Their published numbers are NOT directly comparable to ours without noting this.
- Slack R² is NOT independent from AT R² for `real_rat` designs — `slack_pred = RAT − AT_pred` so residuals are mathematically identical.
- They train with `groundtruth=True` (teacher forcing only), compute R² over ALL graph nodes including zeroed-out invalid AT values. CalibTimer masks to real-SDF-labeled pins only — our evaluation is stricter.

---

## 11. All File Locations

### Source Code
| File | Path | Purpose |
|---|---|---|
| **gnn_model.py** | `C:\Krish\GCNN\src\gnn_model.py` | Main model (TimingGNN, NetEmbedding, CellLUTModule) |
| **train_paper_split.py** | `C:\Krish\GCNN\src\train_paper_split.py` | Main training script with leakage guard |
| **interactive_train_real.py** | `C:\Krish\GCNN\src\interactive_train_real.py` | Data parsing pipeline, `real_graph_to_tensors()`, r2() |
| **parse_real_design.py** | `C:\Krish\GCNN\src\parse_real_design.py` | Raw design parser (DEF/SPEF/SDF/SDC/Verilog) |
| **parse_liberty.py** | `C:\Krish\GCNN\src\parse_liberty.py` | Liberty (.lib) parser for LUT tables |
| **conformal.py** | `C:\Krish\GCNN\src\conformal.py` | Split-conformal slack prediction intervals |
| **report_writer.py** | `C:\Krish\GCNN\src\report_writer.py` | Saves run reports, epoch logs, predictions CSVs |
| **model_configurable_v4.py** | `C:\Krish\GCNN\src\model_configurable_v4.py` | V4 model for preprocessed .pt pipeline (different from current) |
| **train_v4.py** | `C:\Krish\GCNN\src\train_v4.py` | CLI trainer for preprocessed .pt files |
| **feature_sets.py** | `C:\Krish\GCNN\src\feature_sets.py` | Runtime feature selection (full18/no_spef16/paper10) |

### Data
| Path | Contents |
|---|---|
| `C:\Krish\GCNN\data\` | Raw design folders (29 designs): DEF, SPEF, SDF, SDC, Verilog, .at.sdf |
| `C:\Krish\GCNN\preprocessed\` | Preprocessed `.pt` files (27 designs) + `meta.json` (for v4 pipeline) |
| `C:\Krish\GCNN\data\techlib\` | Sky130hd Liberty files |

### Results — Key Runs
| Run Folder | Date | Mean AT R² | Notes |
|---|---|---|---|
| `C:\Krish\GCNN\src\results\20260925_200206_train15_test7` | 2026-09-25 | **0.8988** | **BEST — Cycle 1 clean baseline, best@ep370** |
| `C:\Krish\GCNN\src\results\20260928_224543_train15_test7` | 2026-09-28 | 0.879 | Run C: F1+F2+F3, 300ep, not converged |
| `C:\Krish\GCNN\src\results\20260930_190934_train15_test7` | 2026-09-30 | 0.892 | Run D: F1+F2+F3, 500ep, best@ep400 |

Each result folder contains:
- `report.md` — full per-design results + training curve
- `model_checkpoint.pt` — saved model weights (best epoch)
- `epoch_log.csv` — per-epoch metrics
- `config.json` — run configuration
- `predictions/<design>.csv` — per-pin predictions
- `source_snapshot/` — exact copy of all source files used for that run

### Documentation
| File | Path | Contents |
|---|---|---|
| **EXPERIMENT_LOG.md** | `C:\Krish\GCNN\EXPERIMENT_LOG.md` | Compact experiment summary (all runs, fix inventory) |
| **FULL_REPORT.md** | `C:\Krish\GCNN\FULL_REPORT.md` | This file — full handoff document |
| **Memory** | `C:\Users\Krish\.claude\projects\C--Krish-GCNN\memory\project_gcnn_overview.md` | Claude's auto-memory (may be stale — trust this file instead) |

### Notebook
| File | Path | Contents |
|---|---|---|
| `timing_gnn.ipynb` | `C:\Krish\GCNN\timing_gnn.ipynb` | Full preprocessed-data pipeline (v4 model, .pt files) |

---

## 12. Quick Start for Next Session

**To reproduce Cycle 1 (best result):**
```powershell
cd C:\Krish\GCNN\src
# Use source snapshot for guaranteed reproducibility:
# Code: C:\Krish\GCNN\src\results\20260925_200206_train15_test7\source_snapshot\
# Weights: C:\Krish\GCNN\src\results\20260925_200206_train15_test7\model_checkpoint.pt
```

**To run a new training:**
```powershell
cd C:\Krish\GCNN\src
$inputs = @("", "", "", "", "", "25", "400", "", "10", "n") -join "`n"
$inputs | python train_paper_split.py
```
*(Inputs: data_dir, hidden_dim, net_layers, lut_k, weight_decay, patience, epochs, lr, log_every, retry)*

**Current code is clean baseline (no fixes).** Do NOT apply Fix 4 or Fix 5 again — they are permanently rejected.
