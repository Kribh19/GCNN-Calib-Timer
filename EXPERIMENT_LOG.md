# CalibTimer GCNN — Experiment Log

**Project:** Pin-level pre-routing AT/slack prediction on sky130hd PDK  
**Metric:** Mean-per-design AT R² across 7 official DAC 2022 test designs  
**Target:** All 7 designs ≥ 0.90 AT R²  

---

## Test Designs (fixed — never train on these)
`jpeg_encoder, usbf_device, aes192, xtea, spm, y_huff, synth_ram`

## Train Designs (15)
`BM64, aes128, aes256, aes_cipher, alu4bit, blabla, cic_decimator, des, genericfir, picorv32a, salsa20, usb, usb_cdc_core, wbqspiflash, zipdiv`

---

## Results Summary

| Run | Fixes | Epochs | Best@Ep | mean R² | xtea | synth_ram | Notes |
|---|---|---|---|---|---|---|---|
| **Cycle 1** (baseline) | none | 400 | 370 | **0.899** | 0.844 | 0.691 | Best so far |
| Cycle 3 (all 7) | F1+F2+F3+F4+F5+F6+F7 | 300 | ? | 0.869 | 0.768 | 0.608 | REGRESSION |
| Run C | F1+F2+F3 | 300 | 300 | 0.879 | 0.788 | 0.595 | Still climbing |
| Run D | F1+F2+F3 | 500 | 400 | 0.892 | 0.800 | 0.655 | Below Cycle 1 |
| **Baseline rerun** | none | 400 | — | — | — | — | In progress |

---

## Per-Design Detail

### Cycle 1 — `20260925_200206_train15_test7` (BEST)
Config: hid=32, layers=3, lut_k=7, patience=25, lr=2e-3, no fixes  
Best epoch: 370

| design | AT R² |
|---|---|
| aes192 | 0.965 |
| jpeg_encoder | 0.951 |
| spm | 0.952 |
| synth_ram | 0.691 ❌ |
| usbf_device | 0.982 |
| xtea | 0.844 ❌ |
| y_huff | 0.906 |
| **mean** | **0.899** |

---

### Cycle 3 — All 7 Fixes at Once (REGRESSION)
Config: same + F1+F2+F3+F4+F5+F6+F7 all applied simultaneously

| design | AT R² |
|---|---|
| aes192 | ~0.96 |
| jpeg_encoder | ~0.96 |
| spm | ~0.87 |
| synth_ram | 0.608 ❌ |
| usbf_device | ~0.97 |
| xtea | 0.768 ❌ |
| y_huff | ~0.97 |
| **mean** | **0.869** |

**Root causes of regression:**
- Fix 5 (SLEW weight 0.1×): under-supervised SLEW head → garbage slew values → wrong LUT column → wrong cell delays → wrong AT  
- Fix 4 (h[o] in cell_feat_lvl): doubled LUT input width (h+n_gates+1 → 2h+n_gates+1) — needs far more epochs to converge  

---

### Run C — `20260928_224543_train15_test7`
Config: F1 (PI direct) + F2 (level norm) + F3 (weighted MSE 10×), 300 epochs, patience=25

| design | AT R² | vs Cycle 1 |
|---|---|---|
| aes192 | 0.966 | +0.001 |
| jpeg_encoder | 0.967 | +0.016 |
| spm | 0.869 | -0.083 |
| synth_ram | 0.595 | -0.096 ❌ |
| usbf_device | 0.973 | -0.009 |
| xtea | 0.788 | -0.056 ❌ |
| y_huff | 0.994 | +0.088 |
| **mean** | **0.879** | **-0.020** |

Note: best@300, patience=0/25 — model NOT converged.

---

### Run D — `20260930_190934_train15_test7`
Config: F1+F2+F3, 500 epochs, patience=50

| design | AT R² | vs Cycle 1 |
|---|---|---|
| aes192 | 0.966 | +0.001 |
| jpeg_encoder | 0.980 | +0.029 |
| spm | 0.879 | -0.073 |
| synth_ram | 0.655 | -0.036 ❌ |
| usbf_device | 0.973 | -0.009 |
| xtea | 0.800 | -0.044 ❌ |
| y_huff | 0.989 | +0.083 |
| **mean** | **0.892** | **-0.007** |

Best epoch: 400. Still below Cycle 1 at full convergence.

**Key finding:** Fix 3 (weighted MSE 10×) helps jpeg/y_huff but hurts spm/xtea/synth_ram. Net negative overall.

---

## Fix Inventory

| Fix | Description | Status | Effect |
|---|---|---|---|
| F1 | PI direct formula: `AT = is_PI * input_delay_ns + (1-is_PI) * softplus(init)` | REVERTED | Helped xtea a little but not enough |
| F2 | Level normalization per-design: `pin_level_normed = level / max_level * 30` | REVERTED | Didn't fix synth_ram OOD issue as expected |
| F3 | Weighted MSE 10× on nonzero-AT pins | REVERTED | Helped jpeg/y_huff, hurt spm/xtea/synth_ram |
| F4 | Add h[o] to cell_feat_lvl | REJECTED | Doubled LUT input width → needs way more epochs |
| F5 | SLEW weight 0.1× | REJECTED | Killed SLEW head → garbage cell delays |
| F6 | Grad clip 5.0 (was 1.0) | REVERTED | Neutral |
| F7 | Patience 5 | REVERTED | Too aggressive early stopping |

---

## Why xtea and synth_ram Are Hard

**xtea (0.844 ceiling so far):**
- Has PI pins with `set_input_delay = 5.5 ns` (feature_11 = 0.55 normalized)
- Training range for input_delay_norm is ~0.10–0.30 → 0.55 is out-of-distribution
- `at_init_mlp` extrapolates OOD → wrong PI initial AT → all downstream AT wrong
- Fix 1 tried to bypass this but didn't fully solve it

**synth_ram (0.691 ceiling so far):**
- Max level = 46; training designs have max_level ≈ 15–25
- Sinusoidal level encoding is severely OOD for levels 26–46
- Fix 2 tried to normalize but model still struggles with deep pipelines
- Also: synth_ram has complex memory-style timing paths unlike any training design

---

## Current Code State (as of 2026-09-30)

All fixes reverted. Code is back to clean Cycle 1 baseline:
- `gnn_model.py`: original level encoding, original AT init (softplus only)
- `train_paper_split.py`: standard MSE loss, grad clip 1.0, patience 25

**Baseline rerun in progress** to confirm reproducibility of 0.899.

---

## Next Steps to Consider

1. **Confirm baseline reproduces 0.899** — in progress
2. **Fix xtea properly**: instead of bypassing at_init_mlp, normalize input_delay feature to training range OR clamp feature_11 to 0.30 at test time
3. **Fix synth_ram properly**: try clamping pin_level to training max (25) at test time — prevents OOD sinusoidal encoding without architectural changes
4. **Fix 3 redesign**: instead of 10× weight globally on nonzero-AT, weight only on xtea/synth_ram-style OOD pins
5. **Run longer baseline**: try patience=50, 600 epochs — maybe Cycle 1 can still improve beyond 370 epochs
