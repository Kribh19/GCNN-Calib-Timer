# CalibTimer — Pre-Routing Static Timing Analysis via Graph Neural Networks

Pre-routing timing prediction for digital IC designs using a heterogeneous GNN trained on the **DAC 2022 sky130hd benchmark**.

**Best result:** Mean AT R² = **0.8988** across 7 held-out test designs (DAC 2022 split), epoch 370.

---

## What it does

CalibTimer predicts per-pin **arrival times (AT)** and **slack** from a placed gate-level netlist — before routing, without parasitics — using a graph neural network that unrolls the STA recurrence level-by-level through the timing DAG.

It also produces **coverage-guaranteed slack prediction intervals** via split-conformal prediction (90% marginal coverage, no distributional assumptions).

---

## Per-design AT R² (7 test designs, DAC 2022 split)

| Design | Category | AT R² |
|---|---|---|
| usbf_device | Interface | 0.9824 |
| aes192 | Crypto | 0.9654 |
| spm | Arithmetic | 0.9515 |
| jpeg_encoder | Multimedia | 0.9513 |
| y_huff | Compressor | 0.9063 |
| xtea | Crypto | 0.8441 |
| synth_ram | Memory | 0.6909 |
| **Mean** | | **0.8988** |

---

## Architecture highlights

- **Heterogeneous DAG**: pins as nodes, net edges (driver→sink) + cell edges (gate input→output)
- **NetEmbedding**: 3-layer net-edge message passing with sinusoidal topological level encoding and a virtual global node
- **CellLUTModule**: Kronecker-product interpolation over Liberty 7×7 LUT tables, conditioned on propagated input slew
- **at_init_mlp**: learned initial arrival time per node (critical for primary-input pins)
- **Level-ordered AT/SLEW propagation** with teacher forcing and scheduled sampling during training
- **Split-conformal slack intervals**: distribution-free coverage guarantee using a calibration design

---

## Key files

```
src/
  gnn_model.py               # TimingGNN, NetEmbedding, CellLUTModule
  train_paper_split.py       # Training script (DAC 2022 split, multi-task loss)
  parse_real_design.py       # Netlist → heterogeneous graph
  interactive_train_real.py  # real_graph_to_tensors()
  run_calibration_inference.py
  conformal.py               # Split-conformal slack intervals

Krish_Internship_Work/
  CalibTimer_Technical_Report.pdf   # Full technical report
```

---

## Setup

```bash
pip install torch torch-geometric
# Place DAC 2022 sky130hd design folders under data/
cd src
python train_paper_split.py
```

---

## Dataset

Uses the open **DAC 2022 benchmark** designs synthesized with OpenROAD on the **SkyWater sky130hd PDK**.  
The `data/` directory (8 GB) is excluded from this repo due to size — contact the project maintainer or refer to the DAC 2022 paper for the dataset.

---

## Citation / Reference

Based on the E2E-Timing framework (DAC 2022). Level encoding follows the sinusoidal scheme from  
Vaswani et al. (Transformer, 2017). Conformal prediction follows Angelopoulos & Bates (2021).

---

*IIIT Vadodara Summer Internship 2026 · Krish Bhavsar (230130111040)*
