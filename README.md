# CalibTimer — Pre-Routing Timing Prediction via Graph Neural Networks

> **Predict chip timing before routing — no parasitics, no extraction, under 1 ms per design.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org)
[![PDK](https://img.shields.io/badge/PDK-sky130hd-green)](https://github.com/google/skywater-pdk)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## What is CalibTimer?

Static Timing Analysis (STA) is how chip designers verify that signals arrive on time at every flip-flop. The problem: traditional STA only works *after* full placement and routing — a process that can take hours. By then, fixing a timing violation is expensive.

**CalibTimer solves this by predicting arrival times and slack directly from the gate-level netlist** — before any routing happens. It builds a directed acyclic graph (DAG) of the circuit where nodes are pins and edges represent wires and gate connections, then runs a Graph Neural Network (GNN) that mimics STA's level-by-level timing propagation. The result: accurate timing estimates in milliseconds, using only the netlist topology and estimated wire lengths.

On top of point predictions, CalibTimer also produces **statistically guaranteed slack intervals** via split-conformal prediction — so designers get not just a number, but a rigorous uncertainty bound.

---

## Results — DAC 2022 Benchmark (sky130hd PDK)

Trained on 15 open-source designs, evaluated on 7 completely held-out test designs:

| Design | Type | AT R² |
|:---|:---|:---:|
| usbf_device | USB Interface | **0.9824** |
| aes192 | AES Crypto | **0.9654** |
| spm | Serial Multiplier | **0.9515** |
| jpeg_encoder | JPEG Codec | **0.9513** |
| y_huff | Huffman Coder | **0.9063** |
| xtea | Block Cipher | 0.8441 |
| synth_ram | SRAM Macro | 0.6909 |
| **Mean** | | **0.8988** |

> AT R² measures how well predicted arrival times match OpenSTA ground truth across all endpoint pins.
> Best checkpoint: **epoch 370 / 500**, ~75 k parameters.

---

## How it Works

```
Placed Netlist (Verilog + DEF)
        ↓  parse_real_design.py
Heterogeneous DAG  (pins = nodes, net edges + cell edges)
        ↓  NetEmbedding  (3-layer message passing + virtual global node)
Pin Embeddings  h(N×32)  +  net delay per wire
        ↓  Level-ordered propagation (level 0 → L_max)
            ├─ Net edges:  AT[sink] = max(AT[driver] + wire_delay)
            └─ Cell edges: delay from Liberty LUT (Kronecker interpolation)
        ↓
AT, SLEW, net_delay, cell_delay  per pin/edge
        ↓  conformal.py
Slack prediction intervals  (90% coverage guarantee, no assumptions)
```

### Key architectural ideas

| Component | What it does |
|:---|:---|
| **NetEmbedding** | 3 rounds of net-edge message passing with residual connections |
| **Sinusoidal level encoding** | NeRF/Transformer-style encoding of topological depth — fixes error accumulation on deep DAGs |
| **Virtual node** | One global learned node connected to every pin; gives any pin a 2-hop path to any other pin's info |
| **CellLUTModule** | Kronecker-product interpolation over Liberty 7×7 cell-delay tables, conditioned on live propagated input slew |
| **at_init_mlp** | Learned initial arrival time per pin — critical for primary inputs (otherwise frozen at 0 forever) |
| **Teacher forcing + scheduled sampling** | Training uses true delays early, transitions to self-prediction — stabilizes convergence on small designs |
| **Gradient detach** | Prevents AT's sparse backprop from corrupting the delay-head MSE supervision |

---

## Repository Structure

```
src/
├── gnn_model.py               ← TimingGNN, NetEmbedding, CellLUTModule  [START HERE]
├── train_paper_split.py       ← Training script (DAC 2022 split, multi-task loss)
├── parse_real_design.py       ← Netlist → heterogeneous graph parser
├── interactive_train_real.py  ← real_graph_to_tensors() and DEVICE
├── run_calibration_inference.py ← Inference on new designs
├── conformal.py               ← Split-conformal slack intervals
└── report_writer.py           ← Saves per-epoch metrics and checkpoints

Krish_Internship_Work/
├── CalibTimer_Technical_Report.pdf          ← Full technical report (architecture + results)
├── Krish_Bhavsar_..._Internship_Report.pdf  ← IIIT Vadodara internship report
└── Summer_Internship_2026_Complete.pdf

EXPERIMENT_LOG.md              ← Compact log of all training runs
FULL_REPORT.md                 ← Detailed handoff document
conformal_results.csv          ← Conformal calibration output
```

---

## Setup

```bash
# Install dependencies
pip install torch torchvision torch-geometric

# Place the sky130hd design data under data/
# (download link in Releases — see the v1.0 release assets)

# Train on the DAC 2022 paper split (15 designs, 7 held-out test)
cd src
python train_paper_split.py
# → prompts for config; press Enter to accept defaults (hidden_dim=32, 500 epochs)

# Run inference on new designs
python run_calibration_inference.py \
  --run_dir results/<your_run_folder> \
  --data_dir ../data \
  --calib_designs gcd

# Compute conformal slack intervals
python conformal.py --predictions results/<run>/predictions --alpha 0.1
```

---

## Dataset

The `data/` directory (8.3 GB) contains 22 open-source designs synthesized with **OpenROAD** on the **SkyWater sky130hd PDK**, with reference timing from **OpenSTA**. It is not included in the git history due to size.

**Download:** see the [Releases](https://github.com/Kribh19/GCNN-Calib-Timer/releases) tab — the full dataset is attached as a compressed archive to the v1.0 release.

---

## Papers and References

| Idea | Source |
|:---|:---|
| Dataset split, E2E timing unrolling | DAC 2022 — *Pre-Routing Timing Prediction* |
| Sinusoidal positional encoding | Vaswani et al., *Attention Is All You Need* (2017) |
| Level encoding for timing GNNs | PreRoutGNN, AAAI 2024 |
| Split-conformal prediction | Angelopoulos & Bates (2021) |
| Virtual node augmentation | Gilmer et al. (2017), Hu et al. OGB (2020) |

---

*IIIT Vadodara Summer Internship 2026 · Krish Bhavsar (230130111040)*
*Supervisor: Dr. Kamal Kishor Jha*
