# CalibTimer — Pre-Routing Timing Prediction via Graph Neural Networks

> **Predict chip timing before routing — no parasitics, no extraction, well under a second per design.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org)
[![PDK](https://img.shields.io/badge/PDK-sky130hd-green)](https://github.com/google/skywater-pdk)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## What is CalibTimer?

Static Timing Analysis (STA) is how chip designers verify that signals arrive on time at every flip-flop. The problem: traditional STA only works *after* full placement and routing — a process that can take hours. By then, fixing a timing violation is expensive.

**CalibTimer solves this by predicting arrival times and slack directly from the gate-level netlist** — before any routing happens. It builds a directed acyclic graph (DAG) of the circuit where nodes are pins and edges represent wires and gate connections, computes timing with a differentiable timing propagation built on the Liberty tables, and uses a small learned model for the one thing unknown before routing: wire parasitics.

A design-level conformal interval (`src/v2/conformal_v2.py`, calibrated on training designs predicted by models that did not see them) gives ±2.99 ns; on the 7 test designs it finds all 9,580 true setup violations (20,159 endpoint values are flagged in total). The 5 custom designs (labels regenerated with OpenSTA in the benchmark flow) score 0.9979 to 0.9999.

---

## Results — DAC 2022 Benchmark (sky130hd PDK)

Trained on the DAC 2022 training designs, evaluated on the 7 held-out test designs. Arrival time is predicted on all four timing channels (early/late x rise/fall), as in the DAC 2022 paper.

| Design | Type | DAC'22 paper | CalibTimer v2 |
|:---|:---|:---:|:---:|
| spm | Serial Multiplier | 0.8975 | **0.9999** |
| xtea | Block Cipher | 0.9135 | **0.9999** |
| aes192 | AES Crypto | 0.8605 | **0.9993** |
| synth_ram | SRAM Macro | 0.8656 | **0.9964** |
| usbf_device | USB Interface | 0.9252 | **0.9961** |
| y_huff | Huffman Coder | 0.9256 | **0.9931** |
| jpeg_encoder | JPEG Codec | 0.8820 | **0.9918** |
| **Mean** | | 0.8957 | **0.9966** |

> AT R2 over all non-source pins and all four channels, against OpenSTA with routed parasitics.
> Final run (src/v2/runs/20261112_152424_final): test mean 0.9966, lowest design 0.9918. About 52 k parameters, ~4 minutes of training on a laptop GPU.
> Earlier runs of the same pipeline gave test means between 0.9966 and 0.9977.

Notes on the comparison:
- Primary-input arrival time and slew are taken as given, as in the DAC 2022 code.
- Mean endpoint AT R2 is 0.92; the weak ones are y_huff (0.81) and synth_ram (0.71), and synth_ram setup slack R2 is negative because its slack varies very little.
- Swapped split (fit on these 7, test on the other 14, run before the last parser fixes): 13 designs at 0.995 or above; `des` scores 0.43 because of one net with a 10 ns wire delay that placement does not reveal.
- An earlier version of this repo reported 0.8988. That number used a label that averaged the four channels and is not comparable.

---

## How it Works

Before routing, timing is exact physics except for the wires. CalibTimer v2 therefore computes arrival times the way a timer does and learns only the wire part.

```
at.sdf structure + DEF/LEF placement + Liberty (both corners) + SDC
        |  build_v2.py
4-channel timing graph  (pins, net edges, cell arcs with sense and table ids)
        |  tree_v2.py        spanning tree per net -> length and Elmore terms
        |  WireNet (train_v2.py)   learned from placement only:
        |      wire cap per net, wire delay per edge, effective-cap factor per arc
        |  sta_v2.py         Liberty table lookup + min/max level-by-level propagation
AT and slew on [early-rise, early-fall, late-rise, late-fall]
```

| Component | What it does |
|:---|:---|
| **Differentiable STA** | Bilinear NLDM lookup at (input slew, load); min for early, max for late; rise/fall linked by arc sense |
| **Wire cap** | cap per um x spanning-tree length x learned correction (label: SPEF) |
| **Wire delay** | root resistance x net cap + tree Elmore x learned correction (label: SDF) |
| **Effective cap** | shielding factor from wire vs driver resistance; applied to delay, not slew |
| **Two-stage training** | local teacher-forced losses, then end-to-end fine-tuning through the propagation |

With the true SDF delays the propagation reproduces OpenSTA arrival times with R2 1.0000, which is the check that the graph is right.

---

## Repository Structure

```
src/v2/                        <- current pipeline  [START HERE]
  liberty_v2.py                Liberty parser (ff + ss corners, table bank)
  build_v2.py                  design -> 4-channel timing graph
  tree_v2.py                   placement-only routing-tree estimate
  sta_v2.py                    differentiable timing propagation + oracle checks
  train_v2.py                  wire model training and evaluation
  eval_v2.py                   test / training evaluation, per-pin predictions
  conformal_v2.py              design-level interval from leave-two-out models
  regen_labels.py              regenerate a design's label file with OpenSTA (benchmark flow)
  diag_v2.py                   swap predicted wire quantities for true ones
  runs/                        results.json + checkpoint per run

src/*.py                       earlier single-channel pipeline (kept for history)
EXPERIMENT_LOG.md              log of all runs
FULL_REPORT.md                 detailed handoff document (section 0 = v2)
Krish_Internship_Work/         reports
```

---

## Setup

```bash
pip install torch numpy

# Place the sky130hd design data under data/  (see the v1.0 release assets)
cd src/v2
python sta_v2.py xtea                                   # oracle checks on one design
python train_v2.py --epochs 300 --e2e_epochs 12 --tag e2e
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
| Effective capacitance | Qian, Pullela, Pillage (1994); Dartu, Menezes, Pileggi (1996) |
| Elmore delay | Elmore (1948) |
| Split-conformal prediction | Angelopoulos & Bates (2021) |

---

*IIIT Vadodara Summer Internship 2026 · Krish Bhavsar (230130111040)*
*Supervisor: Dr. Kamal Kishor Jha*
