# Cytosim Epoch-Based Membrane Protrusion Simulation

**Implementing the motor–microtubule driven protrusion model of Oelz, del Castillo, Gelfand & Mogilner (2018)**

> Oelz et al. *Biophys. J.* 115:1614–1624 (2018)

---

## Overview

Standard Cytosim simulates cytoskeletal mechanics within a **fixed** spatial boundary. This repository implements an **epoch-based Python orchestration layer** that iteratively deforms the cell boundary in response to microtubule (MT) confine forces, reproducing the cooperative kinesin–dynein–MT mechanism of directed neurite growth described in Oelz et al. (2018).

Each epoch:
1. Runs a full Cytosim simulation (1 s, 20 steps)
2. Reads MT confine forces via the standalone `report` binary
3. Applies a five-stage protrusion rule to deform the boundary polygon
4. Carries forward MT and motor positions to the next epoch

Over 3600 epochs (≈ 1 hour simulated time), stable finger-like protrusions of 5–8 µm form, matching the paper benchmark, alongside microtubule polarity sorting readout inside process sectors.

---

## Repository structure

```
Cytosim_simulation/
│
├── run_protrusion_v4.py          # Main Python orchestration script
├── oelz2018_epoch_v4.cym.template  # Cytosim config template (filled each epoch)
│
├── output_v4/                    # Generated outputs (created at runtime)
│   ├── boundary_evolution_v4.csv
│   ├── membrane_forces_v4.csv
│   ├── process_growth_v4.csv
│   ├── perimeter_area_v4.csv
│   ├── dynein_positions_v4.csv
│   ├── end_polarity_v4.csv
│   ├── summary_v4.txt
│   └── protrusion_report_v4.png  # 6-panel summary figure
│
└── epoch_files_v4/               # Per-epoch Cytosim files (created at runtime)
    ├── epoch_0000/
    │   ├── epoch_0000.cym
    │   ├── polygon_0000.txt
    │   └── objects.cmo
    └── ...
```

---

## Requirements

### Cytosim
Build Cytosim from source or use a pre-compiled binary:
```
https://gitlab.com/f-nedelec/cytosim
```
The simulation expects two binaries:
- `sim` — runs the Cytosim simulation
- `report` — extracts force and motor position data

### Python
Python 3.8+ with the following packages:
```
numpy
pandas
matplotlib
scipy
```

Install with:
```bash
pip install numpy pandas matplotlib scipy
```

---

## Configuration

Edit the **PATHS** section at the top of `run_protrusion_v4.py` to match your system:

```python
CYTOSIM_BIN = "/path/to/cytosim/bin/sim"
REPORT_BIN  = "/path/to/cytosim/bin/report"
WORK_DIR    = "/path/to/epoch_v4"
TEMPLATE    = os.path.join(WORK_DIR, "oelz2018_epoch_v4.cym.template")
EPOCH_DIR   = os.path.join(WORK_DIR, "epoch_files_v4")
OUT_DIR     = os.path.join(WORK_DIR, "output_v4")
```

Place the `.cym.template` file in `WORK_DIR`.

---

## Running the simulation

```bash
cd /path/to/epoch_v4
python3 run_protrusion_v4.py
```

Progress is printed to stdout each epoch:

```
==================================================================
  EPOCH   1/ 60  t=0-50s  N_dyn=15  perim=62.8um  n_procs=0
==================================================================
    [report] frames=[4]  confine_force ...
    [confine] denom=1  avg_max=2.341 pN
  active=3  max_F=2.34pN  max_ext=0.450um  n_procs=0
```

At completion, figures and CSVs are written to `output_v4/`.

---

## Key tunable parameters

| Parameter | Default | Description |
|---|---|---|
| `N_EPOCHS` | 60 | Number of simulation epochs |
| `DT_EPOCH` | 50.0 s | Simulated time per epoch |
| `N_STEPS` | 1000 | Cytosim integration steps per epoch |
| `N_MT` | 150 | Number of microtubules |
| `N_KINESIN` | 200 | Number of kinesin-1 couples |
| `DYNEIN_DENSITY` | 0.25 /µm | Cortical dynein density |
| `FP_MEMBRANE` | 2.0 pN | Protrusion force threshold (flat membrane) |
| `FP_PROCESS` | 1.0 pN | Protrusion force threshold (inside process) |
| `STEP` | 0.15 µm | Radial displacement per active vertex per epoch |
| `SMOOTH_SIGMA` | 0.5 vertices | Gaussian smoothing of force field |
| `BOUNDARY_SMOOTH_SIGMA` | 2.0 vertices | Post-protrusion boundary smoothing |
| `RADIAL_FRACTION_MIN` | 0.3 | Minimum cos(angle) to accept a confine force |
| `MAX_PROCESSES` | 4 | Maximum simultaneous processes (set `ENABLE_PROCESS_CAP=False` to disable) |
| `PROCESS_TIP_BOOST` | 1.5 pN | Threshold reduction at existing process tips |
| `DYNEIN_PROCESS_ENRICHMENT` | 3.0× | Fold-enrichment of new dyneins inside processes |
| `CARRY_DYNEIN_POSITIONS` | True | Re-inject dynein positions from previous epoch |
| `CARRY_KINESIN_POSITIONS` | True | Re-inject kinesin positions from previous epoch |

### Protrusion width tuning

| Goal | `SMOOTH_SIGMA` | `FP_MEMBRANE` | `STEP` |
|---|---|---|---|
| Narrow finger-like | 0.3–0.5 | 2.0–2.5 pN | 0.1–0.2 µm |
| Broad lamellipodia | 2.0–3.0 | 1.0–1.5 pN | 0.5–1.0 µm |

---

## Motor models

### Kinesin-1 (`set couple kinesin1`, `activity = crosslink`)
- Walks toward MT **plus end** at +0.57 µm/s, stall force 4.7 pN
- Slides antiparallel MT pairs apart, pushing minus ends outward
- Positions carried forward each epoch from `kinesin1:state` report

### Cortical dynein (`set single dynein_cortex`, `activity = fixed`)
- Anchor permanently fixed at cortex; motor walks toward MT **minus end** at −0.86 µm/s, stall force 1.36 pN, binding reach 0.75 µm
- Transports MTs plus-end-forward, driving both protrusion and polarity sorting
- Positions placed explicitly by Python at boundary vertices; carried forward from `dynein_cortex:position` report

---

## MT carry-forward

At epoch > 0, MT node positions from `fiber:point` are re-injected using Cytosim's local-frame convention:

```
s_i = a_i − a_0      (shape in local frame)
position = a_0        (global translation)
→ final position: s_i + a_0 = a_i  ✓
```

Setting `N_MT = 0` in the template prevents double-placement.

---

## Output panels

The script generates a 6-panel summary figure `protrusion_report_v4.png`:

| Panel | Content |
|---|---|
| P1 | Boundary evolution — epoch snapshots overlaid, plasma colourmap by time |
| P2 | Per-process extension vs time; 5 µm paper benchmark shown |
| P3 | Perimeter and enclosed area over all epochs |
| P4 | Plus/minus MT end distribution inside process sectors |
| P5 | Final boundary with process regions, dynein anchors (●), process tips (★) |
| P6 | Polar confine-force rose — last N epochs overlaid |

---

## Simulation cases

Three configurations are studied:

| Case | Kinesin | Dynein | Growth | Polarity sorting |
|---|---|---|---|---|
| Kinesin + Dynein | ✓ | ✓ | Narrow focused protrusions | Plus-end-out (axonal) ✓ |
| Kinesin only | ✓ | ✗ | Broad unfocused protrusions | Minus-end-out (inverted) ✗ |
| Dynein only | ✗ | ✓ | No protrusion | — |

---

## Parameter source

All motor and MT parameters are taken from Tables S1–S2 of:

> Oelz, D., del Castillo, U., Gelfand, V. I. & Mogilner, A. (2018).
> Microtubule dynamics, kinesin-1 sliding, and dynein action drive growth of cell processes.
> *Biophys. J.* **115**(8), 1614–1624.

---


## Acknowledgements

- Cytosim engine: F. Nédélec & D. Foethke ([gitlab.com/f-nedelec/cytosim](https://gitlab.com/f-nedelec/cytosim))
- Model parameters: Oelz, del Castillo, Gelfand & Mogilner (2018)
- Supervisor: Dr. Chaitanya A. Athale, Division of Biology, IISER Pune
