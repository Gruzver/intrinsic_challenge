# AIC Vision Policy

**Vision-based fiber optic connector insertion for the AI for Industry Challenge**

[![ROS2 Kilted](https://img.shields.io/badge/ROS2-Kilted-blue?logo=ros)](https://docs.ros.org/en/kilted/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n--pose-darkgreen?logo=python)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-solvePnP-red?logo=opencv)](https://opencv.org/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-yellow?logo=python)](https://python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

This repository contains the vision-based policy developed for the **[AI for Industry Challenge (AIC)](https://github.com/intrinsic-dev/aic)** organized by Intrinsic. The challenge requires a UR5e robot arm to autonomously insert fiber optic connectors (SFP and SC) into a task board using only onboard cameras and wrist F/T sensing — no ground-truth pose information.

The policy achieves a score of **~224 / 300** in the qualification phase across 3 trials (2× SFP, 1× SC).

### Challenge Setup

- **Robot**: Universal Robots UR5e with impedance control interface
- **Connectors**: SFP (13.5 × 8.8 mm) and SC (25.8 × 10.8 mm) fiber optic plugs
- **Sensors**: 3 RGB cameras (left / center / right) + wrist wrench sensor
- **Evaluation**: 3 trials scored on insertion success, approach smoothness, and insertion force

---

## Pipeline

The policy runs in three sequential phases per trial:

```
Phase A — Localize
  ┌─────────────────────────────────────────────────────┐
  │  3 cameras → YOLOv8n-pose → keypoints (px)          │
  │  solvePnP (IPPE) per camera → 6-DOF port pose       │
  │  Multi-camera fusion (weighted avg by reproj error)  │
  │  Continuous approach: EMA-smoothed pose tracking     │
  │  Multi-shot refinement: 20 static shots at standoff  │
  └─────────────────────────────────────────────────────┘
              ↓
Phase B — Descent
  ┌─────────────────────────────────────────────────────┐
  │  Reduce standoff 50mm → 0mm at 2mm/step             │
  │  Live solvePnP refinement every 4 steps             │
  │  EMA pose correction (XY frozen for SC)             │
  └─────────────────────────────────────────────────────┘
              ↓
Phase C — Insert
  ┌─────────────────────────────────────────────────────┐
  │  Expanding spiral in port XY plane (r max = 8mm)    │
  │  Push along port axis at 1mm/step                   │
  │  Snap detection via wrist F/T delta                 │
  └─────────────────────────────────────────────────────┘
```

---

## Key Technical Contributions

### 1. YOLO-Pose port detection
A YOLOv8n-pose model trained on auto-labeled data (ground-truth TF at dataset collection time) detects SFP and SC ports with 8 keypoints each — 4 entrance face corners (used for solvePnP) and 4 back-face corners. The model generalizes across all 3 cameras from a single `best.pt` checkpoint.

### 2. solvePnP 6-DOF pose estimation
Each camera independently runs `cv2.solvePnPGeneric` with the IPPE solver on the 4 entrance-face keypoints against a known 3D model of the port. The best IPPE solution is selected by port-axis direction (entrance faces upward). Valid estimates (reproj error ≤ 15px, depth ≤ 0.8m) are fused across cameras via inverse-reproj-error weighting and geodesic SVD rotation averaging.

### 3. Multi-shot pose refinement
At standoff (5 cm from port), the arm holds static while 20 independent solvePnP measurements are collected and averaged. This reduces random estimation noise by ~√20 ≈ 4.5×, improving alignment from ~10mm to ~2-4mm error for SC ports.

### 4. Impedance-controlled insertion
Insertion uses per-plug-type stiffness: SFP at 200 N/m (avoids force spikes at snap) and SC at 300 N/m (compensates unfavorable Jacobian in the extended configuration). Snap detection uses plug-type-aware force sign convention.

---

## Port Frame Convention

```
Port entrance face (Z=0 plane):

  KP0 ──── KP1          +Y (up)
   |        |             │
   |   +Z   |     ────────┼──── +X (right)
   |  →out  |             │
  KP3 ──── KP2          -Z into port

  T_base_port[:, 2] = outward normal (approach direction = -Z)
```

3D model points (SFP):
```
KP0 TL = (-0.00675, +0.00440, 0)    KP1 TR = (+0.00675, +0.00440, 0)
KP3 BL = (-0.00675, -0.00440, 0)    KP2 BR = (+0.00675, -0.00440, 0)
```

---

## Repository Structure

```
my_vision_policy/
├── my_vision_policy/
│   ├── ros/
│   │   ├── VisionPolicy.py        # Main policy: phases A, B, C
│   │   ├── DataCollectionPolicy.py# Auto-labeling data collection
│   │   └── DebugGroundTruth.py    # GT comparison logging
│   └── vision/
│       ├── yolo_detector.py       # YOLOv8n-pose wrapper
│       ├── port_pose.py           # solvePnP + multi-camera fusion
│       ├── connector.py           # TCP↔plug geometry, approach pose
│       ├── localizer.py           # Camera frame constants
│       └── trajectory.py         # Smoothstep interpolation
├── weights/
│   └── best.pt                   # Trained YOLOv8n-pose weights (see below)
├── docker/
│   └── my_vision_policy/
│       ├── Dockerfile
│       └── entrypoint.sh
├── setup.py
└── package.xml
```

> **Model weights**: `best.pt` (6.3 MB) is hosted on [Google Drive / HuggingFace — link TBD]. Place it at `my_vision_policy/weights/best.pt` before running.

---

## Installation

This package runs inside the AIC evaluation environment using [pixi](https://pixi.sh/).

### Prerequisites

- [AIC base environment](https://github.com/intrinsic-dev/aic) set up with pixi
- The `best.pt` weights file placed at `my_vision_policy/weights/best.pt`

### Build

```bash
cd ~/ws_aic/src/aic
pixi reinstall ros-kilted-my-vision-policy
```

### Run (simulation)

**Terminal 1 — Simulator (distrobox):**
```bash
distrobox enter -r aic_eval
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

**Terminal 2 — Policy:**
```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=my_vision_policy.ros.VisionPolicy
```

### Docker (submission)

```bash
# Build
docker build -f docker/my_vision_policy/Dockerfile -t my-solution:v1 .

# Run locally
docker-compose -f docker/docker-compose.yaml up
```

---

## Results

| Trial | Connector | Score | Notes |
|-------|-----------|-------|-------|
| 1 | SFP port 0 (mount_0) | 75 / 75 | Full insertion ✅ |
| 2 | SFP port 0 (mount_1) | 50 / 75 | Full insertion ✅ |
| 3 | SC port base | ~39 / 75 | Partial insertion |
| **Total** | | **~224 / 300** | Qualification phase |

Tier-2 bonus (approach smoothness): ~57 pts · Tier-1 bonus: ~3 pts

---

## Dataset Collection

Training data was auto-labeled using ground-truth TF frames available in the simulator:

```bash
# With ground_truth:=true, run DataCollectionPolicy to capture images
# Then run the auto-labeling tool:
python tools/collect_dataset.py
```

The auto-labeler projects the 3D port corner model into each camera using the exact TF pose, producing pixel-accurate YOLO-Pose labels without manual annotation.

---

## Challenge Context

The [AI for Industry Challenge](https://github.com/intrinsic-dev/aic) is an international robotics competition organized by Intrinsic (Google DeepMind). Teams develop autonomous manipulation policies for a UR5e robot that must insert fiber optic connectors into a task board in a controlled industrial setting.

- **No ground truth** at evaluation time — full perception pipeline required
- **1 submission per day** during the qualification phase
- **Scoring**: geometric insertion check + approach force/jerk penalties

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## Acknowledgments

- [Intrinsic / AI for Industry Challenge](https://github.com/intrinsic-dev/aic) — challenge framework and evaluation environment
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — pose estimation model
- [OpenCV solvePnP](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — 6-DOF pose from 2D-3D correspondences
