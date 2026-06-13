# MoE On-Device Appliance Detection — with Device Naming & Memory

On-device Mixture-of-Experts (MoE) appliance detector for the SmartThings
ecosystem. A lightweight scene **router** dispatches each frame to **one** of
three specialist detectors (climate / kitchen / display), and a new
**device-naming + re-identification** layer lets a user rename any detected
appliance and have that name persist — recognised again whenever the camera
returns to it. This is the in-miniature version of the target SmartThings
feature: *point the camera at an appliance → recognise which one → open its
settings.*

---

## What's in this repo

| File | Role |
| --- | --- |
| **`demo_onnx.py`** | **Main entry point.** Fast ONNX-Runtime demo with the naming feature. Run this. |
| `onnx_detector.py` | ONNX-Runtime detector; drop-in for rfdetr's `.predict()` (returns `supervision.Detections`). |
| `export_onnx.py` | One-time export of the experts to ONNX (`weights/climate.onnx`, `weights/coco.onnx`). |
| `probe_onnx.py` | Verifies the ONNX decode matches the original model. |
| `device_memory.py` | The re-identification engine + persistent gallery (`device_memory.json`). |
| `demo.py` | Original eager-PyTorch demo (also has the naming feature; slower — kept as reference). |

---

## How it works

```
                       ┌──────────────┐
   frame ───────────►  │ ResNet18     │  picks ONE expert
                       │ scene router │  (climate / kitchen / display)
                       └──────┬───────┘
                              ▼
              ┌───────────────────────────────┐
              │  active expert (RF-DETR Nano)  │  → bounding boxes
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │  device memory (re-ID)         │  box → 512-d fingerprint
              │  match against saved gallery   │  → swap in the user's name
              └───────────────────────────────┘
```

Only one expert runs per frame — that single-expert dispatch *is* the on-device
efficiency win. Inference runs through ONNX Runtime (no eager-PyTorch overhead),
fully local, no cloud.

### The naming / memory feature
- Press **`r`** → names the highest-confidence appliance on screen (type a name, Enter).
- The box's crop is fingerprinted via the ResNet18 backbone (512-d vector) and saved to `device_memory.json`.
- On later frames, each detected box is fingerprinted and cosine-matched against the gallery; a match (≥ 0.82) shows the user's name in yellow plus a simulated `SmartThings ▸ <name>` banner.
- Press **`x`** to clear all saved devices, **`q`** to quit.
- Matching is by **appearance** (survives the camera leaving and returning); two identical units would share an identity — fine for a single-room demo.

---

## Run it

```bash
# 1. install (once)
pip install -r requirements.txt
pip install "rfdetr[onnx]" onnxruntime

# 2. export the models to ONNX (once)
python export_onnx.py

# 3. (optional) verify the decode
python probe_onnx.py --image test_assets/tv1.jpg

# 4. run — defaults are baked in (video source, width 416, verify off, detect every 3)
python demo_onnx.py
```

Override defaults anytime, e.g. webcam + record output:
```bash
python demo_onnx.py --source 0 --out result.mp4
```

On Apple Silicon, route through the on-device NPU/GPU: `MOE_DEVICE=mps python demo_onnx.py`.

---

## Metrics

### Model quality (reported from training — verify against your own logs)
| Component | Metric | Value |
| --- | --- | --- |
| Scene router (ResNet18, 3-class head) | Validation accuracy | 93.3% |
| Router trainable params | Count | 1,539 |
| Climate expert (RF-DETR Nano) | mAP@[.50:.95], held-out test | 0.9321 |
| Old SSD-MobileNet baseline (replaced) | mAP | 0.14 |

### ONNX parity — `probe_onnx.py` on `tv1.jpg` (500×410)
Decoding the ONNX output matches the original PyTorch model essentially exactly:

| Detection | ONNX | rfdetr reference |
| --- | --- | --- |
| TV (cls 72) | 0.921 @ (110,57,340,191) | 0.920 @ (109,57,340,191) |
| cls 1 | 0.794 @ (123,82,244,156) | 0.803 @ (123,82,244,156) |
| cls 1 | 0.725 @ (48,57,95,110) | 0.733 @ (48,57,95,110) |
| cls 1 | 0.605 @ (232,83,335,162) | 0.609 @ (232,82,335,162) |
| cls 75 | 0.413 @ (285,299,322,313) | 0.435 @ (285,299,322,313) |

Max box delta ≈ 1 px, max confidence delta ≈ 0.02 (floating-point rounding from
the resize/normalize path). **Decode confirmed correct** — no accuracy lost in
the ONNX conversion.

### Runtime
- COCO expert ONNX input resolution: **384×384** (fixed in the graph).
- RF-DETR Nano published latency: **2.3 ms/frame** (NVIDIA T4, TensorRT FP16) — the architecture is real-time; CPU eager-PyTorch was the bottleneck, not the model.
- **On-device FPS (this build):** _measure on your machine_ — `demo_onnx.py` prints `avg ~X FPS` on exit. Record eager-PyTorch (`demo.py`) vs ONNX (`demo_onnx.py`) on the same clip for a clean before/after number.

### Device re-ID
- Fingerprint dimension: **512** (ResNet18 backbone features, L2-normalised).
- Match threshold: cosine ≥ **0.82** (`REID_SIM_THRESH` in `demo_onnx.py`).
- Persistence: `device_memory.json` (survives restarts; re-naming updates in place, no duplicates).

---

## Path to full deployment
The current ONNX build is the on-ramp to the phone/hub target. The next rung is
one function call in `export_onnx.py`:

```python
model.export(format="tflite", quantization="int8",
             calibration_data="path/to/your_frames/")
```

This yields an INT8 TFLite model for edge hardware. The decode logic in
`onnx_detector.py` ports almost directly to a TFLite wrapper. Wiring a matched
device id to a real SmartThings settings deep-link is then the integration step.
