# MoE Object Detection — Samsung Demo

A Mixture of Experts (MoE) object detection prototype for on-device Android deployment. A small scene-classifier router selects which of three specialized detectors runs per frame, paying for one detector's inference cost rather than three running in parallel.

Three product categories are covered: **climate** (air conditioners), **kitchen** (microwaves, refrigerators), and **display** (TVs).

## Architecture
Camera → Router (ResNet18, 3 classes)
│
├── confidence gate (threshold 0.55)
├── temporal smoothing (15-frame majority vote)
│
├──→ Climate expert  (RF-DETR Nano, fine-tuned on AC)
├──→ Kitchen expert  (RF-DETR Nano, COCO + filter)
└──→ Display expert  (RF-DETR Nano, COCO + filter)
│
└──→ OpenCV display + bounding boxes + HUD

Kitchen and display experts share one COCO-pretrained model in memory; only the filter on output classes differs.

See `REPORT.md` for the full implementation report including training results, design decisions, and on-device deployment readiness.

## Performance

| Component | Metric | Value |
|---|---|---|
| Router | Validation accuracy | 93.3% |
| Router | TFLite FP16 numerical match vs PyTorch | within 0.000001 |
| Climate expert | mAP @ 0.50:0.95 (test set, 125 images) | 0.932 |
| Climate expert | mAP @ 0.50 | ~1.000 |
| Demo throughput | CPU laptop, PyTorch | 3–6 FPS |

## Setup

Requires Python 3.10+. Tested on Windows; Linux/macOS should work with the activate command swap.

```bash
# Create virtual env
python -m venv .venv

# Activate
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Linux/macOS:         source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Weights

Model weights are hosted on Google Drive due to file size (router 45 MB, climate expert 121 MB).

**Download from:** [PASTE GOOGLE DRIVE FOLDER LINK HERE]

Place the following files in `weights/`:
- `router_best.pth`
- `class_to_idx.json`
- `climate_expert_v1_map0920.pth`

The COCO-pretrained model used by the kitchen and display experts is auto-downloaded by the `rfdetr` package on first run (~350 MB to local cache, one-time).

## Run

Smoke test — verifies all three models load and predict on one image:

```bash
python smoke_test.py
```

Drop a test image at `test_assets/test.jpg` first to see prediction output.

Router-only diagnostic across multiple images:

```bash
python router_check.py
```

Place test images in `test_assets/` (any `.jpg`, `.jpeg`, `.png`).

Full MoE pipeline on a video file:

```bash
python demo.py --source path/to/video.mp4
```

Save annotated output:

```bash
python demo.py --source path/to/video.mp4 --out output.mp4
```

Webcam:

```bash
python demo.py --source 0
```

Press `q` to quit.

## Demo Tuning Knobs

In `demo.py`:

- `ROUTER_CONF_THRESH` — frames below this confidence don't update the smoothing buffer (default 0.55)
- `SMOOTHING_WINDOW` — rolling window for majority-vote expert selection (default 15)
- `DETECT_THRESH` — per-expert detection confidence threshold (climate: 0.7, others: 0.5)

## Files

| File | Purpose |
|---|---|
| `demo.py` | Main MoE pipeline with router + expert dispatch |
| `smoke_test.py` | Model loading + single-image prediction check |
| `router_check.py` | Router-only multi-image diagnostic |
| `REPORT.md` | Full implementation report |
| `requirements.txt` | Python dependencies |
| `weights/` | Model weights (download from Drive — see above) |
| `docs/` | Architecture diagrams |

## Licensing

All components used in this project are commercial-friendly:

- **RF-DETR** (experts and detection backbone) — Apache 2.0
- **PyTorch / TorchVision** — BSD-3
- **ONNX, TFLite, onnx2tf** — Apache 2.0 / MIT
- **Places365 ResNet18 weights** (router backbone) — CC BY 4.0 with attribution

> Scene classification powered by Places365-CNN. Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A. "Places: A 10 million Image Database for Scene Recognition." IEEE TPAMI, 2017.

AGPL-licensed alternatives (YOLOv5/v8/v11 from Ultralytics, etc.) were deliberately avoided to keep this pipeline commercially deployable.