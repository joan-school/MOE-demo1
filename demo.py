"""
MoE Object Detection Demo
Router → 1 of 3 experts → bounding boxes
"""
import argparse
import json
import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

# ---------- Paths ----------
WEIGHTS_DIR = Path('weights')
ROUTER_PTH = WEIGHTS_DIR / 'router_best.pth'
ROUTER_CLASS_MAP = WEIGHTS_DIR / 'class_to_idx.json'
CLIMATE_PTH = WEIGHTS_DIR / 'climate_expert_v1_map0920.pth'

# ---------- Knobs (tweak during demo prep) ----------
ROUTER_CONF_THRESH = 0.50     # below this, don't update the smoother
SMOOTHING_WINDOW = 15         # rolling majority over N frames
DETECT_THRESH = {
    'climate': 0.7,
    'kitchen': 0.5,
    'display': 0.5,
}
ROUTE_EVERY_N_FRAMES = 1      # run router on every Nth frame; 1 = every frame

# ---------- Setup models ----------
print("Loading router...")
router = tvm.resnet18(weights=None)
router.fc = nn.Linear(router.fc.in_features, 3)
state = torch.load(ROUTER_PTH, map_location='cpu', weights_only=True)
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
router.load_state_dict(state)
router.eval()

with open(ROUTER_CLASS_MAP) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
print(f"  router classes: {idx_to_class}")

router_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading climate expert...")
climate_expert = RFDETRNano(pretrain_weights=str(CLIMATE_PTH), num_classes=1)

print("Loading COCO model (kitchen + display experts)...")
coco_model = RFDETRNano()

# Map name → COCO id
target_names = {'microwave', 'refrigerator', 'tv'}
coco_ids = {v: k for k, v in COCO_CLASSES.items() if v in target_names}
KITCHEN_IDS = {coco_ids['microwave'], coco_ids['refrigerator']}
DISPLAY_IDS = {coco_ids['tv']}
print(f"  COCO IDs: {coco_ids}")
print(f"  kitchen filter: {KITCHEN_IDS}, display filter: {DISPLAY_IDS}")

# ---------- Helpers ----------
COLOR_BY_CLASS = {
    'climate': (0, 165, 255),    # orange (BGR)
    'kitchen': (0, 200, 0),      # green
    'display': (200, 60, 200),   # purple
}

def predict_router(pil_img):
    """Return (chosen_class_name, conf, full_probs_dict)."""
    with torch.no_grad():
        logits = router(router_tf(pil_img).unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return idx_to_class[idx], float(probs[idx]), {
        idx_to_class[i]: float(probs[i]) for i in range(3)
    }

def run_expert(pil_img, expert_name):
    """Returns supervision.Detections for the chosen expert."""
    thresh = DETECT_THRESH[expert_name]
    if expert_name == 'climate':
        return climate_expert.predict(pil_img, threshold=thresh)
    # kitchen + display both go through coco_model, then filter
    raw = coco_model.predict(pil_img, threshold=thresh)
    if expert_name == 'kitchen':
        keep = np.isin(raw.class_id, list(KITCHEN_IDS))
    else:
        keep = np.isin(raw.class_id, list(DISPLAY_IDS))
    return raw[keep]

def draw_boxes(frame_bgr, detections, expert_name):
    color = COLOR_BY_CLASS[expert_name]
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        conf = float(detections.confidence[i])
        cid = int(detections.class_id[i])
        label_name = COCO_CLASSES.get(cid, expert_name) if expert_name != 'climate' else 'AC'
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{label_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_hud(frame_bgr, probs, active_expert, raw_pick, raw_conf, fps, smoothing_buf):
    """Top-left overlay: router bars, active expert, FPS."""
    h, w = frame_bgr.shape[:2]
    panel_w = 290
    panel_h = 175
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    cv2.putText(frame_bgr, "MoE Object Detection", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Router probability bars
    y = 45
    for cls in ['climate', 'display', 'kitchen']:
        p = probs.get(cls, 0.0)
        color = COLOR_BY_CLASS[cls]
        cv2.putText(frame_bgr, f"{cls[:3]}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        # bar
        bar_x0, bar_x1_max = 50, 200
        bar_y, bar_h = y - 10, 12
        cv2.rectangle(frame_bgr, (bar_x0, bar_y), (bar_x1_max, bar_y + bar_h), (60, 60, 60), -1)
        bar_x1 = bar_x0 + int(p * (bar_x1_max - bar_x0))
        cv2.rectangle(frame_bgr, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), color, -1)
        cv2.putText(frame_bgr, f"{p:.2f}", (bar_x1_max + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    # Active expert
    cv2.putText(frame_bgr, f"Active: {active_expert.upper()}",
                (10, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                COLOR_BY_CLASS[active_expert], 2)

    # Raw pick (pre-smoothing) for debugging
    flag = "" if raw_conf >= ROUTER_CONF_THRESH else " (low conf)"
    cv2.putText(frame_bgr, f"raw: {raw_pick} {raw_conf:.2f}{flag}",
                (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # FPS
    cv2.putText(frame_bgr, f"{fps:.1f} FPS",
                (10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

# ---------- Main loop ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0',
                        help="Video path or '0' for webcam")
    parser.add_argument('--out', default=None,
                        help="Optional path to save annotated output video (e.g. out.mp4)")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, in_fps, (in_w, in_h))

    smoothing_buf = deque(maxlen=SMOOTHING_WINDOW)
    active_expert = 'kitchen'  # arbitrary initial pick
    last_probs = {'climate': 0., 'display': 0., 'kitchen': 0.}
    last_raw_pick = '-'
    last_raw_conf = 0.
    frame_idx = 0
    t_prev = time.time()
    fps_smoothed = 0.0

    print("\nRunning... press 'q' to quit.\n")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Convert frame to PIL once
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # Router (every Nth frame)
        if frame_idx % ROUTE_EVERY_N_FRAMES == 0:
            pick, conf, probs = predict_router(pil)
            last_probs = probs
            last_raw_pick = pick
            last_raw_conf = conf
            if conf >= ROUTER_CONF_THRESH:
                smoothing_buf.append(pick)

        # Update active expert via majority over the smoothing window
        if smoothing_buf:
            active_expert = Counter(smoothing_buf).most_common(1)[0][0]

        # Run the active expert
        detections = run_expert(pil, active_expert)
        draw_boxes(frame_bgr, detections, active_expert)

        # FPS (EMA)
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        inst_fps = 1.0 / dt if dt > 0 else 0
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps if fps_smoothed else inst_fps

        # HUD
        draw_hud(frame_bgr, last_probs, active_expert,
                 last_raw_pick, last_raw_conf, fps_smoothed, smoothing_buf)

        cv2.imshow("MoE Demo", frame_bgr)
        if writer is not None:
            writer.write(frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_idx} frames, avg ~{fps_smoothed:.1f} FPS")

if __name__ == '__main__':
    main()
