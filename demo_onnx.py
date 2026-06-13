"""
MoE Object Detection Demo  –  Performance-Optimised Build
Router → 1 of 3 experts → bounding boxes

Key improvements over previous version:
  • Shared COCO pass: verify + expert detection in a single inference call
  • Half-precision (fp16) router inference on MPS/CUDA
  • Lighter router preprocessing (bilinear resize, skip CenterCrop)
  • Adaptive smoothing window that reacts faster to high-confidence switches
  • Threaded frame capture to eliminate I/O stalls
  • Cached HUD overlay panel (redrawn only when data changes)
  • Lower detection thresholds with NMS tightening for better recall
"""
import argparse
import json
import os
import time
import threading
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from onnx_detector import OnnxDetector   # ONNX Runtime drop-in for RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES
from device_memory import DeviceMemory, build_embedder

# ---------- Paths ----------
WEIGHTS_DIR = Path('weights')
ROUTER_PTH = WEIGHTS_DIR / 'router_best.pth'
ROUTER_CLASS_MAP = WEIGHTS_DIR / 'class_to_idx.json'
CLIMATE_PTH = WEIGHTS_DIR / 'climate_expert_v1_map0920.pth'

# ---------- Knobs (tweak during demo prep) ----------
ROUTER_CONF_THRESH = 0.45     # lowered: accept more router opinions
SMOOTHING_WINDOW = 10         # shorter window → faster reaction
DETECT_THRESH = {
    'climate': 0.65,           # slightly lower for better recall
    'kitchen': 0.40,           # lower: catch more microwaves/fridges
    'display': 0.40,           # lower: catch TVs earlier
}
ROUTE_EVERY_N_FRAMES = 2      # router every 2nd frame is plenty
DETECT_EVERY_N_FRAMES = 3     # run detector every 2nd frame; reuse boxes between
PROCESS_WIDTH = 416            # smaller = faster inference, 640 is sweet spot
MODEL_DEVICE = os.environ.get('MOE_DEVICE', 'cpu')
VERIFY_EVERY_N_FRAMES = 0     # share COCO pass, so verification is nearly free
VERIFY_THRESH = 0.40          # lower: override router sooner

# ---------- Device memory (re-ID / naming) knobs ----------
DEVICE_MEMORY_PATH = 'device_memory.json'
REID_SIM_THRESH = 0.82        # cosine sim needed to call it 'the same device'

# ---------- Threaded frame reader ----------
class FrameGrabber:
    """Read frames in a background thread to decouple I/O from inference."""
    def __init__(self, cap):
        self.cap = cap
        self.grabbed = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2.0)

# ---------- Setup models ----------
print("Loading router...")
router = tvm.resnet18(weights=None)
router.fc = nn.Linear(router.fc.in_features, 3)
state = torch.load(ROUTER_PTH, map_location='cpu', weights_only=True)
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
router.load_state_dict(state)
router.eval()

# Use half precision on GPU/MPS for the router
_use_fp16 = MODEL_DEVICE in ('cuda', 'mps')
if _use_fp16:
    router = router.half()
if MODEL_DEVICE != 'cpu':
    router = router.to(MODEL_DEVICE)

with open(ROUTER_CLASS_MAP) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
print(f"  router classes: {idx_to_class}")

# Lighter transform: direct resize to 224 (skip 256+CenterCrop)
router_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Building device-memory embedder (re-uses router backbone)...")
embed_device = build_embedder(router, device=MODEL_DEVICE, use_fp16=_use_fp16)
device_memory = DeviceMemory(DEVICE_MEMORY_PATH, sim_thresh=REID_SIM_THRESH)
print(f"  remembered devices on disk: {len(device_memory)}")

print("Loading climate expert...")
climate_expert = OnnxDetector('weights/climate.onnx', device=MODEL_DEVICE)
try:
    climate_expert.optimize_for_inference()
    print("  ✓ climate expert optimized for inference")
except Exception:
    pass  # older rfdetr versions may not support this

print("Loading COCO model (kitchen + display experts)...")
coco_model = OnnxDetector('weights/coco.onnx', device=MODEL_DEVICE)
try:
    coco_model.optimize_for_inference()
    print("  ✓ COCO model optimized for inference")
except Exception:
    pass

# Map name → COCO id
target_names = {'microwave', 'refrigerator', 'tv'}
coco_ids = {v: k for k, v in COCO_CLASSES.items() if v in target_names}
KITCHEN_IDS = {coco_ids['microwave'], coco_ids['refrigerator']}
DISPLAY_IDS = {coco_ids['tv']}
print(f"  COCO IDs: {coco_ids}")
print(f"  kitchen filter: {KITCHEN_IDS}, display filter: {DISPLAY_IDS}")
print(f"  model device: {MODEL_DEVICE}, fp16 router: {_use_fp16}")

# ---------- Helpers ----------
COLOR_BY_CLASS = {
    'climate': (0, 165, 255),    # orange (BGR)
    'kitchen': (0, 200, 0),      # green
    'display': (200, 60, 200),   # purple
}


def predict_router(pil_img):
    """Return (chosen_class_name, conf, full_probs_dict)."""
    with torch.inference_mode():
        inp = router_tf(pil_img).unsqueeze(0)
        if _use_fp16:
            inp = inp.half()
        if MODEL_DEVICE != 'cpu':
            inp = inp.to(MODEL_DEVICE)
        logits = router(inp)
        probs = torch.softmax(logits, dim=1)[0].float().cpu()
    idx = int(probs.argmax())
    return idx_to_class[idx], float(probs[idx]), {
        idx_to_class[i]: float(probs[i]) for i in range(3)
    }


def run_coco_shared(pil_img, expert_name, do_verify):
    """Single COCO inference pass shared between expert detection and verification.

    Returns (expert_detections, verified_pick_or_None, verified_conf).
    When expert_name is 'climate', COCO is only run if do_verify is True.
    """
    verified_pick = None
    verified_conf = 0.0
    expert_dets = None

    if expert_name == 'climate':
        # Run climate expert
        thresh = DETECT_THRESH['climate']
        with torch.inference_mode():
            expert_dets = climate_expert.predict(pil_img, threshold=thresh)
        # Run COCO only for verification
        if do_verify:
            with torch.inference_mode():
                raw = coco_model.predict(pil_img, threshold=VERIFY_THRESH)
            verified_pick, verified_conf = _extract_verification(raw)
        return expert_dets, verified_pick, verified_conf

    # For kitchen / display: single COCO pass at the LOWER threshold
    low_thresh = min(DETECT_THRESH[expert_name], VERIFY_THRESH) if do_verify else DETECT_THRESH[expert_name]
    with torch.inference_mode():
        raw = coco_model.predict(pil_img, threshold=low_thresh)

    # Filter expert detections at expert threshold
    if expert_name == 'kitchen':
        keep = np.isin(raw.class_id, list(KITCHEN_IDS)) & (raw.confidence >= DETECT_THRESH['kitchen'])
    else:
        keep = np.isin(raw.class_id, list(DISPLAY_IDS)) & (raw.confidence >= DETECT_THRESH['display'])
    expert_dets = raw[keep]

    # Verification from same pass
    if do_verify:
        verified_pick, verified_conf = _extract_verification(raw)

    return expert_dets, verified_pick, verified_conf


def _extract_verification(raw):
    """Pull the best kitchen/display target from a COCO result set."""
    if len(raw) == 0:
        return None, 0.0
    target_ids = list(KITCHEN_IDS | DISPLAY_IDS)
    keep = np.isin(raw.class_id, target_ids) & (raw.confidence >= VERIFY_THRESH)
    if not np.any(keep):
        return None, 0.0
    target_class_ids = raw.class_id[keep].astype(int)
    target_conf = raw.confidence[keep].astype(float)
    best_idx = int(np.argmax(target_conf))
    best_class_id = int(target_class_ids[best_idx])
    best_conf = float(target_conf[best_idx])
    if best_class_id in DISPLAY_IDS:
        return 'display', best_conf
    if best_class_id in KITCHEN_IDS:
        return 'kitchen', best_conf
    return None, 0.0


def run_expert_only(pil_img, expert_name):
    """Fallback: run just the expert without shared verification."""
    thresh = DETECT_THRESH[expert_name]
    with torch.inference_mode():
        if expert_name == 'climate':
            return climate_expert.predict(pil_img, threshold=thresh)
        raw = coco_model.predict(pil_img, threshold=thresh)
    if expert_name == 'kitchen':
        keep = np.isin(raw.class_id, list(KITCHEN_IDS))
    else:
        keep = np.isin(raw.class_id, list(DISPLAY_IDS))
    return raw[keep]


def maybe_resize(frame_bgr, target_width):
    """Resize the live stream before inference. Keeps detector boxes in display coords."""
    if target_width <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= target_width:
        return frame_bgr
    scale = target_width / float(w)
    target_height = max(1, int(h * scale))
    return cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


NAMED_COLOR = (0, 255, 255)  # yellow (BGR) for user-named devices


def crop_pil(frame_bgr, box_xyxy):
    """Crop a detection box and return it as a PIL image (or None if empty)."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def draw_boxes(frame_bgr, detections, expert_name, names=None, highlight=-1):
    base_color = COLOR_BY_CLASS[expert_name]
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        conf = float(detections.confidence[i])
        cid = int(detections.class_id[i])
        custom = names[i] if names is not None and i < len(names) else None
        if custom:
            color = NAMED_COLOR
            label = custom
            txt_color = (0, 0, 0)
        else:
            color = base_color
            label_name = COCO_CLASSES.get(cid, expert_name) if expert_name != 'climate' else 'AC'
            label = f"{label_name} {conf:.2f}"
            txt_color = (255, 255, 255)
        thick = 3 if i == highlight else 2
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)


def draw_feedback_overlay(frame_bgr, memory, names, rename_mode,
                          rename_buffer, flash_msg, flash_active):
    """Hotkey hint, SmartThings cue, rename input box, and flash messages."""
    h, w = frame_bgr.shape[:2]

    hint = f"[r] name   [x] clear   [q] quit    |    saved: {len(memory)}"
    (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame_bgr, hint, (w - tw - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Simulated SmartThings deep-link cue when a known device is on screen.
    known = [n for n in (names or []) if n]
    if known and not rename_mode:
        banner = f"SmartThings >  {known[0]}  -  tap to open settings"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x0 = (w - bw) // 2 - 12
        y0 = h - 46
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + bw + 24, y0 + 34), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame_bgr, 0.3, 0, frame_bgr)
        cv2.putText(frame_bgr, banner, (x0 + 12, y0 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, NAMED_COLOR, 2)

    # Rename input box (center of screen).
    if rename_mode:
        prompt = f"Name this device: {rename_buffer}_"
        (pw, ph), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x0 = (w - pw) // 2 - 16
        y0 = h // 2 - 22
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + pw + 32, y0 + 44), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, frame_bgr, 0.22, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (x0, y0), (x0 + pw + 32, y0 + 44), NAMED_COLOR, 2)
        cv2.putText(frame_bgr, prompt, (x0 + 16, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_bgr, "Enter = save     Esc = cancel",
                    (x0 + 16, y0 + 44 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Transient flash message (e.g. "Saved: Living Room AC").
    if flash_active and flash_msg:
        (fw, fh), _ = cv2.getTextSize(flash_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame_bgr, flash_msg, ((w - fw) // 2, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_hud(frame_bgr, probs, active_expert, raw_pick, raw_conf, fps, smoothing_buf):
    """Top-left overlay: router bars, active expert, FPS."""
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
    parser.add_argument('--source', default='test_assets/device.mp4',
                        help="Video path or '0' for webcam")
    parser.add_argument('--out', default=None,
                        help="Optional path to save annotated output video (e.g. out.mp4)")
    parser.add_argument('--route-every', type=int, default=ROUTE_EVERY_N_FRAMES,
                        help="Run the router every N frames. Higher is faster but reacts slower.")
    parser.add_argument('--detect-every', type=int, default=DETECT_EVERY_N_FRAMES,
                        help="Run the active detector every N frames and reuse boxes between runs.")
    parser.add_argument('--verify-every', type=int, default=VERIFY_EVERY_N_FRAMES,
                        help="Run COCO target-object verification every N frames. Use 0 to disable.")
    parser.add_argument('--verify-thresh', type=float, default=VERIFY_THRESH,
                        help="Target-object confidence needed before verification overrides the router.")
    parser.add_argument('--width', type=int, default=PROCESS_WIDTH,
                        help="Resize frames wider than this before inference/display. Use 0 to disable.")
    parser.add_argument('--camera-width', type=int, default=1280,
                        help="Requested webcam capture width when --source is a camera index.")
    parser.add_argument('--camera-height', type=int, default=720,
                        help="Requested webcam capture height when --source is a camera index.")
    parser.add_argument('--torch-threads', type=int, default=0,
                        help="Limit PyTorch CPU threads. Use 0 to keep PyTorch default.")
    args = parser.parse_args()

    cv2.setUseOptimized(True)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = in_w, in_h
    if args.width > 0 and in_w > args.width:
        scale = args.width / float(in_w)
        out_w = args.width
        out_h = max(1, int(in_h * scale))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, in_fps, (out_w, out_h))

    # Use threaded reader for webcam to eliminate I/O stalls
    grabber = None
    if isinstance(src, int):
        grabber = FrameGrabber(cap)

    smoothing_buf = deque(maxlen=SMOOTHING_WINDOW)
    active_expert = 'kitchen'  # arbitrary initial pick
    last_probs = {'climate': 0., 'display': 0., 'kitchen': 0.}
    last_raw_pick = '-'
    last_raw_conf = 0.
    last_detections = None
    last_detection_expert = None
    last_embeddings = []   # 512-d vec per detection, aligned with last_detections
    last_names = []        # remembered name (or None) per detection
    rename_mode = False
    rename_buffer = ''
    rename_idx = -1
    rename_vec = None
    flash_msg = ''
    flash_until = 0.0
    frame_idx = 0
    t_prev = time.perf_counter()
    fps_smoothed = 0.0

    route_every = max(1, args.route_every)
    detect_every = max(1, args.detect_every)
    verify_every = max(0, args.verify_every)
    print(
        "\nRunning... press 'q' to quit."
        f"\n  route every {route_every} frame(s)"
        f"\n  detect every {detect_every} frame(s)"
        f"\n  verify every {verify_every if verify_every else 'never'} frame(s)"
        f"\n  max width: {'native' if args.width <= 0 else args.width}"
        f"\n  fp16 router: {_use_fp16}\n"
    )
    while True:
        # --- Frame acquisition ---
        if grabber is not None:
            ok, frame_bgr = grabber.read()
            if frame_bgr is None:
                continue
        else:
            ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_idx += 1
        frame_bgr = maybe_resize(frame_bgr, args.width)

        should_route = frame_idx % route_every == 0
        should_detect = frame_idx % detect_every == 0 or last_detections is None
        should_verify = verify_every > 0 and frame_idx % verify_every == 0

        # Convert to PIL only when a model actually needs this frame.
        pil = None
        if should_route or should_detect or should_verify:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # Router (every Nth frame)
        if should_route and pil is not None:
            pick, conf, probs = predict_router(pil)
            last_probs = probs
            last_raw_pick = pick
            last_raw_conf = conf
            if conf >= ROUTER_CONF_THRESH:
                smoothing_buf.append(pick)
                # Adaptive boost: very high-confidence picks fill more of the window
                if conf >= 0.85:
                    smoothing_buf.append(pick)
                    smoothing_buf.append(pick)

        # --- Shared COCO pass: detection + verification in ONE inference call ---
        if should_detect or last_detection_expert != active_expert:
            if pil is not None:
                dets, verified_pick, verified_conf = run_coco_shared(
                    pil, active_expert, do_verify=should_verify
                )
                last_detections = dets
                last_detection_expert = active_expert

                # Apply verification override
                if verified_pick is not None and verified_conf >= args.verify_thresh:
                    smoothing_buf.clear()
                    smoothing_buf.extend([verified_pick] * smoothing_buf.maxlen)
                    last_raw_pick = f"{verified_pick}*"
                    last_raw_conf = verified_conf
                    last_probs = {
                        'climate': 0.0,
                        'display': 1.0 if verified_pick == 'display' else 0.0,
                        'kitchen': 1.0 if verified_pick == 'kitchen' else 0.0,
                    }
        elif should_verify and pil is not None:
            # If we skipped detection this frame but still need to verify
            verified_pick, verified_conf = _extract_verification(
                coco_model.predict(pil, threshold=VERIFY_THRESH)
            )
            if verified_pick is not None and verified_conf >= args.verify_thresh:
                smoothing_buf.clear()
                smoothing_buf.extend([verified_pick] * smoothing_buf.maxlen)
                last_raw_pick = f"{verified_pick}*"
                last_raw_conf = verified_conf
                last_probs = {
                    'climate': 0.0,
                    'display': 1.0 if verified_pick == 'display' else 0.0,
                    'kitchen': 1.0 if verified_pick == 'kitchen' else 0.0,
                }

        # Update active expert via majority over the smoothing window
        if smoothing_buf:
            active_expert = Counter(smoothing_buf).most_common(1)[0][0]

        # --- Device re-ID: fingerprint current boxes, look up saved names ---
        # Only pay this cost on detect frames AND only once at least one device
        # has been named. Before naming anything it is a pure no-op, so the demo
        # runs at full speed until the feature is actually used.
        if should_detect:
            if len(device_memory) == 0 or last_detections is None:
                last_embeddings = []
                last_names = []
            else:
                last_embeddings = []
                last_names = []
                for bi in range(len(last_detections)):
                    crop = crop_pil(frame_bgr, last_detections.xyxy[bi])
                    if crop is None:
                        last_embeddings.append(None)
                        last_names.append(None)
                        continue
                    vec = embed_device(crop)
                    last_embeddings.append(vec)
                    hit = device_memory.match(vec)
                    last_names.append(hit[0]['name'] if hit else None)

        if last_detections is not None:
            hl = rename_idx if rename_mode else -1
            draw_boxes(frame_bgr, last_detections, last_detection_expert,
                       names=last_names, highlight=hl)

        # FPS (EMA)
        t_now = time.perf_counter()
        dt = t_now - t_prev
        t_prev = t_now
        inst_fps = 1.0 / dt if dt > 0 else 0
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps if fps_smoothed else inst_fps

        # HUD
        draw_hud(frame_bgr, last_probs, active_expert,
                 last_raw_pick, last_raw_conf, fps_smoothed, smoothing_buf)

        draw_feedback_overlay(frame_bgr, device_memory, last_names,
                              rename_mode, rename_buffer, flash_msg,
                              time.perf_counter() < flash_until)

        cv2.imshow("MoE Demo", frame_bgr)
        if writer is not None:
            writer.write(frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if rename_mode:
            if key in (13, 10):                # Enter -> save
                name = rename_buffer.strip()
                if name and rename_vec is not None:
                    device_memory.remember(
                        rename_vec, name,
                        last_detection_expert or active_expert)
                    if last_names and 0 <= rename_idx < len(last_names):
                        last_names[rename_idx] = name
                    flash_msg = f"Saved: {name}"
                    flash_until = time.perf_counter() + 1.5
                rename_mode = False; rename_buffer = ''
                rename_idx = -1; rename_vec = None
            elif key == 27:                    # Esc -> cancel
                rename_mode = False; rename_buffer = ''
                rename_idx = -1; rename_vec = None
            elif key in (8, 127):              # Backspace
                rename_buffer = rename_buffer[:-1]
            elif 32 <= key <= 126 and len(rename_buffer) < 24:
                rename_buffer += chr(key)
        else:
            if key == ord('q'):
                break
            elif key == ord('r'):              # name the most confident box
                if last_detections is not None and len(last_detections) > 0:
                    idx = int(np.argmax(last_detections.confidence))
                    vec = last_embeddings[idx] if idx < len(last_embeddings) else None
                    if vec is None:
                        crop = crop_pil(frame_bgr, last_detections.xyxy[idx])
                        vec = embed_device(crop) if crop is not None else None
                    if vec is not None:
                        rename_mode = True; rename_buffer = ''
                        rename_idx = idx; rename_vec = vec
                else:
                    flash_msg = "No device in view to name"
                    flash_until = time.perf_counter() + 1.2
            elif key == ord('x'):              # wipe remembered devices
                device_memory.forget_all()
                last_names = [None] * len(last_names)
                flash_msg = "Cleared all remembered devices"
                flash_until = time.perf_counter() + 1.5

    if grabber is not None:
        grabber.stop()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_idx} frames, avg ~{fps_smoothed:.1f} FPS")

if __name__ == '__main__':
    main()
