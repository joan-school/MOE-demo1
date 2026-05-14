"""
Profile the demo pipeline to find the bottleneck.
Runs each component on the same frame 30 times and reports timing.
"""
import time
import json
import statistics
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

WEIGHTS_DIR = Path('weights')
N_RUNS = 30  # warmup + measure

# ---- Load models ----
print("Loading models...")
router = tvm.resnet18(weights=None)
router.fc = nn.Linear(router.fc.in_features, 3)
state = torch.load(WEIGHTS_DIR / 'router_best.pth', map_location='cpu', weights_only=True)
if 'state_dict' in state: state = state['state_dict']
router.load_state_dict(state)
router.eval()

router_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

climate_expert = RFDETRNano(pretrain_weights=str(WEIGHTS_DIR / 'climate_expert_v1_map0920.pth'), num_classes=1)
coco_model = RFDETRNano()

# Use any image you have
test_img = Path('test_assets/test.jpg')
pil_img = Image.open(test_img).convert('RGB')
frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
print(f"Test image: {test_img}, size {pil_img.size}\n")

# ---- Helper ----
def benchmark(name, fn, n=N_RUNS, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    median = statistics.median(times)
    print(f"  {name:35s}  median: {median:6.1f} ms   ({1000/median:5.1f} FPS if alone)")
    return median

# ---- Benchmark each stage ----
print("Benchmarking pipeline stages (median over 30 runs)...\n")

# Stage 1: cv2 → PIL conversion (per frame in real demo)
def stage_convert():
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
t_convert = benchmark("BGR -> PIL conversion", stage_convert)

# Stage 2: Router transforms + forward
def stage_router():
    with torch.no_grad():
        logits = router(router_tf(pil_img).unsqueeze(0))
        _ = torch.softmax(logits, dim=1)[0]
t_router = benchmark("Router preprocess + forward", stage_router)

# Stage 3: Climate expert prediction
def stage_climate():
    _ = climate_expert.predict(pil_img, threshold=0.5)
t_climate = benchmark("Climate expert (RF-DETR Nano)", stage_climate)

# Stage 4: COCO model prediction (kitchen/display use this)
def stage_coco():
    _ = coco_model.predict(pil_img, threshold=0.5)
t_coco = benchmark("COCO model (kitchen/display)", stage_coco)

print(f"\nTotal one-frame budget if router + climate runs:  ~{t_convert + t_router + t_climate:.1f} ms ({1000/(t_convert + t_router + t_climate):.1f} FPS)")
print(f"Total one-frame budget if router + COCO runs:     ~{t_convert + t_router + t_coco:.1f} ms ({1000/(t_convert + t_router + t_coco):.1f} FPS)")