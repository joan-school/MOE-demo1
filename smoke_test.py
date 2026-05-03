"""
Smoke test: load router + 3 experts, verify everything works on CPU.
No webcam, no display — just plumbing check.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from rfdetr import RFDETRNano

WEIGHTS_DIR = Path('weights')
ROUTER_PTH = WEIGHTS_DIR / 'router_best.pth'
ROUTER_CLASS_MAP = WEIGHTS_DIR / 'class_to_idx.json'
CLIMATE_PTH = WEIGHTS_DIR / 'climate_expert_v1_map0920.pth'

# ---------- 1. Router (ResNet18 + 3-class head) ----------
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
print(f"  classes: {idx_to_class}")
print("  router loaded OK\n")

router_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- 2. Climate expert ----------
print("Loading climate expert (this triggers a one-time COCO weights download ~350MB)...")
climate_expert = RFDETRNano(pretrain_weights=str(CLIMATE_PTH), num_classes=1)
print("  climate expert loaded OK\n")

# ---------- 3. COCO-pretrained model (kitchen + display experts) ----------
print("Loading COCO model for kitchen + display experts...")
coco_model = RFDETRNano()
print("  COCO model loaded OK\n")

# Look up which COCO class IDs we care about
from rfdetr.assets.coco_classes import COCO_CLASSES
target_names = {'microwave', 'refrigerator', 'tv'}
coco_ids = {v: k for k, v in COCO_CLASSES.items() if v in target_names}
print(f"COCO class IDs we need: {coco_ids}\n")

# ---------- 4. Test on an image (optional) ----------
TEST_IMG = Path('test_assets/test.jpg')
if not TEST_IMG.exists():
    print(f"No test image at {TEST_IMG} — skipping prediction test.")
    print("Drop any image (kitchen/AC/TV photo) at that path and re-run to see predictions.")
else:
    print(f"Running predictions on {TEST_IMG}...")
    img = Image.open(TEST_IMG).convert('RGB')

    # Router decision
    with torch.no_grad():
        logits = router(router_tf(img).unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
    print("\nRouter probs:")
    for i in range(3):
        print(f"  {idx_to_class[i]:8s}: {probs[i].item():.3f}")
    chosen = idx_to_class[int(probs.argmax())]
    print(f"  -> picks: {chosen}")

    # Climate
    climate_dets = climate_expert.predict(img, threshold=0.5)
    print(f"\nClimate expert: {len(climate_dets)} detection(s)")

    # COCO + filter
    coco_dets = coco_model.predict(img, threshold=0.5)
    kitchen_targets = {coco_ids.get('microwave'), coco_ids.get('refrigerator')} - {None}
    display_targets = {coco_ids.get('tv')} - {None}
    n_kitchen = sum(1 for c in coco_dets.class_id if c in kitchen_targets)
    n_display = sum(1 for c in coco_dets.class_id if c in display_targets)
    print(f"Kitchen expert (filtered): {n_kitchen} detection(s)")
    print(f"Display expert (filtered): {n_display} detection(s)")

print("\nSmoke test complete.")
