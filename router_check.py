"""Quick router-only check across multiple images."""
import json
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

ROUTER_PTH = Path('weights/router_best.pth')
CLASS_MAP = Path('weights/class_to_idx.json')

router = tvm.resnet18(weights=None)
router.fc = nn.Linear(router.fc.in_features, 3)
state = torch.load(ROUTER_PTH, map_location='cpu', weights_only=True)
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
router.load_state_dict(state)
router.eval()

with open(CLASS_MAP) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Run on every image in test_assets/
test_dir = Path('test_assets')
images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.jpeg')) + list(test_dir.glob('*.png')))

if not images:
    print(f"No images found in {test_dir}")
else:
    print(f"\n{'image':<30} {'climate':>8} {'display':>8} {'kitchen':>8}  -> pick (conf)")
    print("-" * 80)
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        with torch.no_grad():
            logits = router(tf(img).unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
        chosen_idx = int(probs.argmax())
        chosen = idx_to_class[chosen_idx]
        conf = probs[chosen_idx].item()
        flag = "  ⚠️ LOW" if conf < 0.6 else ""
        print(f"{img_path.name:<30} "
              f"{probs[0].item():>8.3f} {probs[1].item():>8.3f} {probs[2].item():>8.3f}"
              f"  -> {chosen:<8} ({conf:.2f}){flag}")