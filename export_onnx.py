"""
export_onnx.py  –  Convert the RF-DETR experts to ONNX for on-device inference.

Run ONCE on the machine that has the weights:

    pip install "rfdetr[onnx]" onnxruntime
    python export_onnx.py

Produces:
    weights/climate.onnx   (your fine-tuned climate expert, 1 class)
    weights/coco.onnx      (the COCO-pretrained Nano used by kitchen + display)

After this, run the fast demo:  python demo_onnx.py
"""
from pathlib import Path
import shutil

from rfdetr import RFDETRNano

WEIGHTS = Path('weights')
CLIMATE_PTH = WEIGHTS / 'climate_expert_v1_map0920.pth'


def _export(model, out_name):
    """rfdetr always writes 'output/inference_model.onnx' — export then rename."""
    out_dir = Path('output')
    model.export(output_dir=str(out_dir))
    src = out_dir / 'inference_model.onnx'
    dst = WEIGHTS / out_name
    if not src.exists():
        raise SystemExit(f"Export did not produce {src}")
    shutil.copy(src, dst)
    print(f"  -> {dst}")


print("Exporting climate expert -> weights/climate.onnx ...")
climate = RFDETRNano(pretrain_weights=str(CLIMATE_PTH), num_classes=1)
_export(climate, 'climate.onnx')

print("Exporting COCO Nano -> weights/coco.onnx ...")
coco = RFDETRNano()           # no weights = the COCO-pretrained model
_export(coco, 'coco.onnx')

print("\nDone. Now run:  python probe_onnx.py   (to verify the decode)")
