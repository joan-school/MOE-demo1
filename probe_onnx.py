"""
probe_onnx.py  –  Sanity-check the ONNX decode against the real RF-DETR model.

Run AFTER export_onnx.py:

    python probe_onnx.py --image some_test_frame.jpg

It prints, for the COCO model:
  * the ONNX output names / shapes / value ranges
  * the top detections decoded by OnnxDetector
  * the top detections from rfdetr's own .predict() on the same image

If the two lists line up (similar boxes, same class ids, similar scores),
demo_onnx.py will work as-is. If they don't, paste this whole output back and
the decode in onnx_detector.py gets a one-line fix.
"""
import argparse
import numpy as np
from PIL import Image

from rfdetr import RFDETRNano
from onnx_detector import OnnxDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='a test frame (jpg/png)')
    ap.add_argument('--onnx', default='weights/coco.onnx')
    ap.add_argument('--thresh', type=float, default=0.3)
    args = ap.parse_args()

    img = Image.open(args.image).convert('RGB')
    print(f"image size (w,h) = {img.size}\n")

    # --- raw ONNX outputs ---
    det = OnnxDetector(args.onnx, device='cpu')
    inp = det._preprocess(img)
    raw = det.session.run(None, {det.input_name: inp})
    print("ONNX raw outputs:")
    for name, arr in zip([o.name for o in det.session.get_outputs()], raw):
        arr = np.asarray(arr)
        print(f"  {name:10s} shape={arr.shape} "
              f"min={arr.min():.3f} max={arr.max():.3f}")
    print()

    # --- decoded by our OnnxDetector ---
    d = det.predict(img, threshold=args.thresh)
    print(f"OnnxDetector decoded {len(d)} dets (thresh {args.thresh}):")
    order = np.argsort(-d.confidence)[:8] if len(d) else []
    for i in order:
        x1, y1, x2, y2 = d.xyxy[i].astype(int)
        print(f"  cls={int(d.class_id[i]):3d} conf={d.confidence[i]:.3f} "
              f"box=({x1},{y1},{x2},{y2})")
    print()

    # --- ground truth: rfdetr's own predict ---
    print("rfdetr .predict() reference:")
    ref = RFDETRNano().predict(img, threshold=args.thresh)
    order = np.argsort(-ref.confidence)[:8] if len(ref) else []
    for i in order:
        x1, y1, x2, y2 = ref.xyxy[i].astype(int)
        print(f"  cls={int(ref.class_id[i]):3d} conf={ref.confidence[i]:.3f} "
              f"box=({x1},{y1},{x2},{y2})")

    print("\nIf the two lists match up, you're good to run demo_onnx.py.")


if __name__ == '__main__':
    main()
