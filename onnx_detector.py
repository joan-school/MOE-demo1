"""
onnx_detector.py  –  ONNX Runtime drop-in replacement for rfdetr's .predict().

Returns a supervision.Detections object with the same (.xyxy, .confidence,
.class_id) interface the rest of demo.py already expects, so it slots straight
into run_coco_shared(), the kitchen/display filters, draw_boxes(), and the
device re-ID feature with no other code changes.

DECODE ASSUMPTIONS (verified by probe_onnx.py against the real model):
  * outputs are (boxes, logits) in that order  -> names 'dets', 'labels'
  * boxes are cxcywh, normalized to [0, 1]
  * logits are per-class scores needing sigmoid (DETR focal-style, not softmax)
If probe_onnx.py shows these are wrong, only this file needs a small tweak.
"""
import numpy as np
import onnxruntime as ort
import supervision as sv

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _providers(device):
    if device == 'cuda':
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if device == 'mps':                      # Apple Silicon on-device path
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


class OnnxDetector:
    """Loads an exported RF-DETR ONNX model and runs it via ONNX Runtime."""

    def __init__(self, onnx_path, device='cpu', box_format='cxcywh'):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=_providers(device))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # input shape is [N, 3, H, W]
        self.in_h = int(inp.shape[2])
        self.in_w = int(inp.shape[3])
        self.box_format = box_format
        print(f"  [onnx] {onnx_path}  in={self.in_w}x{self.in_h}  "
              f"providers={self.session.get_providers()}")

    # demo.py calls this on the rfdetr objects; keep it as a harmless no-op.
    def optimize_for_inference(self):
        pass

    def _preprocess(self, pil_img):
        img = pil_img.convert('RGB').resize((self.in_w, self.in_h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))[None]      # NCHW, batch 1
        return arr.astype(np.float32)

    def predict(self, pil_img, threshold=0.3):
        orig_w, orig_h = pil_img.size
        inp = self._preprocess(pil_img)
        outs = self.session.run(None, {self.input_name: inp})

        boxes = np.asarray(outs[0])
        logits = np.asarray(outs[1])
        if boxes.ndim == 3:
            boxes = boxes[0]
        if logits.ndim == 3:
            logits = logits[0]

        # per-query best class via sigmoid
        scores = 1.0 / (1.0 + np.exp(-logits))        # [Q, C]
        class_id = scores.argmax(axis=1).astype(int)
        confidence = scores.max(axis=1).astype(float)

        keep = confidence >= threshold
        boxes = boxes[keep]
        class_id = class_id[keep]
        confidence = confidence[keep]
        if boxes.shape[0] == 0:
            return sv.Detections.empty()

        if self.box_format == 'cxcywh':
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = (cx - bw / 2.0) * orig_w
            y1 = (cy - bh / 2.0) * orig_h
            x2 = (cx + bw / 2.0) * orig_w
            y2 = (cy + bh / 2.0) * orig_h
        else:                                          # already xyxy in [0,1]
            x1 = boxes[:, 0] * orig_w
            y1 = boxes[:, 1] * orig_h
            x2 = boxes[:, 2] * orig_w
            y2 = boxes[:, 3] * orig_h

        xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        return sv.Detections(xyxy=xyxy, confidence=confidence.astype(np.float32),
                             class_id=class_id)
