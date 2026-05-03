How the Router Works
The router is the decision-making heart of the MoE pipeline. It looks at each incoming camera frame and decides which of the three expert detectors should process it. This document explains the design, the training, the inference path, and the export pipeline.

Why have a router at all?
The naive way to detect objects across multiple product categories is to run one detector per category and merge the results. For three categories, that means three detectors running on every frame.
For on-device deployment, this is the wrong design. Phone NPUs have a limited compute budget per frame, and running three full-size detectors in parallel either burns the budget or forces each detector to be smaller (and therefore less accurate).
The MoE alternative: run one tiny classifier first that picks which detector to run, then run only that detector. The classifier is small enough to be effectively free; the inference cost per frame is dominated by the one chosen expert, not three.
For this to work, the classifier needs to be fast, accurate, and small. That is what the router is.

What the router actually does
Input: A 224×224 RGB image — one camera frame, resized and normalized.
Output: A 3-element probability vector, one entry per scene category: climate, display, kitchen.
Decision: The expert corresponding to the highest-probability class is selected to run on this frame, subject to a confidence gate and temporal smoothing (described below).
The router is not a detector. It does not produce bounding boxes. It only answers the question: "What kind of scene is this?"

Architecture
The router uses a ResNet18 backbone with a custom 3-class classification head.
Input: [1, 3, 224, 224] RGB image
   │
   ▼
ResNet18 backbone
(pretrained on Places365, frozen)
   │
   ▼
Adaptive average pool → 512-d feature vector
   │
   ▼
Linear layer (512 → 3)
   │
   ▼
Output: [1, 3] raw logits
   │
   ▼
Softmax (applied at inference time, in app code)
   │
   ▼
[climate_prob, display_prob, kitchen_prob]
Class index mapping (canonical, do not change):
{climate: 0, display: 1, kitchen: 2}
This mapping is also stored in weights/class_to_idx.json and must match between training and inference.
Why ResNet18?
Three reasons:

Places365 pretrained weights are publicly available. Places365 is a scene-classification dataset of 10 million images covering 365 scene categories. Models pretrained on Places365 already understand kitchens, living rooms, offices, and other scene contexts — exactly what our router needs to distinguish.
License-clean. The Places365 ResNet18 weights are CC BY 4.0 (commercial use allowed with attribution), unlike many other pretrained backbones.
Small and fast. ResNet18 is one of the smallest ResNet variants — 11.7M parameters total, runs in a few milliseconds on phone NPUs.

Why freeze the backbone?
The backbone (everything except the final layer) is frozen during training. Only the new 3-class head is updated.
This means:

Trainable parameters: 1,539 (a 512×3 weight matrix plus 3 biases). Three thousand times fewer than a fully trainable ResNet18.
Training is fast: ~5 minutes on a Colab T4 GPU for 15 epochs.
Overfitting is hard: With only 1,539 parameters trying to fit 514 training images, there's no capacity to memorize.
Places365 priors are preserved: The backbone already knows what a kitchen looks like; we don't want to overwrite that knowledge with our small custom dataset.

This is a standard transfer learning pattern called linear probing — using a powerful frozen feature extractor and learning only a thin classification layer on top.

Training
Dataset
Built as a Roboflow single-label classification project named moe-router:

climate: 161 images
display: 183 images
kitchen: 170 images
Total: 514 images
Split: Stratified 80/20 train/val → 409 training, 104 validation

Stratified means each split preserves the per-class proportions, so we don't accidentally end up with all kitchen examples in train and none in val.
Hyperparameters
ParameterValueOptimizerAdamLearning rate1e-3Batch size32Epochs15LossCross-entropyHardwareColab T4 GPU (free tier)Wall time~5 minutes
Input preprocessing (must match at inference)
Every image goes through this exact pipeline:
pythonimport torchvision.transforms as T

router_tf = T.Compose([
    T.Resize(256),                        # shorter edge → 256px
    T.CenterCrop(224),                    # center 224×224 crop
    T.ToTensor(),                         # HWC uint8 → CHW float [0,1]
    T.Normalize(
        mean=[0.485, 0.456, 0.406],       # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
])
Critical: This must be identical between training and inference, and identical between PyTorch and the TFLite Android version. A mismatch in normalization is one of the most common silent bugs in deployed CV — the model still produces output, it's just numerically wrong.
Results
Validation accuracy: 93.3% after 15 epochs.
The model converged quickly — most of the accuracy is reached within the first 5 epochs, with later epochs producing small refinements.

Inference
At runtime, the router does this on every frame (or every Nth frame, configurable):
Step 1: Forward pass
pythonimport torch

with torch.no_grad():
    logits = router(router_tf(pil_img).unsqueeze(0))  # [1, 3]
    probs = torch.softmax(logits, dim=1)[0]           # [3]
The [1, 3] shape is one batch of one image producing three class scores. The softmax converts raw logits into a probability distribution that sums to 1.0.
Step 2: Confidence gate
pythonROUTER_CONF_THRESH = 0.55

chosen_idx = int(probs.argmax())
chosen_class = idx_to_class[chosen_idx]
chosen_conf = float(probs[chosen_idx])

if chosen_conf >= ROUTER_CONF_THRESH:
    smoothing_buffer.append(chosen_class)
# else: skip — don't add this frame's vote
The gate is the key reliability mechanism. If the router is uncertain about a frame (no class above 0.55 confidence), we don't trust that frame's decision at all. We don't update the smoothing buffer; we keep coasting on what we already decided.
This handles a known weakness: on tightly-cropped object photos that don't look like the scenes the router was trained on, the model produces lower-confidence outputs. Without the gate, those frames would inject noise into the routing decision. With the gate, they're safely ignored.
Step 3: Temporal smoothing
pythonfrom collections import deque, Counter

smoothing_buffer = deque(maxlen=15)

# After each frame's gated update:
if smoothing_buffer:
    active_expert = Counter(smoothing_buffer).most_common(1)[0][0]
The active expert is determined by majority vote over the last 15 high-confidence frames. This means:

A single misclassified frame can never flip the active expert.
Transitioning between scenes (e.g., walking from kitchen to living room) takes a few frames to register, which feels natural rather than jittery.
Low-confidence frames don't pollute the buffer (they were already gated out in Step 2).

The window size of 15 is a tunable knob. Smaller windows are more responsive but more jittery; larger windows are smoother but slower to react.
Step 4: Dispatch
python# active_expert is one of: 'climate', 'kitchen', 'display'
detections = run_expert(frame, active_expert)
Only the chosen expert runs. The other two are not invoked for this frame.

Export pipeline (PyTorch → ONNX → TFLite)
The router has been validated through the full Android deployment chain.
Step 1: PyTorch → ONNX
pythonimport torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    router,
    dummy_input,
    "router.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["logits"],
)
ONNX is a cross-platform model format. We use opset 17, which is widely supported by mobile inference engines.
Step 2: ONNX → TFLite
We use the onnx2tf library, which converts ONNX models into TensorFlow Lite format with optional quantization.
bashonnx2tf -i router.onnx -o router_tflite_out
This produces:

router_fp32.tflite — full-precision, 45 MB
router_fp16.tflite — half-precision, 22 MB ← Android production

We tried INT8 conversion (would have produced an ~11 MB file) but without proper calibration data the model collapsed to predicting equal probabilities for all three classes. FP16 at 22 MB is a fine size for the Android app, so we stuck with that.
Step 3: Numerical verification
Critically, we verified that the TFLite model produces the same outputs as the original PyTorch model:
pythonimport numpy as np

# Same input through both models
pytorch_out = router(input_tensor).detach().numpy()
tflite_out  = run_tflite(input_tensor.numpy())

max_diff = np.abs(pytorch_out - tflite_out).max()
# Result: max_diff = 0.000001
A maximum absolute difference of 0.000001 confirms the conversion preserved numerical correctness end-to-end. This is well within the noise floor of FP16 quantization.
Inference contract for Android
For teammates building the Android integration, here is the exact contract:
PropertyValueInput shape[1, 224, 224, 3] (NHWC, TFLite convention)Input dtypefloat32Input normalizationImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]Output shape[1, 3]OutputRaw logits (apply softmax in Java/Kotlin app code)Class orderclimate=0, display=1, kitchen=2Recommended delegateNNAPI on Samsung Exynos, GPU delegate as fallbackConfidence thresholdmax(probs) >= 0.6 (slightly stricter than Python demo at 0.55)
The slightly higher threshold for production reflects the stricter usability bar of a shipped app vs. a demo.

Performance characteristics
Accuracy
MetricValueValidation accuracy93.3%Confident-and-correct rate~98% (when conf ≥ 0.55)
Speed (rough estimates)
PlatformLatency per frameColab T4 GPU<1 msLaptop CPU (PyTorch)~30 msPhone NPU (FP16, projected)3–5 ms
The phone NPU number is a projection based on published ResNet18 benchmarks and has not been measured directly on our target hardware.
Size
FormatSizePyTorch .pth45 MBONNX45 MBTFLite FP3245 MBTFLite FP16 (production)22 MBTFLite INT8 (broken without calibration)~11 MB (not used)

Known limitations
1. Distribution shift on tight crops. The router was trained on scene-style images from Roboflow Universe — wide shots showing context. On tight phone-camera crops of single objects, the router's confidence drops below the gate threshold. Our diagnostic test showed:

Wide AC shots: routed correctly at >0.95 confidence
Wide TV shots: routed correctly at ~0.69 confidence
Tight microwave crop with no scene context: low confidence, ambiguous

The confidence gate plus temporal smoothing handle this gracefully — the active expert holds steady through low-confidence moments — but the underlying classifier could be improved by augmenting training data with phone-camera-style images.
2. Three classes only. Adding a fourth product category (e.g., "lighting") requires retraining the head with a new 4-class output. The backbone does not need to change. This is a small change but it is a change.
3. No multi-label support. The router predicts one class per frame. A scene that genuinely contains both a kitchen appliance and a TV (e.g., an open-plan living room) is forced to pick one. The temporal smoother helps in motion (you'll see kitchen first, then display as you walk past) but a static frame with both products is a hard case.

File reference
FilePurposeweights/router_best.pthTrained PyTorch state dict (45 MB)weights/class_to_idx.jsonClass index mappingrouter_check.pyStandalone router diagnostic across multiple imagesdemo.pyFull pipeline; router is loaded near the top
For the broader pipeline context, see REPORT.md and docs/architecture.png.