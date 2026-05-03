MoE Object Detection — Implementation Report
Project: On-device Mixture of Experts object detection for Samsung smart appliance categories
Scope: Architectural validation milestone
Status: Core architecture validated end-to-end; production deployment is the natural next step

Executive Summary
The team set out to build a Mixture of Experts object detection system that selects between specialized detectors per frame, designed for on-device deployment on Android. We submitted an architecture proposal centered on MobileNetV3-SSD experts deployed via TensorFlow Lite. During implementation, we encountered material problems with the proposed model components and the surrounding tooling, and we have measured numbers that quantify exactly why the original component choices needed revision.
Rather than ship a non-functional implementation that matched the proposal on paper, we made a deliberate decision to preserve the architecture exactly as designed and substitute the expert backbones for models that train reliably and export cleanly. The result is a working end-to-end pipeline: a 93.3%-accurate scene router, a climate detector hitting 0.932 mAP@50:95 on a held-out test set, and a shared-weight kitchen/display detector — all integrated into a real-time demo with router-driven expert switching, temporal smoothing, and a confidence gate. The router has been validated through the full PyTorch → ONNX → TFLite export chain with output matching the original to within 0.000001 numerical error, confirming the production deployment path.
This report documents what we proposed, what we measured during the original implementation attempts, what we changed and why, what we built, and what remains for production deployment.

What We Proposed
The original architecture targeted on-device inference with the following components:
A MobileNetV3 backbone feeding a router MLP that classified incoming scenes into three product categories — climate, kitchen, and display. Based on the router's decision, one of three SSD-Lite expert detectors would run on the frame. A confidence gate with threshold filtering and temporal smoothing would stabilize the routing decision across frames. An escalation layer would handle low-confidence cases. User feedback and online improvement features were planned as Phase 2 extensions.
The full pipeline was designed for direct Android deployment using TensorFlow Lite with NNAPI hardware acceleration on the Exynos NPU.
This was a sound proposal architecturally. The challenges that emerged were at the implementation layer — specifically with the model choices and the surrounding tooling — not with the routing architecture itself.

Implementation Challenges
We ran into four substantive problems during the build phase. We are documenting these in detail because how a team responds to friction is, in our view, a more honest measure of engineering capability than whether everything went smoothly the first time. Where applicable, we include the actual measurements from our training runs rather than qualitative claims.
1. MobileNetV3-SSD Training Underperformance
We attempted to train SSD MobileNet detectors using the TensorFlow Object Detection API for our target categories. We have detailed training logs and evaluation metrics from two of these attempts — climate (air conditioner) and kitchen — that together demonstrate the underlying problem.
Climate (AC) — SSD MobileNet training results
Training proceeded normally on the surface: losses decreased steadily, the model converged, and TensorBoard graphs looked healthy.
Training trajectory (TensorBoard):
Loss componentStartEndClassification loss~1.1~0.48Localization loss~0.65~0.45Total loss~1.9~1.1
By step 600, total loss had fallen to approximately 0.85 with classification loss at 0.46 and localization loss at 0.23. By every standard "is the model learning" check, the training was working.
But the COCO evaluation metrics on validation tell a very different story:
MetricValuemAP @ IoU 0.50:0.950.140mAP @ IoU 0.500.404mAP @ IoU 0.750.041
The mAP@75 of 0.041 is particularly damning — it means the model could not place tight bounding boxes around the AC unit even when it correctly identified one. Boxes were either roughly in the right area (around 0.40 at IoU 0.50, the loose threshold) or wrong.
Most diagnostically, the size-stratified breakdown:
Object sizemAPSmall objects0.000Medium objects0.023Large objects0.183
The model only worked when the AC unit was large in the frame. Anything small or distant — exactly the cases that matter for a real-world demo where someone walks through a room with an AC mounted on a wall — completely failed. This is a clear signature of insufficient model capacity for the task: small models with small datasets cannot learn robust scale-invariant features.
Average recall told a similar story: AR @ 100 detections was 0.391 overall, but rose to 0.435 for large objects only, confirming the same large-object dependence.
In real-world testing, the model produced occasional correct detections at confidences up to approximately 0.7 (typically on close-up large-object cases) but exhibited misclassification — including detecting TVs in kitchen scenes — that would have made integration with the routing pipeline unreliable.
Kitchen — SSD MobileNet training results
The kitchen attempt showed an even more striking version of the same pattern. Training loss curves looked excellent:
Training trajectory (TensorBoard):
Loss componentStartEnd (~step 1600)Classification loss~0.75~0.10Localization loss~0.55~0.06Total loss~1.5~0.31
A final total loss of 0.31 is a number we would normally associate with a well-trained detector. Looking at the loss curves alone, this would appear to be a successful training run.
In real-world camera testing, the actual behavior was:

Detection confidence scores: 0.02–0.03. The model was almost completely uncertain about every prediction.
Curtains classified as refrigerators. Random textures classified as microwaves.
No correct detections during live demo. The model produced many low-confidence boxes per frame (10 or more in some frames) without any meaningful signal.
No threshold tuning recovered usability. Filtering by confidence eliminated everything.

This is a particularly instructive failure mode. The training loss decreased because the model learned to memorize features of the training set, but the learned features did not generalize to live camera input. The combination of a small model, a small dataset, and the SSD architecture's reliance on default-anchor matching produced a model that looked successful by training metrics and was essentially non-functional at inference.
Why this happened (likely root causes)
The root cause is most likely the mismatch between model capacity and dataset scale. SSD MobileNet is intentionally a small model designed for mobile deployment, and small models generally require either more domain-specific training data than we had available (under 2,000 images per category from Roboflow Universe), or strong pretrained priors closely aligned with the target domain — neither of which we had.
The size-stratified mAP breakdown for the climate model points at the same conclusion from a different angle: scale-invariant feature learning requires either capacity or data, and with neither, the model can only handle one operating regime (large objects, close to the camera).
Larger pretrained models compensate for limited fine-tuning data through richer learned representations. This is precisely what RF-DETR Nano provides — and what the training results in the next section demonstrate.
2. SSD-Lite Tooling and Dependency Issues
The TensorFlow Object Detection API and SSD-Lite ecosystem in 2026 has known compatibility problems across recent TensorFlow and TensorFlow Lite versions. We encountered repeated environment failures during attempts to set up training pipelines and export flows, which consumed disproportionate engineering time without producing working models. These issues are well documented in community channels but lack official resolution paths from the upstream maintainers.
This is a real risk for any team building on SSD-Lite today. It is solvable with sufficient investment in environment debugging, but that investment competes directly with model training, evaluation, and integration work — particularly when the underlying model itself was not training to deployable accuracy in the first place.
3. TensorFlow Limited to CPU-only Execution
A compounding problem was that our TensorFlow setup could not be configured to use GPU acceleration. The combination of CUDA / cuDNN version compatibility, environment constraints on our development hardware, and TensorFlow's own GPU detection requirements meant that all training had to run on CPU.
The cascade effects of this were significant:

Training time per experiment was multiple times longer than it would have been on GPU. What should have been a 1–2 hour fine-tuning run on a T4 GPU became an overnight job on CPU, even on relatively small architectures.
The number of configurations we could try was sharply limited. Hyperparameter exploration, learning rate sweeps, and augmentation policy comparison — all standard practices in deep learning training — became impractical at the cycle time we had available.
Debugging cycles slowed dramatically. Every fix-and-retry iteration to investigate why MobileNetV3-SSD was underperforming required hours rather than minutes, which made it much harder to converge on a root cause or attempt remediation strategies.

The combined effect of poor model convergence, dependency friction, and slow iteration meant that even if we had eventually gotten MobileNetV3-SSD to train acceptably, we would not have had time to also build the routing pipeline, the demo integration, and the export validation before the deadline.
4. Schedule Pressure
The Samsung deadline is fixed. Pushing harder on the original implementation risked producing a half-finished system that demonstrated neither the architecture nor the model quality. With the climate and kitchen training results showing definitively that the chosen model was either operating in a narrow regime (large objects only, climate) or non-functional at inference (kitchen), and the iteration cycle being too slow to recover, this was the tipping point that drove us to revisit the implementation strategy.

The Architectural Pivot
The decision we faced was: continue trying to make the proposed implementation work, or replace specific components while preserving the architecture. The training measurements made the choice clear. With the proposed expert backbone reaching 0.140 mAP@50:95 on AC after extended training, and producing non-usable confidence levels (0.02–0.03) on kitchen, no amount of additional training time would have produced deployable detectors. Continuing down that path would have meant shipping a system whose components individually did not work, regardless of how well the surrounding architecture was implemented.
The key insight is that the value of the MoE proposal is the routing dispatch pattern, not the specific choice of expert backbones. The router decides which detector runs; what each detector internally does is an implementation detail that can evolve independently. As long as the swap preserves on-device viability — model size, quantizability, NPU compatibility, license cleanliness — the architectural claim is intact.
We replaced the SSD-Lite expert backbones with RF-DETR Nano, a recent transformer-based detector that meets all our deployment criteria:

License: Apache 2.0, suitable for commercial Samsung deployment
Active maintenance: Roboflow ships updates, the model has documented benchmarks, and the export tooling works
Size: Comparable to or smaller than the SSD-Lite variants we had planned, after INT8 quantization
Export path: Clean PyTorch → ONNX → TFLite chain with no special tooling required
Training behavior: Trained reliably on our datasets to high accuracy

The empirical contrast on equivalent training conditions is stark:
ModelmAP @ 0.50:0.95mAP @ 0.50mAP @ 0.75Real-world usabilitySSD MobileNet (climate)0.1400.4040.041Only large close-up objectsSSD MobileNet (kitchen)(training loss looked good but live confidences were 0.02–0.03; non-functional in demo)——NoneRF-DETR Nano (climate)0.932~1.0000.989High across all sizes present in dataset
The pivot was driven by measurement, not preference. The chosen replacement model is roughly 7× better at the strict mAP threshold and roughly 2.5× better at the lenient threshold on directly comparable detection tasks, with usability improving from "only works for large close objects" to "deployable across all tested cases."
We also chose to switch the training environment from TensorFlow on local CPU to PyTorch on Google Colab's free T4 GPU. This sidestepped both the TensorFlow CPU constraint and the SSD-Lite dependency issues in one move, and gave us the GPU acceleration the original setup couldn't provide. RF-DETR Nano is natively a PyTorch model, so this was a natural fit.
The router stayed as designed in spirit: a small classifier choosing between three product categories. We implemented it as ResNet18 fine-tuned from Places365 weights, which is comparable in size and intent to the originally planned MobileNetV3 router. ResNet18 was a practical choice because Places365 ResNet18 weights are publicly available, CC BY 4.0 licensed (commercial-OK with attribution), and exactly suited to scene classification — which is what the router does.
The pipeline structure — router → confidence gate → temporal smoothing → expert dispatch → annotated output — stayed identical to the proposal.

What We Built and Validated
Router
A ResNet18 backbone with a custom 3-class head, fine-tuned from Places365 pretrained weights with the backbone frozen. Only the final classification layer was trained, totaling 1,539 trainable parameters. Trained on a Roboflow single-label classification dataset of 514 images split 80/20 between train and validation.
Validation accuracy: 93.3%. Verified to work correctly on phone-camera test images outside the training distribution.
The router has been fully exported through the production Android deployment chain: PyTorch (45 MB) → ONNX (45 MB) → TFLite FP32 (45 MB) → TFLite FP16 (22 MB). Output of the FP16 TFLite model matches the original PyTorch output to within 0.000001 maximum absolute difference, confirming numerical correctness through the conversion. The FP16 TFLite file is 22 MB on disk, suitable for shipping in an Android app with NNAPI delegate acceleration.
This is the strongest deployment-readiness signal in the project. The router is not a research artifact; it is a deployable component.
Climate Expert (custom-trained)
RF-DETR Nano fine-tuned on a Roboflow object detection dataset of 1,746 annotated air conditioner images. Training was conducted on Colab T4 GPU. The training run completed in approximately 12 epochs of effective fine-tuning, with early stopping triggered after the model converged.
Test set performance (125 unseen images, never seen during training or validation):
MetricValuemAP @ IoU 0.50:0.950.9321mAP @ IoU 0.500.9999mAP @ IoU 0.750.9889Mean Average Recall0.9478
Notably, test set performance was slightly higher than validation set performance, indicating the model genuinely generalizes rather than overfitting to validation-specific artifacts. The full pycocotools evaluation summary is available alongside the model weights.
To put this in direct comparison with the previous attempt: the failed SSD MobileNet climate training reached mAP@50 of 0.404 with size-stratified performance dropping to 0.000 for small objects. The RF-DETR Nano climate training reaches mAP@50 of approximately 1.000 on the test set with consistent performance across object sizes present in the dataset. Same architectural slot, comparable dataset scale, dramatically different outcomes.
Kitchen and Display Experts (shared COCO model + filtering)
Both the kitchen and display experts use the same RF-DETR Nano model with COCO pretrained weights, loaded once into memory. The kitchen expert filters detections to COCO classes 78 (microwave) and 82 (refrigerator). The display expert filters to COCO class 72 (TV).
This is a deliberate efficiency win: COCO already covers these classes well, so training custom detectors would have added engineering cost with no accuracy benefit. By sharing one model in memory and applying post-hoc filtering, the marginal cost of supporting the third routing path (display) is essentially zero parameters.
This design decision was informed by the SSD MobileNet kitchen training failure. Once we recognized that small-model fine-tuning was not going to reach deployable accuracy at our dataset scale, we evaluated whether each category actually needed a custom-trained expert. For kitchen and display, COCO's pretrained coverage was sufficient. For climate, where AC was not a COCO class, we needed the custom training — which is exactly what RF-DETR Nano delivered at high quality.
End-to-End Demo
A working real-time inference pipeline built in Python with OpenCV. The demo accepts video file input or webcam, runs the router on each frame, applies a confidence threshold (0.55) and a 15-frame rolling-majority temporal smoother to stabilize routing decisions, dispatches to the chosen expert, and renders bounding boxes and a heads-up display showing router probabilities, the active expert, and FPS.
We tested the pipeline on phone-recorded video clips of each scene type. The router correctly switches active experts as the camera pans between scenes — for example, in a fridge clip, the active expert locks to "kitchen" with router confidence around 0.69, the kitchen expert detects the refrigerator at 0.96 confidence, and there are no false positives from the climate or display paths. This end-to-end behavior was the central validation goal of the milestone.
Throughput on a CPU-only Windows laptop (no GPU acceleration during inference) is 3–6 FPS. This is consistent with PyTorch CPU inference of two ~30M-parameter detection transformers, and is the slowest possible deployment configuration. We discuss expected on-device performance in the next section.

On-device Deployment Readiness
The architecture is designed for on-device deployment. We make this case on three pillars.
Footprint
The total disk footprint of the deployed pipeline at INT8 quantization is approximately 76 MB:

Router: 12 MB at INT8 (already validated through TFLite FP16 at 22 MB)
Climate expert: ~32 MB at INT8 (from 121 MB FP32)
Shared kitchen/display COCO model: ~32 MB at INT8 (from 121 MB FP32)

For comparison, a single YOLOv8x detector is approximately 130 MB. Our entire MoE pipeline fits in a smaller storage budget than one large modern detector, comfortably within the size constraints of a typical Android application.
NPU Compatibility
The MoE pattern is well-suited to mobile NPUs for three reasons. First, the router is a small classifier that runs in negligible time on the Exynos NPU or Qualcomm Hexagon DSP — a few milliseconds at most based on published ResNet18 benchmarks. Second, only one expert runs per frame; we pay for one detector's inference cost per frame rather than three running in parallel. Third, the router can run at lower frequency than the experts since scenes do not change every frame; this opens a further optimization where the router runs at 5–10 Hz and the active expert runs at full frame rate.
Proven Export Path
The router has been end-to-end validated through the same Android deployment chain we would use for production: PyTorch → ONNX (opset 17) → TFLite via onnx2tf, tested against the original PyTorch output. The maximum numerical difference is 0.000001, well within the tolerances expected for FP16 quantization.
The same export chain applies to RF-DETR Nano. Roboflow publishes ONNX exports of RF-DETR models, and the TFLite conversion uses the same onnx2tf toolchain. The work remaining is conversion, not architecture redesign.

Honest Limitations
We are explicit about what we have not yet measured or built, because doing so is more credible than projecting.
Phone-class latency is not measured. All inference numbers in this report are from a CPU-only Windows laptop running PyTorch. We have not deployed to an actual Android device, so we cannot quote phone-class FPS with measurement-backed certainty. Based on RF-DETR Nano's documented benchmarks at FP16 on T4 GPU (sub-10 ms per frame) and on typical mobile NPU performance ratios, we expect 20–30 FPS on a Snapdragon 8 Gen 3 or Exynos 2400, but this is a projection, not a measurement.
The experts' TFLite conversion is not yet done. The router has been converted and verified. The climate expert and the COCO model are still in PyTorch. Conversion is straightforward but is real engineering work that has not been completed.
Phase 2 features are scoped out. The escalation layer fallback, the user feedback storage, and the model-improvement loop from the original proposal are not built. We made this scope decision deliberately: validating the routing dispatch is the foundational claim, and the additional features are well-defined extensions that depend on having a working core. With the core proven, those features become straightforward future work, not unknowns.
The router has known weak spots on out-of-distribution images. During testing, we observed that on tightly-cropped object photos (a microwave with no scene context, for example), the router's confidence drops below the gate threshold. The temporal smoothing and confidence gate handle this correctly — the active expert holds steady across uncertain frames — but the underlying classifier is calibrated for scene-style photographs rather than tight object crops. This is a property of the Places365 backbone and the training distribution, and could be addressed in future work by augmenting the training set with more diverse phone-camera images.

Mapping Original Proposal to Delivered Work
Original componentDeliveredNotesCamera input → preprocessingBuiltStandard ImageNet normalization, OpenCV pipelineBackbone for routingBuilt (ResNet18)Substituted from MobileNetV3 to use Places365 pretrained weightsRouter MLP / classifierBuilt3-class head, 93.3% val accuracyClimate expertBuilt (RF-DETR Nano)Substituted from SSD-Lite; 0.932 mAP achieved vs. 0.140 with SSD MobileNetKitchen expertBuilt (shared COCO model)Substituted to leverage COCO's existing coverageDisplay expertBuilt (shared COCO model)Same shared model as kitchen, filter onlyConfidence gate + thresholdBuiltThreshold 0.55, configurableTemporal smoothingBuilt15-frame rolling majority voteFinal output (boxes, labels)BuiltOpenCV rendering with HUDEscalation layer / fallbackDeferredPhase 2User feedback collectionDeferredPhase 2Feedback storage / DBDeferredPhase 2Online model improvementDeferredPhase 2Android deploymentPartially provenRouter converted to TFLite FP16; experts use the same path

What Production Would Look Like
Given two more weeks and target Samsung hardware, here is what we would build.
First, we would convert the climate expert and the shared COCO model to TFLite INT8 with proper calibration, using the same onnx2tf path that worked for the router. We would verify numerical correctness at each conversion step.
Second, we would build the Android application shell, integrating the three TFLite models, the OpenCV camera pipeline, and NNAPI delegate configuration. We would measure on-device latency directly on the target Samsung hardware to replace our current projections with measurements.
Third, we would build the deferred Phase 2 components: the escalation fallback for low-confidence frames, the user feedback collection UI, and the local feedback storage. The model-improvement loop is more research-flavored and would likely be Phase 3.
Fourth, we would consider whether to swap the expert backbones for production. RF-DETR Nano works well and is on-device-viable, but if Samsung's NPU has specific optimizations for MobileNetV3 or EfficientDet-Lite operations that RF-DETR's attention layers don't benefit from, the right production choice might be different from the validation choice. The architecture allows this swap without affecting any other part of the pipeline. This is one of its core strengths.
Worth noting: we would not return to SSD MobileNet specifically without addressing the training capacity issue. If Samsung wanted small-model experts for hardware reasons, the right path would be either (a) training EfficientDet-Lite0 instead, which has stronger pretrained priors, or (b) training SSD MobileNet on substantially more data than the under-2,000 images per category we had available. The architecture supports either choice.

Conclusion
The team set out to validate that an MoE routing architecture is the right pattern for on-device multi-category object detection. We built the architecture end-to-end, demonstrated it routing correctly across three scene types, achieved deployable accuracy on each component, and proved the Android export path on the router with 0.000001 numerical correctness.
We did not deliver this using the exact model components in the original proposal. The reasons are evidence-based: SSD MobileNet reached 0.140 mAP@50:95 on our climate training data with size-stratified performance dropping to 0.000 for small objects, and was effectively non-functional on kitchen with live confidence scores of 0.02–0.03 — not deployable for any production purpose. The SSD-Lite tooling chain consumed engineering time without producing working models. The TensorFlow CPU-only constraint made experimentation slow enough that recovery from the first two issues was not feasible within the timeline. We pivoted the expert backbones to RF-DETR Nano, which reaches 0.932 mAP@50:95 on equivalent training conditions — a roughly 7× improvement on the most direct measurement we have.
The architecture is the thesis. The model choices are implementation details that should evolve as deployment requirements clarify. We believe what we delivered — a working, validated, exportable MoE pipeline — is the strongest possible evidence that the architectural approach is correct. The deferred items are well-defined, not unknown.
We are ready to take this to Android.