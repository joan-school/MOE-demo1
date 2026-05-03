https://universe.roboflow.com/joans-workspace-i8meq/moe-router


How to Improve the Router
This is a guide for teammates who want to improve the router's accuracy and robustness. The router currently sits at 93.3% validation accuracy, but it has known weak spots on phone-camera images that don't look like the scene-style photos it was trained on. The fastest improvement path is better training data, not architecture changes. This guide walks you through how to collect that data and retrain.

What we know about the current router
Before you start, here's what's working and what's not:
Working well:

Wide scene shots of an AC unit on a wall — routes to climate at 0.95+ confidence
Wide kitchen scenes — routes to kitchen correctly
Living room with a TV in context — routes to display at ~0.7 confidence

Working poorly:

Tight close-up object crops with no surrounding context (e.g., just a microwave filling the frame) — confidence drops below the 0.55 gate, occasionally misroutes
Phone-camera lighting / framing that differs from Roboflow Universe stock photos
Scenes where two product categories are visible at once (e.g., a kitchen with a TV mounted on the wall)

Why this happens: The current training set has 514 images sourced from Roboflow Universe scene-classification datasets. These are mostly stock-style scene photos, not phone snaps. The model learned what those kinds of images look like, but the test distribution (real phone video walked through real homes) is different.
This is fixable with more diverse training data.

Approach 1: Add phone-camera training images (recommended first try)
This is the highest-ROI improvement. You don't need to change the model, the training script, or any code — you just need to add more images to the existing Roboflow project and retrain.
Goal
Add ~50–80 new phone-camera images per class to the existing dataset, biased toward the scenarios that currently fail. After retraining, target validation accuracy of 96%+ and (more importantly) confidence ≥0.7 on phone-camera tests.
Step 1: Understand the three classes
The router classifies each frame into exactly one of three scene categories:
ClassWhat goes hereWhat does NOT go hereclimateAir conditioner units (wall-mounted split units, window ACs, portable ACs)Heaters, fans, air purifierskitchenRefrigerators, microwaves, ovens, kitchen scenes with appliances visibleJust a sink, just countertops with no appliancedisplayTVs, computer monitors, large display screensPhones, tablets, laptops
If a scene has multiple categories visible, pick the one that's most prominent in the frame when you label.
Step 2: Take the photos — what we need
You're aiming for diverse, phone-camera-realistic images. The currently underrepresented scenarios are:
High priority (these are where the router fails today)

Tight object crops — the appliance fills 60%+ of the frame, minimal background. Take 10 of these per class.
Phone-walking-around angles — slight motion blur, off-axis perspectives, not perfectly framed. 15 per class.
Mixed-light scenes — same room photographed in daylight, evening, with overhead lights only. 10 per class.
Distance variation — same object photographed from 1m, 2m, 4m away. 10 per class.

Medium priority

Cluttered backgrounds — appliance with other stuff visible (towels on the fridge, clutter on counters near the microwave, etc.). 10 per class.
Different appliance models — not just your fridge, but anyone's fridge you can photograph. Variety in shape, color, brand. Spread across the others.

Photography rules

Use phone cameras, not stock photos. The whole point is to match the deployment distribution.
Hold the phone the way someone would naturally use the demo — handheld, not on a tripod.
Resolution does not matter much; the router resizes everything to 224×224 anyway. But don't go below 480×480 — too low and Roboflow may not accept the upload.
Do not edit, filter, or color-correct the photos. We want what the camera actually captures.
Shoot in .jpg, not .heic (Roboflow handles both but .jpg is friction-free).

Diversity checklist (per class)
Before uploading, scroll through your photos and mentally check that you have:

☐ Multiple different physical locations (your kitchen, a friend's kitchen, an office break room)
☐ At least 3 different lighting conditions
☐ A mix of close, medium, and far framing
☐ Some "messy" real-world backgrounds, not just clean staged shots
☐ Different times of day

If you only photographed the same fridge from your kitchen at noon ten times, you've added redundancy, not diversity. The model already knows your fridge.
Step 3: Upload to the existing Roboflow project
The router dataset is on Roboflow as a single-label classification project named moe-router. Ask whoever owns the Roboflow workspace to add you as a collaborator if you don't have access.

Open the moe-router project in Roboflow
Click Upload → drag in your batch of photos
Critical: choose the correct class label during upload — this is a single-label classification project, so each image gets exactly one of: climate, display, kitchen
Roboflow will auto-resize and process the images
Once uploaded, add them to the training set, not the validation set — we want validation to remain a clean signal of generalization
Click Generate → create a new dataset version

When generating the dataset version, use these settings to match the original training:

Augmentations: turn on horizontal flip, brightness ±15%, slight rotation ±5° — but nothing aggressive (no large rotations, no extreme color jitter, no random erasing). The model learns better from realistic variations.
Resize: 224×224 stretch or fit-within (Roboflow will handle this, default is fine)
Output format: Folder Structure (one folder per class) — this is what classification training expects

Step 4: Retrain
Open the original training Colab notebook (ask the team for the link if you don't have it). The training is fast — ~5 minutes on a T4 GPU.
Key things to keep the same:

ResNet18 backbone, frozen
Final 3-class linear layer
Learning rate 1e-3, Adam optimizer
15 epochs (more than this and you're overfitting on a small dataset)
Stratified 80/20 train/val split

Key things to verify after training:

Validation accuracy prints at the end of training. Should be ≥93% (existing baseline). If it's worse than 93%, your new images may have label issues — go back and check.
Save checkpoint as router_v2.pth — do NOT overwrite the existing router_best.pth until you've confirmed the new model is better in real testing.

Step 5: Verify on your phone-camera test set
Before declaring victory, run router_check.py on your held-out phone-camera test images (the ones we know fail today: microwave.jpg, tv1.jpg, etc. — these should already be in test_assets/).
bashpython router_check.py
What success looks like:
ImageOld routerNew router (target)ac1.jpgclimate 0.98 ✓climate 0.98 ✓ (no regression)ac2.jpgclimate 0.96 ✓climate 0.96 ✓fridge.jpgkitchen 0.61 ✓ borderlinekitchen ≥ 0.80 ✓microwave.jpgclimate 0.40 ❌kitchen ≥ 0.70 ✓tv1.jpgkitchen 0.46 ❌display ≥ 0.70 ✓tv2.jpgdisplay 0.69 ✓display ≥ 0.85 ✓
If validation accuracy went up but the failing test images still fail, the new training data wasn't diverse enough or wasn't representative of the failure mode. Add more, focus on the failing scenarios.
If new images pushed correct cases above 0.7 but you broke something that previously worked (regression), you may have over-corrected — the model is now too biased toward phone-camera style and missed something about the original training distribution. Retrain with a more balanced mix.
Step 6: Replace the production weights
Once the new router is convincingly better:

Rename router_v2.pth → router_best.pth in the Drive weights folder
Update version notes in the team doc / README
Re-run the export pipeline (PyTorch → ONNX → TFLite) — see docs/ROUTER.md for the exact commands
Verify the new TFLite model still matches the new PyTorch model within 0.000001
Replace router_fp16.tflite in the Drive folder


Approach 2: Try a different backbone (more advanced)
If Approach 1 hits a ceiling and the router still has weak spots, the next thing to try is swapping the backbone. The current ResNet18 + Places365 weights are good for scene-style images but may not be optimal for phone-camera distributions.
Candidate backbones to try (all small, all license-clean):
BackbonePretrained onWhy try itMobileNetV3-LargeImageNetDesigned for mobile, smaller than ResNet18EfficientNet-B0ImageNetBetter accuracy/size tradeoff than MobileNetConvNeXt-TinyImageNet-22kModern architecture, strong feature extractor
The training procedure is identical — same frozen-backbone, linear-probe approach. Only the import line changes:
python# Current
import torchvision.models as tvm
backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
# Replace with:
backbone = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT)
You'll need to adjust the head dimensions (MobileNetV3 output is 1280-d, not 512-d) and re-export the model — but the training time stays at ~5 minutes.
Caveat: None of these backbones have CC BY 4.0 Places365 weights as far as we know. ImageNet-pretrained weights are BSD/Apache (license-clean) but were trained on object-centric images, not scene-centric. They may actually do worse on scene classification than Places365 ResNet18 — this is an empirical question you'd need to test.
This is research-level work, not a "weekend hack." Try Approach 1 first.

Approach 3: Add multi-frame context (even more advanced)
Our current router classifies each frame independently. The temporal smoother stitches the per-frame decisions into a stable choice, but the model itself never sees more than one frame at a time.
A more sophisticated router could take 3 or 5 sequential frames as input and produce a single decision over them. This would help with:

Scenes that take a moment to fully come into view (you walk through a doorway, the kitchen reveals itself over 4–5 frames)
Disambiguating ambiguous single-frame views by leveraging the temporal context

This is a real architectural change and outside the scope of "improve the existing router." Mention it as future work, don't try to ship it for the Samsung milestone.

What NOT to do
A few common traps when improving classifiers:
Don't unfreeze the backbone. With only ~600 training images, fine-tuning a 12M-parameter ResNet18 will overfit catastrophically. Keep the linear-probe approach. The whole reason it works is because there's nothing for the model to overfit on.
Don't add aggressive augmentation. Random erasing, color jitter, large rotations — these all help when you have a lot of data and want to prevent overfitting. We have the opposite problem: we have too few real-world samples and need real diversity, not synthetic diversity. Keep augmentation light (flip + small brightness/rotation).
Don't train for 50+ epochs. The model converges in 10–15 epochs on this dataset. Longer training lets the small head memorize the training set, which makes validation accuracy look good but real-world performance worse.
Don't change the input preprocessing. The 224×224 resize, ImageNet normalization, etc. is part of the inference contract. If you change preprocessing during training, you have to change it everywhere — Python demo, TFLite Android app — or things will silently break.
Don't add classes without team agreement. If you think the router needs a fourth class (e.g., "lighting" or "audio"), discuss with the team first. Adding a class is a breaking change to every downstream component.
Don't blindly trust validation accuracy. A 95% validation number means little if the validation set is from the same distribution as training. The real test is router_check.py on phone-camera images you haven't seen. That's what reflects deployment behavior.

Photography quick reference (print this)
When you're standing in your kitchen with your phone, here's the cheat sheet:
Aim forAvoid50–80 photos per classJust 5–10 (not enough)Multiple different rooms / homesAll photos from one placeDifferent lighting (day, night, mixed)All photos in same conditionsClose, medium, far distancesAll photos same distanceSlight handheld imperfectionTripod-perfect framingCluttered real-world backgroundsClean staged scenes onlyPhone JPG straight from cameraEdited / filtered imagesSingle most prominent appliance per frameThree appliances in one shot (ambiguous)

Workflow summary

☐ Get added to the Roboflow moe-router workspace
☐ Take 50–80 phone-camera photos per class (climate, display, kitchen) using the diversity checklist
☐ Upload to Roboflow, label correctly during upload, add to training set
☐ Generate a new dataset version with light augmentation
☐ Open the training Colab notebook, run training (~5 min)
☐ Save the new checkpoint as router_v2.pth
☐ Run router_check.py on the existing phone-camera test images
☐ Compare results to the table in Step 5 above
☐ If better, re-export to TFLite (docs/ROUTER.md has the commands)
☐ Replace production weights in Drive, update README, notify team

Total time: probably 4–6 hours including photography. Most of that is taking the photos. The training and verification are fast.

Questions, blockers, escalation
If you hit:

Roboflow upload errors → check image format, file size; try smaller batch
Training Colab errors → most likely a package version drift; pin torch and torchvision to the versions in the original notebook
New router does worse than the old one → your new images probably have label noise. Open the Roboflow project, click through your uploads, double-check each label.
Numerical mismatch between PyTorch and TFLite after export → preprocessing pipeline drift. The 224 resize and ImageNet normalization must be byte-exact across both.

For anything else, ping in the team channel before sinking too much time. The router is small and fast to iterate on — if something feels stuck, it usually means the workflow is wrong, not that the model is broken.