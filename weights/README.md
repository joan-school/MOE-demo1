# Weights

This folder is empty in the repo. Model weights are hosted on Google Drive.

**Download from:** [\[PASTE GOOGLE DRIVE FOLDER LINK HERE\]](https://drive.google.com/drive/folders/1rtMoBmmxXoyaTl400hZutJiP4M7idlti)

Place the following files here:

- `router_best.pth` — 45 MB, ResNet18 + 3-class head trained on Places365 backbone
- `class_to_idx.json` — 50 B, router class index mapping
- `climate_expert_v1_map0920.pth` — 121 MB, RF-DETR Nano fine-tuned on AC dataset (mAP 0.932)

The COCO-pretrained RF-DETR Nano weights used for kitchen and display experts are auto-downloaded by the `rfdetr` package on first use.