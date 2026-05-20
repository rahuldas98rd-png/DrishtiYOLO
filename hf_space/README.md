---
title: DrishtiYOLO
emoji: 🛺
colorFrom: orange
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: agpl-3.0
tags:
  - object-detection
  - yolo
  - ultralytics
  - indian-roads
  - computer-vision
---

# DrishtiYOLO

YOLO11s fine-tuned on the Indian Driving Dataset (IDD) to detect road objects
specific to Indian streets — auto-rickshaws, riders, cattle, electric vehicles,
and more that off-the-shelf COCO models miss.

## How to use

Upload any Indian road scene image and click **Detect**.
Adjust the confidence slider to filter weak predictions.

## Model details

| Field        | Value                                   |
|--------------|-----------------------------------------|
| Architecture | YOLO11s                                 |
| Dataset      | IDD Detection (~10k images, 15 classes) |
| Training     | Colab free tier, T4 GPU                 |
| Input size   | 640 × 640                               |
| License      | AGPL-3.0                                |

## Environment variable

Set `MODEL_REPO` to `your-username/DrishtiYOLO` in the Space settings,
or update the default in `app.py` before pushing.
