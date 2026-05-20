# DrishtiYOLO

**YOLOWorld zero-shot detection fine-tuned for Indian roads** — identifying road objects
that off-the-shelf COCO models miss: auto-rickshaws, riders, cattle, electric vehicles, and more.

> _"Drishti" (दृष्टि) means vision in Sanskrit._

---

## Why this exists

Standard YOLO models trained on COCO perform poorly on Indian streets because
COCO contains no auto-rickshaws, cycle-rickshaws, riders (persons on two-wheelers),
or the chaotic mix of vehicle types common in South Asian traffic.

DrishtiYOLO uses **YOLOWorld open-vocabulary detection** with IDD class prompts —
no fine-tuning or training required. The core question: _can zero-shot detection
outperform a COCO-trained model on Indian roads?_

## Results

| Model                        | mAP50 | mAP50-95 | FPS (T4) |
|------------------------------|-------|----------|----------|
| COCO baseline (yolo11s)      | —     | —        | —        |
| DrishtiYOLO (YOLOWorld-x)    | —     | —        | —        |

Side-by-side comparison video: `assets/comparison.mp4`

## Detected classes (15)

`auto_rickshaw` · `bicycle` · `bus` · `car` · `caravan` · `electric_vehicle`
· `motorcycle` · `person` · `rider` · `traffic_light` · `traffic_sign`
· `truck` · `vehicle_fallback` · `animal` · `wheelchair`

Classes marked **bold** are India-specific (absent from COCO):
**auto_rickshaw**, **rider**, **vehicle_fallback**, **animal** (partial),
**caravan**, **electric_vehicle**, **wheelchair**

## Repo structure

```
DrishtiYOLO/
├── scripts/
│   ├── webcam_demo.py        # real-time inference via Iriun virtual camera
│   ├── side_by_side_video.py # COCO vs DrishtiYOLO comparison video
│   └── download_clip.py      # yt-dlp wrapper for CC Indian traffic clips
├── notebooks/
│   ├── drishti_eval.ipynb    # mAP, per-class AP, latency, error analysis
│   └── drishti_demo.ipynb    # Gradio demo
├── configs/
│   └── idd.yaml              # IDD dataset YAML (for eval)
├── hf_space/
│   ├── app.py                # Gradio Space (HF CPU tier)
│   └── requirements.txt
├── assets/                   # GIFs, thumbnails, charts (no binaries in git)
├── MODEL_CARD.md
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
pip install ultralytics opencv-python
```

**Live webcam demo** (via Iriun):
```bash
python scripts/webcam_demo.py
# Optional: --model yolov8x-worldv2.pt for higher accuracy
# Optional: --device 1  if camera isn't on index 0
```

**Side-by-side comparison video**:
```bash
python scripts/side_by_side_video.py \
  --clip assets/clips/traffic.mp4 \
  --out assets/comparison.mp4
```

No weights to download — YOLOWorld fetches its own model on first run.

## How it works

YOLOWorld combines YOLO's speed with CLIP-based vision-language alignment.
Instead of fixed class IDs, it accepts **text prompts** at inference time:

```python
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8x-worldv2.pt")
model.set_classes([
    "auto rickshaw", "rider", "electric vehicle",
    "animal", "bicycle", "bus", ...
])
results = model("indian_traffic.jpg")
```

This means the 15 IDD classes — including India-specific ones with no COCO equivalent —
are detected without any training.

## Dataset (for evaluation only)

**Indian Driving Dataset (IDD)** — Detection split  
~10,000 images, 15 classes, non-commercial research licence.  
Registration at [idd.insaan.iiit.ac.in](https://idd.insaan.iiit.ac.in)

## Stack

| Component    | Tool                          | Cost  |
|--------------|-------------------------------|-------|
| Detection    | YOLOWorld (Ultralytics)       | ₹0    |
| Dataset      | IDD (eval only, free reg.)    | ₹0    |
| Demo hosting | Hugging Face Spaces CPU       | ₹0    |
| Writeup      | Dev.to / Hashnode             | ₹0    |
| **Total**    |                               | **₹0**|

## License

Model weights: AGPL-3.0 (Ultralytics).  
Training data: IDD non-commercial research licence.  
Scripts and notebooks: MIT.
