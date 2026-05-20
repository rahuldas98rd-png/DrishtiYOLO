# Model Card — DrishtiYOLO

## Model details

| Field             | Value                                                         |
|-------------------|---------------------------------------------------------------|
| Name              | DrishtiYOLO                                                   |
| Base architecture | YOLO11s (Ultralytics)                                         |
| Fine-tuned on     | Indian Driving Dataset (IDD) Detection split                  |
| Classes           | 15 (see list below)                                           |
| Input resolution  | 640 × 640                                                     |
| Framework         | Ultralytics ≥ 8.3, PyTorch ≥ 2.0                             |
| Trained by        | [Your Name]                                                   |
| License           | AGPL-3.0 (Ultralytics); IDD data: non-commercial research     |

## Intended use

Detecting road objects in Indian urban and semi-urban traffic scenes,
particularly objects absent from or under-represented in the COCO benchmark:
auto-rickshaws, cycle-rickshaws, riders (persons on two-wheelers), cattle,
electric vehicles, and miscellaneous vehicle types common on Indian roads.

**Out-of-scope uses:**
- Production autonomous driving or ADAS systems without further validation
- Surveillance or law-enforcement applications
- Non-Indian road scenes (model may generalise, but was not evaluated there)

## Training data

**Dataset:** Indian Driving Dataset (IDD) — Detection split
**Source:** idd.insaan.iiit.ac.in (IIIT Hyderabad)
**License:** Research / non-commercial
**Size:** ~10,000 images across train / val / test splits
**Capture conditions:** Daytime road footage from Hyderabad, India

### Classes
| ID | Class             | In COCO? |
|----|-------------------|----------|
| 0  | auto_rickshaw     | No       |
| 1  | bicycle           | Yes      |
| 2  | bus               | Yes      |
| 3  | car               | Yes      |
| 4  | caravan           | No       |
| 5  | electric_vehicle  | No       |
| 6  | motorcycle        | Yes      |
| 7  | person            | Yes      |
| 8  | rider             | No       |
| 9  | traffic_light     | Yes      |
| 10 | traffic_sign      | Yes      |
| 11 | truck             | Yes      |
| 12 | vehicle_fallback  | No       |
| 13 | animal            | Partial  |
| 14 | wheelchair        | No       |

## Evaluation results

> _Fill in after training is complete._

| Model                   | mAP50 | mAP50-95 |
|-------------------------|-------|----------|
| COCO baseline (yolo11s) | —     | —        |
| DrishtiYOLO (yolo11s)   | —     | —        |

Per-class AP chart: `assets/per_class_ap50.png`

## Limitations

- **Nighttime / low-light:** IDD is predominantly daytime footage; performance
  degrades significantly in poor lighting.
- **Small objects at distance:** YOLO11s at 640 px struggles with objects
  smaller than ~16×16 px in the original image.
- **Heavy occlusion:** Crowded intersections with stacked vehicles cause
  missed detections and merged bounding boxes.
- **Data imbalance:** `caravan`, `electric_vehicle`, and `wheelchair` have
  very few IDD examples — expect low AP on these classes.
- **Geographic scope:** Validated on Hyderabad-area footage; performance on
  other Indian cities (narrow lanes, hill roads, coastal traffic) is untested.

## Ethical considerations

- This model detects people (`person`, `rider`). Do not deploy in systems
  that track, surveil, or profile individuals without explicit consent.
- IDD annotations were collected under a research licence; redistribution
  of derived model weights for commercial use requires re-licensing.
- Class `animal` covers cattle, dogs, and other animals common on Indian
  roads. The model is not validated for wildlife detection or animal welfare
  applications.

## How to use

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("your_image.jpg", conf=0.25)
results[0].show()
```

## Citation

```bibtex
@dataset{idd2018,
  author    = {Varma, Girish and Subramanian, Anbumani and Namboodiri, Anoop
               and Chandraker, Manmohan and Jawahar, C.V.},
  title     = {IDD: A Dataset for Exploring Problems of Autonomous Navigation
               in Unconstrained Environments},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2019},
}
```
