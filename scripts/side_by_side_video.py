"""
side_by_side_video.py — run two models on the same clip and stack frames horizontally.

Left panel : COCO baseline (yolo11s.pt) — 80 standard classes, misses Indian objects.
Right panel: DrishtiYOLO  (YOLOWorld)   — same 15 IDD classes, zero-shot via text prompts.

This comparison is the core visual argument: can open-vocabulary zero-shot detection
outperform a COCO-trained model on Indian roads?

Usage:
    python scripts/side_by_side_video.py --clip assets/clips/traffic.mp4
    python scripts/side_by_side_video.py \
        --clip assets/clips/traffic.mp4 \
        --coco yolo11s.pt \
        --world yolov8x-worldv2.pt \
        --out assets/comparison.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld


LABEL_HEIGHT    = 40
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.9
FONT_THICKNESS  = 2
LABEL_COLOR     = (30,  30,  30)
LABEL_BG_LEFT   = (220, 220, 220)
LABEL_BG_RIGHT  = (180, 230, 180)

IDD_CLASSES = [
    "auto rickshaw", "bicycle", "bus", "car", "caravan",
    "electric vehicle", "motorcycle", "person", "rider",
    "traffic light", "traffic sign", "truck",
    "vehicle", "animal", "wheelchair",
]


def draw_label_bar(frame: np.ndarray, text: str, bg_color: tuple) -> np.ndarray:
    bar = np.full((LABEL_HEIGHT, frame.shape[1], 3), bg_color, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    x = (frame.shape[1] - tw) // 2
    y = (LABEL_HEIGHT + th) // 2
    cv2.putText(bar, text, (x, y), FONT, FONT_SCALE, LABEL_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return np.vstack([bar, frame])


def annotate_frame(model, frame: np.ndarray, conf: float) -> np.ndarray:
    results = model(frame, conf=conf, verbose=False)
    return results[0].plot()


def process(clip: Path, coco_model: YOLO, world_model: YOLOWorld,
            out_path: Path, conf: float, scale: float) -> None:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {clip}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    w     = int(orig_w * scale)
    h     = int(orig_h * scale)
    out_w = w * 2
    out_h = h + LABEL_HEIGHT

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    print(f"Processing {total} frames at {fps:.1f} fps...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (w, h))

        left  = annotate_frame(coco_model,  frame, conf)
        right = annotate_frame(world_model, frame, conf)

        left  = draw_label_bar(left,  "COCO baseline (yolo11s.pt)",       LABEL_BG_LEFT)
        right = draw_label_bar(right, "DrishtiYOLO (YOLOWorld zero-shot)", LABEL_BG_RIGHT)

        writer.write(np.hstack([left, right]))

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames done", end="\r")

    cap.release()
    writer.release()
    print(f"\nSaved to {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Side-by-side: COCO vs DrishtiYOLO (YOLOWorld)")
    parser.add_argument("--clip",  required=True,                  help="Input video path")
    parser.add_argument("--coco",  default="yolo11s.pt",           help="COCO baseline weights")
    parser.add_argument("--world", default="yolov8x-worldv2.pt",   help="YOLOWorld variant")
    parser.add_argument("--out",   default="assets/comparison.mp4", help="Output path")
    parser.add_argument("--conf",  type=float, default=0.25,       help="Confidence threshold")
    parser.add_argument("--scale", type=float, default=1.0,        help="Frame resize scale (e.g. 0.5)")
    args = parser.parse_args()

    print("Loading COCO baseline...")
    coco_model = YOLO(args.coco)

    print(f"Loading YOLOWorld ({args.world})...")
    world_model = YOLOWorld(args.world)
    world_model.set_classes(IDD_CLASSES)
    print(f"YOLOWorld classes: {IDD_CLASSES}")

    process(
        clip        = Path(args.clip),
        coco_model  = coco_model,
        world_model = world_model,
        out_path    = Path(args.out),
        conf        = args.conf,
        scale       = args.scale,
    )


if __name__ == "__main__":
    main()
