"""
webcam_demo.py — real-time DrishtiYOLO inference via YOLOWorld zero-shot detection.

Uses YOLOWorld with IDD class prompts — no fine-tuned weights needed.
Iriun exposes a standard V4L2 / DirectShow virtual device, so OpenCV
reads it like any webcam. If the default device index (0) doesn't work,
pass --device 1 or 2 until you get the phone feed.

Usage:
    python scripts/webcam_demo.py
    python scripts/webcam_demo.py --device 1 --conf 0.3
    python scripts/webcam_demo.py --model yolov8x-worldv2.pt  # larger, more accurate

Keys while running:
    q  — quit
    s  — save current frame to assets/snapshots/
    r  — toggle recording (saves to assets/recordings/)
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLOWorld


SNAPSHOT_DIR  = Path("assets/snapshots")
RECORDING_DIR = Path("assets/recordings")
FONT          = cv2.FONT_HERSHEY_SIMPLEX

IDD_CLASSES = [
    "auto rickshaw", "bicycle", "bus", "car", "caravan",
    "electric vehicle", "motorcycle", "person", "rider",
    "traffic light", "traffic sign", "truck",
    "vehicle", "animal", "wheelchair",
]


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="DrishtiYOLO live webcam demo (YOLOWorld)")
    parser.add_argument("--model",  default="yolov8s-worldv2.pt",
                        help="YOLOWorld variant (yolov8s/m/l/x-worldv2.pt). "
                             "Smaller = faster; larger = more accurate.")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (Iriun is usually 0 or 1)")
    parser.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz",  type=int,   default=640,  help="Inference image size")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = YOLOWorld(args.model)
    model.set_classes(IDD_CLASSES)
    print(f"Classes set: {IDD_CLASSES}")
    print(f"Opening camera device {args.device}...")

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera device {args.device}. "
            "Make sure Iriun Webcam app is running on your phone and connected."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0

    SNAPSHOT_DIR.mkdir(parents=True,  exist_ok=True)
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)

    writer    = None
    recording = False
    prev_time = time.perf_counter()

    print("Press  q=quit  s=snapshot  r=toggle recording")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed — check Iriun connection.")
            break

        results   = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        annotated = results[0].plot()

        now       = time.perf_counter()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_fps(annotated, fps)

        if recording and writer is not None:
            writer.write(annotated)
            cv2.circle(annotated, (annotated.shape[1] - 20, 20), 8, (0, 0, 255), -1)

        cv2.imshow("DrishtiYOLO — live (q=quit, s=snap, r=record)", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = SNAPSHOT_DIR / f"snap_{ts}.jpg"
            cv2.imwrite(str(snap_path), annotated)
            print(f"Snapshot saved: {snap_path}")

        elif key == ord("r"):
            if not recording:
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                rec_path = RECORDING_DIR / f"rec_{ts}.mp4"
                h, w     = annotated.shape[:2]
                fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                writer   = cv2.VideoWriter(str(rec_path), fourcc, fps_cap, (w, h))
                recording = True
                print(f"Recording started: {rec_path}")
            else:
                recording = False
                if writer:
                    writer.release()
                    writer = None
                print("Recording stopped.")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
