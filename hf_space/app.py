"""
Hugging Face Space — DrishtiYOLO
CPU-tier Gradio app using YOLOWorld zero-shot detection.
No custom weights needed — model downloads automatically from Ultralytics on first run.

Deploy steps:
1. Push this Space to hf.co/<your-username>/DrishtiYOLO.
2. HF will build and run it automatically — no model upload required.
"""

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld


IDD_CLASSES = [
    "auto rickshaw", "bicycle", "bus", "car", "caravan",
    "electric vehicle", "motorcycle", "person", "rider",
    "traffic light", "traffic sign", "truck",
    "vehicle", "animal", "wheelchair",
]

INDIA_SPECIFIC = {
    "auto rickshaw", "rider", "vehicle",
    "caravan", "electric vehicle", "wheelchair",
}

# Small variant for CPU-tier HF Space inference speed
MODEL = YOLOWorld("yolov8s-worldv2.pt")
MODEL.set_classes(IDD_CLASSES)


def predict(image: np.ndarray, conf: float) -> tuple[Image.Image, str]:
    results   = MODEL(image, conf=conf, verbose=False)
    annotated = results[0].plot()

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return Image.fromarray(annotated), "No detections above threshold."

    names  = results[0].names  # {0: 'auto rickshaw', 1: 'bicycle', ...}
    counts: dict[str, int] = {}
    for cls_id in boxes.cls.cpu().numpy().astype(int):
        name = names.get(cls_id, str(cls_id))
        counts[name] = counts.get(name, 0) + 1

    parts = []
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        tag = " *" if name in INDIA_SPECIFIC else ""
        parts.append(f"{cnt}× {name}{tag}")
    summary = "  ·  ".join(parts) + "\n(* = India-specific class, absent from COCO)"

    return Image.fromarray(annotated), summary


with gr.Blocks(title="DrishtiYOLO — Indian Road Scene Detection") as demo:
    gr.Markdown(
        "## DrishtiYOLO\n"
        "**YOLOWorld zero-shot detection with [IDD](https://idd.insaan.iiit.ac.in/) class prompts**\n\n"
        "Off-the-shelf COCO models miss India-specific road objects: auto-rickshaws, riders, "
        "cattle, and overloaded two-wheelers. DrishtiYOLO uses open-vocabulary detection to "
        "identify all 15 IDD classes — **no fine-tuning required**.\n\n"
        "_Classes marked with * are India-specific and absent from COCO._"
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp  = gr.Image(label="Upload an Indian road scene", type="numpy")
            conf = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="Confidence threshold")
            btn  = gr.Button("Detect", variant="primary")
        with gr.Column(scale=1):
            out_img  = gr.Image(label="Detections")
            out_text = gr.Textbox(label="Object counts", lines=3)

    btn.click(predict, inputs=[inp, conf], outputs=[out_img, out_text])

    gr.Markdown(
        "---\n"
        "**Model:** YOLOWorld (yolov8s-worldv2) | **Classes:** 15 IDD prompts | "
        "**Approach:** Zero-shot open-vocabulary | "
        "**Repo:** [GitHub](https://github.com/your-username/DrishtiYOLO)"
    )

if __name__ == "__main__":
    demo.launch()
