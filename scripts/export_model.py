"""
Export a trained YOLO model to ONNX for the Lambda inference function.

By default it exports YOUR landmine detector. Point it at your trained weights
(`best.pt`, produced by the training notebook in the landmine-detection-yolo repo):

    pip install ultralytics onnx onnxruntime
    python scripts/export_model.py --weights /path/to/best.pt --names landmine

If you don't have your weights handy yet, you can export a pretrained COCO model so
the API is demoable end-to-end immediately:

    python scripts/export_model.py --pretrained yolo11n.pt

Outputs:
    app/model/model.onnx
    app/model/labels.json   (class names, indexed by class id)

Notes
-----
- The handler letterboxes inputs to 640x640 to match YOLO training. If you trained at a
  different imgsz, pass --imgsz and set INPUT_SIZE in app/handler.py to match.
- ONNX is exported with opset 12 and simplified for portable CPU inference on Lambda.
"""

import argparse
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "model")


def export(weights: str, names, imgsz: int):
    from ultralytics import YOLO

    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(weights)

    # Ultralytics writes <weights_stem>.onnx next to the .pt; we move it to app/model/.
    exported = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
    onnx_src = str(exported)
    onnx_dst = os.path.join(OUT_DIR, "model.onnx")
    os.replace(onnx_src, onnx_dst)
    print(f"Exported ONNX -> {onnx_dst}")

    # Resolve class names: prefer the model's own names, else the --names override.
    model_names = getattr(model, "names", None)
    if model_names:
        labels = [model_names[i] for i in sorted(model_names)]
    elif names:
        labels = names
    else:
        labels = ["object"]
    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f)
    print(f"Wrote {len(labels)} label(s) -> {os.path.join(OUT_DIR, 'labels.json')}: {labels}")


def main():
    p = argparse.ArgumentParser(description="Export a YOLO model to ONNX for Lambda.")
    p.add_argument("--weights", help="Path to your trained best.pt")
    p.add_argument(
        "--pretrained",
        help="Pretrained checkpoint name to download instead of using your own (e.g. yolo11n.pt)",
    )
    p.add_argument(
        "--names",
        nargs="*",
        default=["landmine"],
        help="Class names if the checkpoint doesn't carry them (default: landmine)",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Export input size (default 640)")
    args = p.parse_args()

    weights = args.weights or args.pretrained
    if not weights:
        raise SystemExit(
            "Provide --weights /path/to/best.pt (your trained model) "
            "or --pretrained yolo11n.pt (a downloadable demo model)."
        )
    export(weights, args.names, args.imgsz)


if __name__ == "__main__":
    main()
