# Serverless Landmine Detection API on AWS

A production-style, fully serverless **object-detection** API. Send a thermal image, get
back bounding boxes for detected landmines plus an annotated image — every request logged
for observability, all running on AWS with no servers to manage.

```
browser demo / client ──POST /predict──▶ API Gateway (HTTP API) ──▶ AWS Lambda (container image)
                                                                      │  ONNX Runtime · YOLO11
                                                                      │  letterbox → infer → NMS
                                                                      ▼
                                                                 Amazon DynamoDB  ◀──GET /stats──
                                                                 (inference logs)
```

## Why this exists

A trained model is only useful once it is deployed and serving real requests. This project
takes the YOLO11 landmine detector from
[landmine-detection-yolo](https://github.com/dawodghifari/landmine-detection-yolo)
(thermal UAV imagery, mAP@50 0.68), exports it to ONNX, packages it as a container-image
Lambda, and exposes it through API Gateway — with DynamoDB capturing a log of every
inference (detection count, top confidence, latency). A static browser demo draws the
returned boxes over the uploaded image.

## Stack

| Concern | Choice |
|---|---|
| Compute | AWS Lambda (container image, so large ML deps fit) |
| API | Amazon API Gateway (HTTP API) |
| Model | YOLO11 (Ultralytics), exported to ONNX |
| Inference | ONNX Runtime (CPU) + NumPy letterbox/NMS post-processing |
| Storage / logging | Amazon DynamoDB (on-demand) |
| Infrastructure as code | AWS SAM |
| Packaging | Docker |
| Demo | Static HTML/JS page (`web/index.html`) — no build step |

## Endpoints

- `POST /predict` — body `{"image_base64": "<base64>", "conf": 0.25, "iou": 0.45, "annotate": true}`
  → detected boxes (`[x1,y1,x2,y2]` in original-image pixels) + latency, optional annotated
  PNG, logged to DynamoDB.
- `GET /stats` — most recent inference logs (count, top confidence, latency).

Example `/predict` response:

```json
{
  "id": "…",
  "latency_ms": 180.3,
  "num_detections": 2,
  "detections": [
    {"label": "landmine", "class_id": 0, "confidence": 0.81, "box": [412.0, 233.5, 470.2, 291.8]},
    {"label": "landmine", "class_id": 0, "confidence": 0.54, "box": [120.4,  88.1, 165.0, 140.9]}
  ],
  "annotated_image_base64": "<png>"
}
```

## Quick start

See **[DEPLOY.md](DEPLOY.md)** for the full step-by-step. In short:

```bash
# 1. Export your trained YOLO weights to ONNX (or a pretrained model to demo immediately)
python scripts/export_model.py --weights /path/to/best.pt --names landmine
#   …or:  python scripts/export_model.py --pretrained yolo11n.pt

# 2. Build + deploy
sam build
sam deploy --guided

# 3. Test
python scripts/test_request.py <API_BASE_URL> test_thermal.jpg
#   then open web/index.html, paste your API URL, and drop an image
```

## Serving your own weights

The handler is **class-agnostic**: it reads the number of classes straight from the model
output and labels from `model/labels.json`, so the same code serves any YOLOv8/YOLO11
detection model. To serve your landmine detector, point `export_model.py` at the `best.pt`
produced by the training notebook. If you trained at a non-default image size, pass
`--imgsz` and set `INPUT_SIZE` in `app/handler.py` to match.

## Project layout

```
template.yaml            # AWS SAM: Lambda + API Gateway + DynamoDB
app/
  Dockerfile             # Lambda container image
  handler.py             # routing, letterbox preprocess, ONNX inference, NMS, annotate, logging
  requirements.txt
  model/                 # model.onnx + labels.json (generated, git-ignored)
scripts/
  export_model.py        # YOLO (.pt) -> ONNX exporter
  test_request.py        # send a local image to the API, save the annotated result
web/
  index.html             # browser demo: drop an image, see boxes drawn
DEPLOY.md                # step-by-step deployment + teardown
```

## Notes / next steps

- `/stats` uses a small DynamoDB scan for demo simplicity; for scale, add a GSI on
  `timestamp` and query it.
- Cold starts: the first request after idle pays model-load time; provisioned concurrency
  removes it if needed.
- Pre-processing letterboxes to 640×640 (aspect-ratio preserving, grey padding) to match
  YOLO training; boxes are mapped back to original-image pixels before they're returned.
- The negative-image (false-positive) behaviour discussed in the model repo applies here
  too — tune `conf` per request to trade recall against precision.

---

*Built by Dawod Ghifari to deploy ML models as serverless APIs on AWS.*
