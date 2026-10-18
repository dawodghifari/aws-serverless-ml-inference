# Serverless YOLO Object-Detection API on AWS

![CI](https://github.com/dawodghifari/aws-serverless-ml-inference/actions/workflows/ci.yml/badge.svg)

A fully serverless **object-detection** API. Send an image, get back bounding boxes plus an
annotated image, with every inference logged to DynamoDB, all on AWS with no servers to
manage. Built to serve a YOLO11 landmine detector, it was deployed with a pretrained COCO
YOLO11n (see [Limitations](#limitations)); the handler serves either without code changes.

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
is the deployment path for the YOLO11 landmine detector from
[landmine-detection-yolo](https://github.com/dawodghifari/landmine-detection-yolo)
(thermal UAV imagery, test mAP@50 0.904): export to ONNX, package as a container-image
Lambda, expose through API Gateway, and capture a DynamoDB log of every inference (detection
count, top confidence, latency). The pipeline is model-agnostic. The stack was deployed with a
pretrained COCO YOLO11n, because the trained landmine weights were not to hand at deploy
time. A static browser demo draws the returned boxes over the uploaded image.

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

Example `/predict` response, produced by `app/handler.py` run locally with the pretrained
COCO YOLO11n on Ultralytics' `bus.jpg` sample image (annotated PNG elided; the latency is a
local CPU figure, not a Lambda one):

```json
{
  "id": "6973286e-f7d5-4f07-b104-2aa4d9e215df",
  "latency_ms": 60.08,
  "num_detections": 5,
  "detections": [
    {"label": "bus",    "class_id": 5, "confidence": 0.9399, "box": [16.3, 228.4, 798.1, 734.2]},
    {"label": "person", "class_id": 0, "confidence": 0.9014, "box": [48.3, 398.7, 242.9, 905.3]},
    {"label": "person", "class_id": 0, "confidence": 0.8454, "box": [670.6, 394.1, 809.9, 879.8]},
    {"label": "person", "class_id": 0, "confidence": 0.8258, "box": [220.4, 403.7, 345.0, 859.7]},
    {"label": "person", "class_id": 0, "confidence": 0.3818, "box": [-0.0, 550.2, 66.6, 871.3]}
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

## Tests

`pip install -r requirements-test.txt && python -m pytest tests` runs 14 tests in under a
second, with no model file and no AWS account: ONNX Runtime and DynamoDB are replaced by
fakes. They cover letterboxing, IoU and NMS, decoding raw YOLO output back to
original-image pixels, routing and request parsing, per-request thresholds, best-effort
logging, and `/stats`. CI also runs `cfn-lint` on `template.yaml` and builds the Lambda image.

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
tests/                   # unit tests (fake ONNX session, fake DynamoDB table)
DEPLOY.md                # step-by-step deployment + teardown
```

## Three things that broke on the way here

**onnxruntime crashed on cold start, only on arm64.** The Graviton build reads `/sys` to
detect CPU features, and that path does not exist in the Lambda execution environment. It
failed at import, before any of my code ran, so the logs showed almost nothing. Moved the
image to x86_64.

**`Runtime.InvalidEntrypoint` with a Dockerfile that was demonstrably correct.** Docker's
containerd image store produces a multi-manifest image by default, and Lambda cannot pick
an architecture out of one. The fix is a build flag, not a Dockerfile change — which is why
reading the Dockerfile repeatedly got me nowhere.

**CORS preflight on the browser demo.** The demo's JSON `POST` triggers an `OPTIONS`
preflight, which failed until CORS was configured on the API itself. HTTP APIs answer the
preflight from `CorsConfiguration` in `template.yaml` and ignore CORS headers returned by a
Lambda proxy integration, so the fix lives in the template and the handler sets none.

## Notes / next steps

- `/stats` uses a small DynamoDB scan for demo simplicity; for scale, add a GSI on
  `timestamp` and query it.
- Cold starts: the first request after idle pays model-load time; provisioned concurrency
  removes it if needed.
- Pre-processing letterboxes to 640×640 (aspect-ratio preserving, grey padding) to match
  YOLO training; boxes are mapped back to original-image pixels before they're returned.

## Limitations

- **The deployed model is a pretrained COCO YOLO11 checkpoint, not the landmine detector.**
  The trained `best.pt` was not to hand when this was built. The handler is class-agnostic
  and reads its class count from the model output, so swapping in the real weights is one
  command and a redeploy — but as it stands this serves a general object detector.
- Single-region (`ap-southeast-2`), single-environment. No staging stack, no blue/green.
- Logging is inference metadata to DynamoDB. There are no alarms, no dashboards and no SLO
  on this service; it records what happened rather than telling anyone about it.
- Once the landmine weights are swapped in, expect its 15.1% false-alarm rate on mine-free
  scenes (measured in the landmine repo). `conf` can be tuned per request to trade recall
  against precision.
