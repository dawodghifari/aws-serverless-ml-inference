"""
Serverless YOLO object-detection inference handler.

Routes (behind API Gateway HTTP API):
  POST /predict   body: {"image_base64": "<base64 image>",
                         "conf": 0.25, "iou": 0.45, "annotate": true}
                  -> runs the YOLO ONNX model, returns detected boxes + (optional)
                     an annotated image, and logs the inference to DynamoDB.
  GET  /stats     -> returns count + most recent inference logs.

The ONNX model lives at model/model.onnx. Class names are read from
model/labels.json (a JSON array of class-name strings indexed by class id).
Export your trained YOLO weights with scripts/export_model.py.

This serves a single-class landmine detector by default, but the code is
class-agnostic: it reads the class count straight from the model output, so the
same handler serves any YOLOv8/YOLO11 detection model you export.
"""

import base64
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# ---- Cold-start initialisation (runs once per container) ------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.onnx")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "model", "labels.json")
TABLE_NAME = os.environ.get("TABLE_NAME", "ml-inference-logs")

# YOLO is trained at a square input size (640 by default). Override via env if needed.
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "640"))
DEFAULT_CONF = float(os.environ.get("CONF_THRESHOLD", "0.25"))
DEFAULT_IOU = float(os.environ.get("IOU_THRESHOLD", "0.45"))

# On AWS Lambda, /sys/devices/system/cpu is not exposed, so ONNX Runtime's CPU
# auto-detection fails and then crashes trying to log the failure before its logger
# exists. Setting the thread counts explicitly skips that auto-detection path.
_sess_options = ort.SessionOptions()
_sess_options.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "2"))
_sess_options.inter_op_num_threads = 1

_session = ort.InferenceSession(
    MODEL_PATH, sess_options=_sess_options, providers=["CPUExecutionProvider"]
)
_input_name = _session.get_inputs()[0].name

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH) as f:
        _labels = json.load(f)
else:
    _labels = None

# DynamoDB is loaded lazily so the inference code can be tested locally without AWS.
_table = None


def _get_table():
    global _table
    if _table is None:
        import boto3

        _table = boto3.resource("dynamodb").Table(TABLE_NAME)
    return _table


# ---- Pre/post-processing --------------------------------------------------
def _letterbox(img: Image.Image, size: int):
    """Resize keeping aspect ratio, pad to a square. Returns padded image + the
    scale and padding so detected boxes can be mapped back to the original."""
    w0, h0 = img.size
    scale = min(size / w0, size / h0)
    nw, nh = round(w0 * scale), round(h0 * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def _preprocess(img: Image.Image):
    canvas, scale, pad_x, pad_y = _letterbox(img.convert("RGB"), INPUT_SIZE)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0   # HWC, 0-1
    arr = np.transpose(arr, (2, 0, 1))                   # CHW
    return arr[np.newaxis, :].astype(np.float32), scale, pad_x, pad_y


def _iou(box, boxes):
    """IoU of one box [x1,y1,x2,y2] against an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float):
    """Plain NumPy non-max suppression. Returns indices to keep."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thr]
    return keep


def _postprocess(output: np.ndarray, scale, pad_x, pad_y, conf_thr, iou_thr):
    """Decode raw YOLOv8/YOLO11 output -> list of detections in original-image coords.

    Output shape is (1, 4 + num_classes, num_boxes). Row layout per box:
    [cx, cy, w, h, class_score_0, ... class_score_{nc-1}] (no separate objectness)."""
    preds = np.squeeze(output, axis=0).T          # (num_boxes, 4 + nc)
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    mask = confidences >= conf_thr
    boxes_xywh, class_ids, confidences = boxes_xywh[mask], class_ids[mask], confidences[mask]
    if boxes_xywh.shape[0] == 0:
        return []

    # cx,cy,w,h (letterboxed space) -> x1,y1,x2,y2 -> undo padding & scale
    cx, cy, w, h = boxes_xywh.T
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms(boxes, confidences, iou_thr)
    detections = []
    for i in keep:
        cid = int(class_ids[i])
        detections.append(
            {
                "label": _labels[cid] if _labels and cid < len(_labels) else str(cid),
                "class_id": cid,
                "confidence": round(float(confidences[i]), 4),
                "box": [round(float(v), 1) for v in boxes[i]],  # [x1,y1,x2,y2]
            }
        )
    return detections


def _annotate(img: Image.Image, detections) -> str:
    """Draw boxes + labels on a copy of the image, return it as base64 PNG."""
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)
        caption = f'{det["label"]} {det["confidence"]:.2f}'
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, x1 + 8 * len(caption), ty + 12], fill=(255, 60, 60))
        draw.text((x1 + 1, ty), caption, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        # CORS is handled at the API Gateway level (see CorsConfiguration in template.yaml);
        # HTTP APIs ignore CORS headers returned by the integration, so none are set here.
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


# ---- Route handlers -------------------------------------------------------
def _predict(event: dict) -> dict:
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    payload = json.loads(body)

    if "image_base64" not in payload:
        return _response(400, {"error": "Provide 'image_base64' in the request body."})

    conf_thr = float(payload.get("conf", DEFAULT_CONF))
    iou_thr = float(payload.get("iou", DEFAULT_IOU))
    want_annotated = bool(payload.get("annotate", True))

    image_bytes = base64.b64decode(payload["image_base64"])
    img = Image.open(io.BytesIO(image_bytes))

    start = time.perf_counter()
    tensor, scale, pad_x, pad_y = _preprocess(img)
    output = _session.run(None, {_input_name: tensor})[0]
    detections = _postprocess(output, scale, pad_x, pad_y, conf_thr, iou_thr)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    result = {
        "id": str(uuid.uuid4()),
        "latency_ms": latency_ms,
        "num_detections": len(detections),
        "detections": detections,
    }
    if want_annotated:
        result["annotated_image_base64"] = _annotate(img, detections)

    _log_inference(result)
    return _response(200, result)


def _log_inference(result: dict) -> None:
    """Best-effort write to DynamoDB; never fail the request on a logging error."""
    try:
        top_conf = max((d["confidence"] for d in result["detections"]), default=0.0)
        _get_table().put_item(
            Item={
                "id": result["id"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "num_detections": result["num_detections"],
                "top_confidence": Decimal(str(round(top_conf, 4))),
                "latency_ms": Decimal(str(result["latency_ms"])),
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] DynamoDB log failed: {exc}")


def _stats(_event: dict) -> dict:
    # Demo-scale scan; for production use a GSI on timestamp instead of a scan.
    resp = _get_table().scan(Limit=25)
    items = resp.get("Items", [])
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return _response(
        200,
        {
            "count_returned": len(items),
            "recent": [
                {
                    "id": it["id"],
                    "timestamp": it.get("timestamp"),
                    "num_detections": int(it.get("num_detections", 0)),
                    "top_confidence": float(it.get("top_confidence", 0)),
                    "latency_ms": float(it.get("latency_ms", 0)),
                }
                for it in items
            ],
        },
    )


# ---- Entry point ----------------------------------------------------------
def lambda_handler(event, _context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    path = event.get("requestContext", {}).get("http", {}).get("path", "")
    try:
        if path.endswith("/predict") and method == "POST":
            return _predict(event)
        if path.endswith("/stats") and method == "GET":
            return _stats(event)
        return _response(404, {"error": f"No route for {method} {path}"})
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the caller
        return _response(500, {"error": str(exc)})
