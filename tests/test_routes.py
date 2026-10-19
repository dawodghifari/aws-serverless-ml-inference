"""The Lambda entry point: routing, request parsing, logging, /stats."""

import base64
import io
import json

import pytest
from conftest import yolo_output
from PIL import Image


def event(method, path, body=None, b64=False):
    raw = json.dumps(body) if body is not None else None
    if b64 and raw is not None:
        raw = base64.b64encode(raw.encode()).decode()
    return {"requestContext": {"http": {"method": method, "path": path}},
            "body": raw, "isBase64Encoded": b64}


def png_b64(size=(640, 640)):
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 30, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeTable:
    def __init__(self, items=()):
        self.items = list(items)
        self.put = []

    def put_item(self, Item):
        self.put.append(Item)

    def query(self, **kwargs):
        # Mimic the by-time GSI: items in timestamp order, newest first when
        # ScanIndexForward is False.
        self.last_query = kwargs
        ordered = sorted(self.items, key=lambda it: it["timestamp"],
                         reverse=not kwargs.get("ScanIndexForward", True))
        return {"Items": ordered[: kwargs["Limit"]]}


@pytest.fixture
def table(handler, monkeypatch):
    t = FakeTable()
    monkeypatch.setattr(handler, "_get_table", lambda: t)
    return t


def call(handler, ev):
    resp = handler.lambda_handler(ev, None)
    return resp["statusCode"], json.loads(resp["body"])


def test_unknown_route_is_404(handler):
    status, body = call(handler, event("GET", "/nope"))
    assert status == 404 and "No route" in body["error"]


def test_predict_without_an_image_is_400(handler, table):
    status, _ = call(handler, event("POST", "/predict", {"conf": 0.5}))
    assert status == 400


@pytest.mark.parametrize("b64", [False, True])
def test_predict_end_to_end(handler, session, table, b64):
    session.output = yolo_output([(320, 320, 64, 64, 0, 0.8)])
    status, body = call(handler, event("POST", "/predict", {"image_base64": png_b64()}, b64=b64))
    assert status == 200
    assert body["num_detections"] == 1
    assert body["detections"][0]["box"] == pytest.approx([288.0, 288.0, 352.0, 352.0])
    # The annotated image is a real PNG.
    annotated = base64.b64decode(body["annotated_image_base64"])
    assert Image.open(io.BytesIO(annotated)).format == "PNG"
    # The model saw a letterboxed NCHW tensor.
    assert session.last_input.shape == (1, 3, 640, 640)
    # And the inference was logged, into the partition /stats reads.
    assert table.put[0]["id"] == body["id"]
    assert table.put[0]["pk"] == handler.STATS_PARTITION
    assert table.put[0]["num_detections"] == 1


def test_per_request_thresholds_and_no_annotation(handler, session, table):
    session.output = yolo_output([(320, 320, 64, 64, 0, 0.4)])
    _, strict = call(handler, event("POST", "/predict",
                                    {"image_base64": png_b64(), "conf": 0.5, "annotate": False}))
    assert strict["num_detections"] == 0
    assert "annotated_image_base64" not in strict


def test_logging_failure_never_fails_the_request(handler, session, monkeypatch):
    def broken():
        raise RuntimeError("DynamoDB unavailable")

    monkeypatch.setattr(handler, "_get_table", broken)
    session.output = yolo_output([])
    status, _ = call(handler, event("POST", "/predict", {"image_base64": png_b64()}))
    assert status == 200


def test_stats_reads_the_newest_rows_from_the_index(handler, monkeypatch):
    items = [{"id": f"r{i:02d}", "timestamp": f"2026-06-30T{i:02d}:00:00",
              "num_detections": 0, "top_confidence": 0, "latency_ms": 90}
             for i in range(30)]
    fake = FakeTable(items)
    monkeypatch.setattr(handler, "_get_table", lambda: fake)
    status, body = call(handler, event("GET", "/stats"))
    assert status == 200
    assert fake.last_query["IndexName"] == "by-time"
    assert fake.last_query["ScanIndexForward"] is False
    # The newest 25 of 30, newest first: r29 down to r05.
    ids = [r["id"] for r in body["recent"]]
    assert ids[0] == "r29" and ids[-1] == "r05" and len(ids) == 25


@pytest.mark.parametrize("body, raw, b64, message", [
    (None, "{not json", False, "must be JSON"),
    (None, "[1, 2]", False, "JSON object"),
    (None, "%%%", True, "must be JSON"),
    ({"image_base64": "***"}, None, False, "not valid base64"),
    ({"image_base64": base64.b64encode(b"not an image").decode()}, None, False, "readable image"),
    ({"image_base64": "aGk=", "conf": "high"}, None, False, "must be numbers"),
    ({"image_base64": "aGk=", "iou": 1.5}, None, False, "between 0 and 1"),
])
def test_bad_requests_are_400_with_a_reason(handler, table, body, raw, b64, message):
    ev = event("POST", "/predict", body)
    if raw is not None:
        ev["body"], ev["isBase64Encoded"] = raw, b64
    status, resp = call(handler, ev)
    assert status == 400
    assert message in resp["error"]


def test_oversized_image_is_413(handler, table, monkeypatch):
    monkeypatch.setattr(handler, "MAX_IMAGE_BYTES", 100)
    status, resp = call(handler, event("POST", "/predict", {"image_base64": png_b64()}))
    assert status == 413


def test_unexpected_errors_do_not_leak_details(handler, session, table, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(handler, "_postprocess", boom)
    session.output = yolo_output([])
    status, resp = call(handler, event("POST", "/predict", {"image_base64": png_b64()}))
    assert status == 500
    assert resp == {"error": "Internal error."}
