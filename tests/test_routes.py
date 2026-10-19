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

    def scan(self, Limit):
        return {"Items": self.items[:Limit]}


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
    # And the inference was logged.
    assert table.put[0]["id"] == body["id"]
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


def test_stats_returns_newest_first(handler, monkeypatch):
    items = [{"id": "a", "timestamp": "2026-06-30T10:00:00", "num_detections": 1,
              "top_confidence": 0.5, "latency_ms": 100},
             {"id": "b", "timestamp": "2026-06-30T11:00:00", "num_detections": 0,
              "top_confidence": 0, "latency_ms": 90}]
    monkeypatch.setattr(handler, "_get_table", lambda: FakeTable(items))
    status, body = call(handler, event("GET", "/stats"))
    assert status == 200
    assert [r["id"] for r in body["recent"]] == ["b", "a"]
