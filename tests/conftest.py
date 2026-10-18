"""Import the Lambda handler with ONNX Runtime replaced by a fake.

handler.py loads model/model.onnx at import time, which is what a Lambda cold
start should do. The tests swap in a fake `onnxruntime` first, so they need no
model file, no ONNX Runtime and no AWS account. Each test sets the raw output
tensor the fake session returns.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

INPUT = 640


class FakeSession:
    output = None  # set per test: the raw YOLO tensor to return
    last_input = None

    def __init__(self, *args, **kwargs):
        pass

    def get_inputs(self):
        return [types.SimpleNamespace(name="images")]

    def run(self, names, feeds):
        FakeSession.last_input = feeds["images"]
        return [FakeSession.output]


fake_ort = types.ModuleType("onnxruntime")
fake_ort.SessionOptions = types.SimpleNamespace
fake_ort.InferenceSession = FakeSession
sys.modules["onnxruntime"] = fake_ort
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import handler as _handler  # noqa: E402


@pytest.fixture
def handler(monkeypatch):
    monkeypatch.setattr(_handler, "_labels", ["landmine", "rock"])
    FakeSession.output = None
    return _handler


@pytest.fixture
def session():
    return FakeSession


def yolo_output(boxes, num_classes=2, num_anchors=50):
    """Build a raw YOLOv8/YOLO11 output: shape (1, 4 + nc, anchors).

    `boxes` is a list of (cx, cy, w, h, class_id, score) in letterboxed
    640x640 pixel space. Unused anchors score zero.
    """
    out = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(boxes):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out
