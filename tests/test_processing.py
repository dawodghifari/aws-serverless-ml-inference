"""Letterbox pre-processing, IoU, NMS and output decoding: plain NumPy."""

import numpy as np
import pytest
from conftest import yolo_output
from PIL import Image


def test_letterbox_keeps_aspect_and_pads_grey(handler):
    img = Image.new("RGB", (1280, 640), (255, 0, 0))
    canvas, scale, pad_x, pad_y = handler._letterbox(img, 640)
    assert canvas.size == (640, 640)
    assert (scale, pad_x, pad_y) == (0.5, 0, 160)
    assert canvas.getpixel((10, 10)) == (114, 114, 114)   # padding
    assert canvas.getpixel((320, 320)) == (255, 0, 0)     # image


def test_preprocess_gives_nchw_float_in_unit_range(handler):
    tensor, *_ = handler._preprocess(Image.new("L", (300, 200), 255))  # greyscale in
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_iou(handler):
    box = np.array([0, 0, 10, 10], dtype=float)
    others = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]], dtype=float)
    assert handler._iou(box, others) == pytest.approx([1.0, 0.0, 1 / 3])


def test_nms_keeps_the_best_of_overlapping_boxes(handler):
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.6, 0.9, 0.5])
    assert [int(i) for i in handler._nms(boxes, scores, 0.45)] == [1, 2]


def test_postprocess_maps_boxes_back_to_original_pixels(handler):
    # A 1280x640 image letterboxes at scale 0.5 with 160 px of padding top and
    # bottom. The original box [100, 100, 300, 200] is therefore centred at
    # (100, 235) in letterbox space, 100 wide and 50 tall.
    out = yolo_output([
        (100, 235, 100, 50, 0, 0.90),   # the detection
        (102, 236, 100, 50, 0, 0.70),   # near-duplicate: NMS removes it
        (400, 300, 40, 40, 1, 0.10),    # below the confidence threshold
    ])
    dets = handler._postprocess(out, 0.5, 0, 160, conf_thr=0.25, iou_thr=0.45)
    assert len(dets) == 1
    assert dets[0]["label"] == "landmine"
    assert dets[0]["confidence"] == pytest.approx(0.9)
    assert dets[0]["box"] == pytest.approx([100.0, 100.0, 300.0, 200.0])


def test_postprocess_with_nothing_above_threshold(handler):
    out = yolo_output([(100, 100, 20, 20, 0, 0.1)])
    assert handler._postprocess(out, 1.0, 0, 0, conf_thr=0.25, iou_thr=0.45) == []


def test_class_count_comes_from_the_model_output(handler, monkeypatch):
    # The same handler serves an 80-class COCO model: labels fall back to the id
    # when labels.json is shorter than the model's class count.
    monkeypatch.setattr(handler, "_labels", ["person"])
    out = yolo_output([(320, 320, 50, 50, 5, 0.8)], num_classes=80)
    dets = handler._postprocess(out, 1.0, 0, 0, conf_thr=0.25, iou_thr=0.45)
    assert (dets[0]["class_id"], dets[0]["label"]) == (5, "5")
