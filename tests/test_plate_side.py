"""Unit tests for the plate side. Tests that need OpenCV, PyTorch or fonts are skipped when those
are not installed (CI installs only numpy + pytest)."""

import re

import importlib.util

import numpy as np
import pytest

from alpr.evaluation.text_metrics import edit_distance, text_metrics

needs_cv2 = pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="OpenCV not installed")


def test_edit_distance_and_metrics():
    assert edit_distance("117T3989", "117T3989") == 0
    assert edit_distance("117T389", "117T3989") == 1
    m = text_metrics(["117T3989", "12T5"], ["117T3989", "12T6"])
    assert m["exact"] == 0.5
    assert m["cer"] == pytest.approx(1 / 12)


def plate_like(width=200, height=50, angle=0.0):
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((height * 3, width * 2, 3), np.uint8)
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    cv2.rectangle(img, (cx - width // 2, cy - height // 2), (cx + width // 2, cy + height // 2), (255, 255, 255), 2)
    cv2.putText(img, "117 3989", (cx - width // 2 + 10, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))


@needs_cv2
@pytest.mark.parametrize("angle", [-8.0, 0.0, 6.0])
def test_estimate_skew_recovers_rotation(angle):
    from alpr.plate.rectification import estimate_skew

    # cv2 rotates counter-clockwise for positive angles; on screen y points down, so the line slope flips sign
    assert estimate_skew(plate_like(angle=angle)) == pytest.approx(-angle, abs=1.5)


@needs_cv2
@pytest.mark.parametrize("angle", [-8.0, 6.0])
def test_deskew_makes_lines_horizontal(angle):
    from alpr.plate.rectification import deskew, estimate_skew

    assert estimate_skew(deskew(plate_like(angle=angle))) == pytest.approx(0.0, abs=1.5)


@needs_cv2
def test_rectify_fixed_height_and_input_untouched():
    from alpr.plate.rectification import rectify

    img = plate_like(angle=5)
    before = img.copy()
    out = rectify(img, height=64)
    assert out.shape[0] == 64 and out.dtype == np.uint8
    assert np.array_equal(img, before)


@needs_cv2
def test_enhance_keeps_shape_and_input():
    pytest.importorskip("cv2")
    from alpr.plate.enhancement import PlateEnhanceConfig, enhance_plate

    img = plate_like()
    before = img.copy()
    for cfg in (PlateEnhanceConfig(), PlateEnhanceConfig(binarize=True)):
        out = enhance_plate(img, cfg)
        assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, before)


def test_synthetic_labels_follow_the_ocr_format():
    pytest.importorskip("cv2")
    from alpr.plate.synth import TunisianPlateSynth

    try:
        synth = TunisianPlateSynth(seed=0)
    except RuntimeError:
        pytest.skip("fonts not installed")
    for _ in range(20):
        img, label = synth.sample()
        assert re.fullmatch(r"\d{1,3}T\d{1,4}|\d{5,6}N", label)
        assert img.ndim == 3 and img.dtype == np.uint8


def test_ctc_greedy_decode_collapses_repeats_and_blanks():
    torch = pytest.importorskip("torch")
    from alpr.plate.ocr import CTCCodec

    codec = CTCCodec("0123456789TN")
    # path: 1 1 blank 1 T T 2  ->  "11T2"
    path = [codec.index["1"], codec.index["1"], 0, codec.index["1"], codec.index["T"], codec.index["T"], codec.index["2"]]
    logits = torch.full((len(path), 1, codec.num_classes), -10.0)
    for t, k in enumerate(path):
        logits[t, 0, k] = 0.0
    (text, conf), = codec.decode_greedy(logits.log_softmax(2))
    assert text == "11T2"
    assert 0.99 < conf <= 1.0


def test_crnn_output_shape():
    torch = pytest.importorskip("torch")
    from alpr.plate.ocr import CRNN, IMG_H, IMG_W

    model = CRNN(num_classes=13).eval()
    out = model(torch.zeros(2, 1, IMG_H, IMG_W))
    assert out.shape == (32, 2, 13)


def test_recognizer_returns_contract_results_with_fakes():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from alpr.contracts import BBox, PlateStatus, VehicleClass, VehicleROI
    from alpr.plate.recognizer import TunisianPlateRecognizer

    class FakeDetector:
        def detect(self, image):
            return [(BBox(10, 20, 90, 40), 0.9)]

    class FakeReader:
        def read_batch(self, images):
            return [("117T3989", 0.95) for _ in images]

    rec = TunisianPlateRecognizer.__new__(TunisianPlateRecognizer)
    rec.detector, rec.reader = FakeDetector(), FakeReader()
    rec.enhance, rec.rectify_plates = True, True
    from alpr.plate.enhancement import PlateEnhanceConfig

    rec.enhance_config = PlateEnhanceConfig()
    crop = np.zeros((60, 100, 3), np.uint8)
    vehicle = VehicleROI(3, BBox(0, 0, 100, 60), VehicleClass.CAR, 0.9, crop=crop)
    (plate,) = rec.recognize(vehicle)
    assert plate.vehicle_id == 3
    assert plate.text == "117 TUN 3989" and plate.status is PlateStatus.VALID
    assert plate.plate_crop.shape == (20, 80, 3)
    assert plate.ocr_input is not None and plate.ocr_input.shape[0] == 64
