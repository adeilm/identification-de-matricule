"""Contract tests: both students' code must keep these passing.

The fakes below stand in for the real models, so each side can be developed
and tested before the other side exists.
"""

import numpy as np
import pytest

from alpr.contracts import (
    BBox,
    FrameResult,
    PlateResult,
    PlateStatus,
    VehicleClass,
    VehicleROI,
)
from alpr.pipeline import process_frame


def blank(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# --- BBox -------------------------------------------------------------------


def test_bbox_crop_matches_numpy_slicing():
    image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    box = BBox(2, 3, 7, 9)
    assert box.crop(image).shape == (6, 5, 3)
    assert np.array_equal(box.crop(image), image[3:9, 2:7])


@pytest.mark.parametrize("coords", [(5, 0, 5, 4), (0, 5, 4, 2), (-1, 0, 4, 4)])
def test_bbox_rejects_empty_inverted_or_negative(coords):
    with pytest.raises(ValueError):
        BBox(*coords)


def test_bbox_rejects_floats():
    with pytest.raises(TypeError):
        BBox(0.5, 0, 4, 4)


def test_bbox_accepts_numpy_ints():
    box = BBox(np.int64(1), np.int32(2), 3, 4)
    assert type(box.x1) is int


def test_from_float_rounds_outward_and_clips():
    box = BBox.from_float(-3.2, 1.7, 19.1, 8.2, image_shape=(10, 15, 3))
    assert box == BBox(0, 1, 15, 9)


def test_shift_moves_to_parent_coordinates():
    assert BBox(1, 2, 3, 4).shift(10, 20) == BBox(11, 22, 13, 24)


# --- VehicleROI ---------------------------------------------------------------


def test_vehicle_crop_must_match_bbox():
    with pytest.raises(ValueError):
        VehicleROI(0, BBox(0, 0, 10, 10), VehicleClass.CAR, 0.9, crop=blank(5, 10))


def test_enhanced_crop_must_match_crop_shape():
    with pytest.raises(ValueError):
        VehicleROI(
            0, BBox(0, 0, 10, 10), VehicleClass.CAR, 0.9,
            crop=blank(10, 10), enhanced_crop=blank(8, 8),
        )


def test_plate_search_image_prefers_enhanced_crop():
    original, enhanced = blank(10, 10), blank(10, 10) + 1
    plain = VehicleROI(0, BBox(0, 0, 10, 10), VehicleClass.CAR, 0.9, crop=original)
    assert plain.plate_search_image is original
    boosted = VehicleROI(
        0, BBox(0, 0, 10, 10), VehicleClass.CAR, 0.9, crop=original, enhanced_crop=enhanced
    )
    assert boosted.plate_search_image is enhanced


# --- FrameResult --------------------------------------------------------------


def make_plate(vehicle_id: int, bbox: BBox) -> PlateResult:
    return PlateResult(
        vehicle_id=vehicle_id,
        bbox=bbox,
        detection_confidence=0.8,
        plate_crop=blank(bbox.height, bbox.width),
        raw_text="123 TUN 4567",
        text="123 TUN 4567",
        status=PlateStatus.VALID,
    )


def test_frame_rejects_duplicate_vehicle_ids():
    v = VehicleROI(0, BBox(0, 0, 4, 4), VehicleClass.CAR, 0.9, crop=blank(4, 4))
    with pytest.raises(ValueError):
        FrameResult(vehicles=[v, v], plates=[])


def test_frame_rejects_plate_for_unknown_vehicle():
    with pytest.raises(ValueError):
        FrameResult(vehicles=[], plates=[make_plate(7, BBox(0, 0, 4, 2))])


# --- End-to-end with fakes ----------------------------------------------------


class FakeDetector:
    """Pretends to find one car at a fixed place in the frame."""

    def detect(self, frame):
        box = BBox(100, 50, 300, 200)
        return [VehicleROI(0, box, VehicleClass.CAR, 0.95, crop=box.crop(frame))]


class FakeRecognizer:
    """Pretends to find one plate near the bottom of every vehicle crop."""

    def __init__(self, enhance: bool = False):
        self.enhance = enhance

    def recognize(self, vehicle):
        return [make_plate(vehicle.vehicle_id, BBox(60, 120, 140, 145))]


def test_pipeline_maps_plate_box_back_to_frame():
    result = process_frame(blank(480, 640), FakeDetector(), FakeRecognizer())
    assert len(result.vehicles) == 1 and len(result.plates) == 1
    plate = result.plates[0]
    assert plate.bbox == BBox(60, 120, 140, 145)  # relative to the vehicle crop
    assert result.plate_bbox_in_frame(plate) == BBox(160, 170, 240, 195)  # in the frame
    assert set(result.timings_ms) == {"vehicles", "plates", "total"}
