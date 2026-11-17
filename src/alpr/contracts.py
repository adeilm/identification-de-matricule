"""Shared data contract between the two halves of the pipeline.

Student A (vehicle side) produces ``VehicleROI`` objects.
Student B (plate side) receives one ``VehicleROI`` and returns ``PlateResult`` objects.
``pipeline.py`` (the orchestrator) combines them into a ``FrameResult``.

Changing anything in this file changes the agreement between both students:
open a pull request and get the other person's approval before merging.

Conventions
-----------
* Images are NumPy arrays of shape (H, W, 3), dtype uint8, OpenCV **BGR** order.
* Boxes are integer pixel coordinates ``(x1, y1, x2, y2)`` with x2/y2 exclusive,
  so a box crops exactly as ``image[y1:y2, x1:x2]``.
* Every box field says which image it is relative to.
* Original pixels are never overwritten: enhanced images are stored next to
  the originals, not instead of them, so on/off comparisons stay possible.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Integer pixel box; crops as ``image[y1:y2, x1:x2]``."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral):
                raise TypeError(f"BBox.{name} must be an int, got {value!r}; use BBox.from_float")
            object.__setattr__(self, name, int(value))
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError(f"negative coordinates: {self}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"empty or inverted box: {self}")

    @classmethod
    def from_float(
        cls, x1: float, y1: float, x2: float, y2: float, image_shape: tuple[int, ...]
    ) -> BBox:
        """Convert a detector's float box: round outward, then clip to the image."""
        height, width = image_shape[:2]
        return cls(
            max(0, math.floor(x1)),
            max(0, math.floor(y1)),
            min(width, math.ceil(x2)),
            min(height, math.ceil(y2)),
        )

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return a copy of the pixels inside this box."""
        return image[self.y1 : self.y2, self.x1 : self.x2].copy()

    def shift(self, dx: int, dy: int) -> BBox:
        """Move the box, e.g. from crop coordinates to the parent image's coordinates."""
        return BBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)


class VehicleClass(StrEnum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"


class PlateStatus(StrEnum):
    VALID = "valid"  # text matches a known Tunisian plate format
    INVALID_FORMAT = "invalid_format"  # text was read but matches no known format
    NO_TEXT = "no_text"  # plate was found but OCR produced nothing usable


def _check_image(name: str, image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be an (H, W, 3) BGR array")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {image.dtype}")


@dataclass(eq=False)
class VehicleROI:
    """One detected vehicle, as handed from Student A's side to Student B's side."""

    vehicle_id: int  # unique within one frame; PlateResult.vehicle_id points here
    bbox: BBox  # region cropped from the frame, in FRAME coordinates (includes any padding)
    vehicle_class: VehicleClass
    confidence: float  # vehicle detector confidence, 0..1
    crop: np.ndarray  # original pixels of `bbox`; never enhanced
    enhanced_crop: np.ndarray | None = None  # same shape as `crop`; only when enhancement is on
    track_id: int | None = None  # stable identity across video frames; None for still images

    def __post_init__(self) -> None:
        _check_image("crop", self.crop)
        if self.crop.shape[:2] != (self.bbox.height, self.bbox.width):
            raise ValueError(
                f"crop shape {self.crop.shape[:2]} does not match bbox size "
                f"{(self.bbox.height, self.bbox.width)}"
            )
        if self.enhanced_crop is not None:
            _check_image("enhanced_crop", self.enhanced_crop)
            if self.enhanced_crop.shape != self.crop.shape:
                raise ValueError("enhanced_crop must have the same shape as crop")

    @property
    def plate_search_image(self) -> np.ndarray:
        """The image the plate detector should run on."""
        return self.enhanced_crop if self.enhanced_crop is not None else self.crop


@dataclass(eq=False)
class PlateResult:
    """One plate found on one vehicle, as returned by Student B's side."""

    vehicle_id: int  # the VehicleROI this plate belongs to
    bbox: BBox  # relative to the VEHICLE CROP, not the frame
    detection_confidence: float  # plate detector confidence, 0..1
    plate_crop: np.ndarray  # pixels of `bbox` cut from VehicleROI.plate_search_image
    raw_text: str  # OCR output before any cleanup ("" if nothing was read)
    text: str  # cleaned/normalized text; never invents characters to force validity
    status: PlateStatus
    ocr_confidence: float | None = None
    ocr_input: np.ndarray | None = None  # exact image given to OCR (after rectification/enhancement)
    plate_format: str | None = None  # which Tunisian format matched, named by the validator


@dataclass(eq=False)
class FrameResult:
    """Everything found in one image or video frame."""

    vehicles: list[VehicleROI]
    plates: list[PlateResult]
    frame_index: int | None = None  # None for still images
    timings_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [v.vehicle_id for v in self.vehicles]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate vehicle_id in frame: {ids}")
        unknown = {p.vehicle_id for p in self.plates} - set(ids)
        if unknown:
            raise ValueError(f"plates refer to unknown vehicle_id(s): {sorted(unknown)}")

    def vehicle(self, vehicle_id: int) -> VehicleROI:
        for v in self.vehicles:
            if v.vehicle_id == vehicle_id:
                return v
        raise KeyError(vehicle_id)

    def plate_bbox_in_frame(self, plate: PlateResult) -> BBox:
        """Convert a plate box from vehicle-crop coordinates to frame coordinates."""
        v = self.vehicle(plate.vehicle_id)
        return plate.bbox.shift(v.bbox.x1, v.bbox.y1)


class VehicleDetector(Protocol):
    """Student A's side. Settings (weights, thresholds, enhancement on/off) go in __init__."""

    def detect(self, frame: np.ndarray) -> list[VehicleROI]: ...


class PlateRecognizer(Protocol):
    """Student B's side. __init__ must accept ``enhance: bool`` so the UI and the
    ablation experiments can switch plate enhancement on and off."""

    def recognize(self, vehicle: VehicleROI) -> list[PlateResult]: ...
