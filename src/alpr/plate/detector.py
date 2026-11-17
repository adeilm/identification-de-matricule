"""Model 2: license plate detector, a YOLO model fine-tuned on vehicle crops.

It runs on one vehicle crop (``VehicleROI.plate_search_image``), the same kind of image it was
trained on, and returns boxes in that crop's coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alpr.contracts import BBox


class PlateDetector:
    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 416,
        max_plates: int = 2,
        device: str | None = None,
    ):
        from ultralytics import YOLO  # imported here so the rest of the package loads without it

        self.model = YOLO(str(weights))
        self.conf, self.iou, self.imgsz, self.max_plates, self.device = conf, iou, imgsz, max_plates, device

    def detect(self, image: np.ndarray) -> list[tuple[BBox, float]]:
        """Plate boxes in ``image`` coordinates, best first. ``image`` is BGR uint8."""
        result = self.model.predict(
            image, conf=self.conf, iou=self.iou, imgsz=self.imgsz, max_det=self.max_plates,
            device=self.device, verbose=False,
        )[0]
        found = []
        for (x1, y1, x2, y2), score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            try:
                found.append((BBox.from_float(x1, y1, x2, y2, image.shape), float(score)))
            except ValueError:  # box collapsed to nothing after clipping
                continue
        return sorted(found, key=lambda item: -item[1])
