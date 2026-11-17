"""Student B's side of the contract: one ``VehicleROI`` in, a list of ``PlateResult`` out.

    detect plate (Model 2) -> crop -> rectify -> [enhance] -> OCR (Model 3) -> validate

Usage from the orchestrator::

    recognizer = TunisianPlateRecognizer("weights/plate_yolo11n_v1.pt", "weights/ocr_crnn_v1.pt", enhance=True)
    plates = recognizer.recognize(vehicle_roi)
"""

from __future__ import annotations

from pathlib import Path

from alpr.contracts import PlateResult, PlateStatus, VehicleROI
from alpr.plate.detector import PlateDetector
from alpr.plate.enhancement import PlateEnhanceConfig, enhance_plate
from alpr.plate.ocr import CRNNPlateReader
from alpr.plate.rectification import rectify
from alpr.plate.validator import validate


class TunisianPlateRecognizer:
    def __init__(
        self,
        detector_weights: str | Path,
        ocr_weights: str | Path,
        enhance: bool = False,
        # Off by default: on the held-out OCR test set, deskewing did not improve exact accuracy
        # (experiments/ocr/crnn_*_v1.json). Kept as a switch for the UI and the ablation.
        rectify_plates: bool = False,
        enhance_config: PlateEnhanceConfig = PlateEnhanceConfig(),
        det_conf: float = 0.25,
        device: str | None = None,
    ):
        self.detector = PlateDetector(detector_weights, conf=det_conf, device=device)
        self.reader = CRNNPlateReader(ocr_weights, device=device)
        self.enhance = enhance
        self.rectify_plates = rectify_plates
        self.enhance_config = enhance_config

    def prepare(self, plate_crop):
        """The exact image given to OCR; exposed so evaluation and the UI can show it."""
        image = rectify(plate_crop) if self.rectify_plates else plate_crop.copy()
        return enhance_plate(image, self.enhance_config) if self.enhance else image

    def recognize(self, vehicle: VehicleROI) -> list[PlateResult]:
        search = vehicle.plate_search_image
        detections = self.detector.detect(search)
        if not detections:
            return []
        crops = [bbox.crop(search) for bbox, _ in detections]
        ocr_inputs = [self.prepare(c) for c in crops]
        readings = self.reader.read_batch(ocr_inputs)
        results = []
        for (bbox, det_conf), crop, ocr_input, (raw, ocr_conf) in zip(detections, crops, ocr_inputs, readings):
            check = validate(raw)
            results.append(
                PlateResult(
                    vehicle_id=vehicle.vehicle_id,
                    bbox=bbox,
                    detection_confidence=det_conf,
                    plate_crop=crop,
                    raw_text=raw,
                    text=check.text,
                    status=check.status,
                    ocr_confidence=ocr_conf if check.status is not PlateStatus.NO_TEXT else None,
                    ocr_input=ocr_input,
                    plate_format=check.plate_format,
                )
            )
        return results
