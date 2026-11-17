"""Orchestrator: runs the vehicle side, then the plate side, on one frame.

Owner: Student A. It only depends on the interfaces in ``contracts.py``,
so it works with any detector/recognizer, including the test fakes.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from alpr.contracts import FrameResult, PlateRecognizer, VehicleDetector


def process_frame(
    frame: np.ndarray,
    detector: VehicleDetector,
    recognizer: PlateRecognizer,
    frame_index: int | None = None,
) -> FrameResult:
    start = perf_counter()
    vehicles = detector.detect(frame)
    after_vehicles = perf_counter()
    plates = [plate for vehicle in vehicles for plate in recognizer.recognize(vehicle)]
    end = perf_counter()
    return FrameResult(
        vehicles=vehicles,
        plates=plates,
        frame_index=frame_index,
        timings_ms={
            "vehicles": (after_vehicles - start) * 1000,
            "plates": (end - after_vehicles) * 1000,
            "total": (end - start) * 1000,
        },
    )
