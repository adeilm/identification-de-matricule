"""Geometric normalization of a plate crop: deskew, then resize to a fixed height.

The datasets give axis-aligned boxes, not plate corners, so a full perspective warp is not
possible from labels alone. We estimate the dominant text-line angle instead and rotate it
back to horizontal. Small angles only: a wrong large rotation would hurt more than help.
"""

from __future__ import annotations

import cv2
import numpy as np

MAX_SKEW_DEG = 20.0


def estimate_skew(plate_bgr: np.ndarray) -> float:
    """Angle in degrees of the dominant near-horizontal lines (plate borders, character rows).

    Returns 0.0 when no reliable line is found.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 8 or w < 16:
        return 0.0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(10, w // 6), minLineLength=w // 3, maxLineGap=max(2, w // 20)
    )
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):  # (N,1,4) in OpenCV 4, (N,4) in OpenCV 5
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= MAX_SKEW_DEG:
            angles.append(angle)
            weights.append(np.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 0.0
    order = np.argsort(angles)  # length-weighted median, robust to a few stray lines
    cum = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)])


def deskew(plate_bgr: np.ndarray, angle: float | None = None) -> np.ndarray:
    angle = estimate_skew(plate_bgr) if angle is None else angle
    if abs(angle) < 0.5:
        return plate_bgr.copy()
    h, w = plate_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(plate_bgr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def resize_to_height(image: np.ndarray, height: int = 64) -> np.ndarray:
    h, w = image.shape[:2]
    width = max(1, round(w * height / h))
    interp = cv2.INTER_CUBIC if h < height else cv2.INTER_AREA
    return cv2.resize(image, (width, height), interpolation=interp)


def rectify(plate_bgr: np.ndarray, height: int = 64) -> np.ndarray:
    """Deskew then scale to a fixed height. Returns a new BGR array; the input is untouched."""
    return resize_to_height(deskew(plate_bgr), height)
