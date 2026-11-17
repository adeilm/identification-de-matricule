"""Optional plate enhancement, switched on/off by ``TunisianPlateRecognizer(enhance=...)``.

Each step targets one problem named in the brief: noise (denoising), contrast (CLAHE on the
luminance), blur (unsharp mask). Binarization is available but off by default, because it often
removes thin strokes on small plates; the ablation measures whether any of this helps OCR.

All functions return a new BGR uint8 array and never modify their input.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PlateEnhanceConfig:
    denoise: bool = True
    clahe: bool = True
    sharpen: bool = True
    binarize: bool = False
    clahe_clip: float = 2.0


def denoise(image: np.ndarray) -> np.ndarray:
    # Bilateral filter: smooths flat areas while keeping character edges.
    return cv2.bilateralFilter(image, d=5, sigmaColor=40, sigmaSpace=5)


def clahe_luminance(image: np.ndarray, clip: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    tiles = (max(1, image.shape[1] // 32), max(1, image.shape[0] // 32))
    l_channel = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a, b)), cv2.COLOR_LAB2BGR)


def unsharp(image: np.ndarray, amount: float = 0.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def enhance_plate(image: np.ndarray, config: PlateEnhanceConfig = PlateEnhanceConfig()) -> np.ndarray:
    out = image.copy()
    if config.denoise:
        out = denoise(out)
    if config.clahe:
        out = clahe_luminance(out, config.clahe_clip)
    if config.sharpen:
        out = unsharp(out)
    if config.binarize:
        out = binarize(out)
    return out
