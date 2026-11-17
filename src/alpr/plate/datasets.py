"""Readers for the plate datasets and helpers to build training sets from them.

Every reader returns ``PlateSample`` objects with boxes in the ORIGINAL image's pixel
coordinates. Raw files under ``data/raw/`` are only read, never modified.
"""

from __future__ import annotations

import csv
import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PlateSample:
    image_path: Path
    plates: list[tuple[float, float, float, float]]  # (x1, y1, x2, y2) in image pixels
    source: str  # dataset name
    group: str = ""  # samples sharing a group always go to the same split
    meta: dict = field(default_factory=dict)


# --- readers -------------------------------------------------------------------------


def read_voc_dir(root: Path, source: str) -> list[PlateSample]:
    """Pascal VOC XML files next to their images (LabelImg output)."""
    samples = []
    for xml_path in sorted(root.rglob("*.xml")):
        tree = ET.parse(xml_path).getroot()
        image_path = xml_path.with_name(tree.findtext("filename") or xml_path.stem + ".jpg")
        if not image_path.exists():
            image_path = xml_path.with_suffix(".jpg")
        boxes = []
        for obj in tree.iter("object"):
            b = obj.find("bndbox")
            boxes.append(tuple(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")))
        if boxes and image_path.exists():
            samples.append(PlateSample(image_path, boxes, source, meta={"orig_split": xml_path.parent.name}))
    return samples


def read_cvat_xml(xml_path: Path, image_dir: Path, source: str) -> list[PlateSample]:
    """CVAT for images 1.1 export."""
    samples = []
    for image in ET.parse(xml_path).getroot().iter("image"):
        boxes = [
            tuple(float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            for box in image.iter("box")
            if box.get("label") == "license_plate"
        ]
        path = image_dir / image.get("name")
        if boxes and path.exists():
            samples.append(PlateSample(path, boxes, source))
    return samples


def read_ocr_csv(csv_path: Path, source: str) -> list[tuple[Path, str]]:
    """``filename,words`` CSV next to plate crops (Zindi recognition set mirror)."""
    with open(csv_path, newline="") as f:
        return [(csv_path.parent / row["filename"], row["words"].strip()) for row in csv.DictReader(f)]


# --- duplicates and splits -----------------------------------------------------------


def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash of a grayscale image; near-identical photos get near-identical hashes."""
    import cv2

    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def group_near_duplicates(hashes: list[int], max_distance: int = 4) -> list[int]:
    """Union-find over hashes: returns a group id per item (same id = near duplicate)."""
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= max_distance:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(hashes))]


def split_of(group_key: str, seed: int = 42, val: float = 0.15, test: float = 0.15) -> str:
    """Deterministic split from a hash of the group key, so adding data never reshuffles old items."""
    h = int(hashlib.sha1(f"{seed}:{group_key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


# --- geometry ------------------------------------------------------------------------


def box_inside_fraction(inner: Iterable[float], outer: Iterable[float]) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    w = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    h = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    area = max(1e-9, (ix2 - ix1) * (iy2 - iy1))
    return w * h / area


def to_yolo_line(box: Iterable[float], width: int, height: int, cls: int = 0) -> str:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
