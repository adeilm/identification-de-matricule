"""End-to-end evaluation of the plate side (Model 2 + rectification/enhancement + Model 3 + validator)
on held-out VEHICLE CROPS, i.e. exactly what Dali's side will hand over.

The detection datasets have plate boxes but no plate text, so the text truth comes from a small
sheet that a person checks by hand:

1. ``python scripts/plate/evaluate_plate_side.py --make-sheet``
   writes ``experiments/plate_side/plate_text_test.csv`` (one row per near test crop, ``true_text`` pre-filled
   with the model's reading and ``checked=no``) and contact sheets in ``runs/labeling/``.
2. A person opens the contact sheets, corrects ``true_text`` (format ``117T3989``; empty if the
   plate is unreadable even for a human) and sets ``checked=yes``.
3. ``python scripts/plate/evaluate_plate_side.py`` evaluates, using only rows with ``checked=yes``,
   with plate enhancement off and on, on near crops and their far (downscaled) copies.
   Result: ``experiments/plate_side/<name>.json``.

Pre-filled text is a convenience, not ground truth: unchecked rows are never scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from alpr.contracts import BBox, VehicleClass, VehicleROI
from alpr.evaluation.text_metrics import text_metrics
from alpr.plate.recognizer import TunisianPlateRecognizer
from alpr.plate.validator import to_compact

ROOT = Path(__file__).resolve().parents[2]
# Kept outside data/ (which Git ignores) so the hand-checked labels are versioned and shared.
SHEET = ROOT / "experiments" / "plate_side" / "plate_text_test.csv"


def as_vehicle(image: np.ndarray) -> VehicleROI:
    h, w = image.shape[:2]
    return VehicleROI(0, BBox(0, 0, w, h), VehicleClass.CAR, 1.0, crop=image)


def make_sheet(recognizer: TunisianPlateRecognizer, data_dir: Path) -> None:
    files = sorted((data_dir / "images" / "test").glob("*__near.jpg"))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "runs" / "labeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for i, path in enumerate(files):
        img = cv2.imread(str(path))
        plates = recognizer.recognize(as_vehicle(img))
        read = plates[0].raw_text if plates else ""
        rows.append({"id": i, "file": path.name, "true_text": read, "checked": "no", "model_read": read})
        if plates:
            crop = cv2.resize(plates[0].plate_crop, (240, 60))
        else:  # no detection: show the lower half of the vehicle so a person can still read it
            crop = cv2.resize(img[img.shape[0] // 2:], (240, 60))
        tile = np.full((90, 260, 3), 255, np.uint8)
        tile[5:65, 10:250] = crop
        cv2.putText(tile, f"#{i}  {read or '-'}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1)
        tiles.append(tile)
    with open(SHEET, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for page in range(0, len(tiles), 40):
        chunk = tiles[page:page + 40]
        chunk += [np.full_like(tiles[0], 255)] * (-len(chunk) % 4)
        grid = np.vstack([np.hstack(chunk[r:r + 4]) for r in range(0, len(chunk), 4)])
        cv2.imwrite(str(out_dir / f"sheet_{page // 40:02d}.png"), grid)
    print(f"wrote {SHEET} ({len(rows)} rows) and contact sheets in {out_dir}")


def evaluate(det_w: str, ocr_w: str, data_dir: Path, name: str) -> None:
    with open(SHEET, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["checked"].strip().lower() == "yes"]
    if not rows:
        raise SystemExit(f"no checked rows in {SHEET}; see the module docstring")
    results = {"name": name, "detector": det_w, "ocr": ocr_w, "n_checked": len(rows), "runs": {}}
    for enhance in (False, True):
        rec = TunisianPlateRecognizer(det_w, ocr_w, enhance=enhance)
        for scale, split, suffix in (("near", "test", "__near.jpg"), ("far", "test_small", "__far.jpg")):
            preds, truths, found, times = [], [], 0, []
            for r in rows:
                path = data_dir / "images" / split / r["file"].replace("__near.jpg", suffix)
                if not path.exists():
                    continue
                img = cv2.imread(str(path))
                t0 = time.perf_counter()
                plates = rec.recognize(as_vehicle(img))
                times.append((time.perf_counter() - t0) * 1000)
                found += bool(plates)
                preds.append(to_compact(plates[0].raw_text) if plates else "")
                truths.append(to_compact(r["true_text"]))
            m = text_metrics(preds, truths)
            m.update({"plate_found_rate": found / max(1, len(truths)),
                      "ms_per_vehicle_median": float(np.median(times)) if times else None})
            key = f"{scale}_enhance_{'on' if enhance else 'off'}"
            results["runs"][key] = m
            print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    out = ROOT / "experiments" / "plate_side"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=str(ROOT / "weights" / "plate_yolo11n_v1.pt"))
    ap.add_argument("--ocr", default=str(ROOT / "weights" / "ocr_crnn_mixed_v1.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "plate_det_v1"))
    ap.add_argument("--make-sheet", action="store_true")
    ap.add_argument("--name", default="plate_side_v1")
    args = ap.parse_args()
    if args.make_sheet:
        make_sheet(TunisianPlateRecognizer(args.detector, args.ocr), Path(args.data))
    else:
        evaluate(args.detector, args.ocr, Path(args.data), args.name)


if __name__ == "__main__":
    main()
