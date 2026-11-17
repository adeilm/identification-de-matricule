"""Build the Model 2 training set: vehicle crops with plate boxes, in YOLO format.

Model 2 runs on vehicle crops at inference (``VehicleROI.plate_search_image``), so it is trained
on vehicle crops too. The plate datasets only give full photos with plate boxes, so we:

1. detect vehicles with a COCO-pretrained YOLO (car, motorcycle, bus, truck);
2. give each labelled plate to the smallest vehicle box that contains it;
3. crop that vehicle with a small random margin and move the plate box into crop coordinates;
4. if no vehicle box contains the plate (very close shots), keep a context crop around the plate.

Vehicles whose plate is not labelled are skipped, never used as "no plate" examples, because the
datasets label only one plate per photo even when other cars are visible.

Scale mix: every crop is also saved as a downscaled "far" copy (longer side 64-224 px) to mimic
small, distant vehicles. In the test split the far copies go to a separate ``test_small`` split,
so near and far performance are reported separately.

Splits are made per group of near-duplicate photos (perceptual hash), so the same car photo
never appears in both train and test.

Usage:  python scripts/plate/build_plate_det_dataset.py [--vehicle-model yolo11s.pt]
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import cv2

from alpr.plate.datasets import (
    PlateSample,
    box_inside_fraction,
    dhash,
    group_near_duplicates,
    read_cvat_xml,
    read_voc_dir,
    split_of,
    to_yolo_line,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_sources() -> list[PlateSample]:
    samples: list[PlateSample] = []
    prof = RAW / "achrafkhazri_labeled-licence-plates-dataset" / "dataset"
    if prof.exists():
        samples += read_voc_dir(prof, "kaggle_achrafkhazri_709")
    cvat = RAW / "mohamedalitrabelsi_tunisiania-license-plate-detection"
    if (cvat / "annotations.xml").exists():
        samples += read_cvat_xml(cvat / "annotations.xml", cvat, "kaggle_mohamedalitrabelsi_28")
    return samples


def context_box(plate, width, height, rng):
    """A pseudo vehicle box around a plate when no vehicle was detected (plate ~25-45% of width)."""
    x1, y1, x2, y2 = plate
    pw, ph = x2 - x1, y2 - y1
    cw = pw / rng.uniform(0.25, 0.45)
    ch = cw * rng.uniform(0.6, 0.9)
    cx = (x1 + x2) / 2 + rng.uniform(-0.1, 0.1) * cw
    top = y1 - rng.uniform(0.45, 0.75) * ch  # plates sit in the lower part of a vehicle
    return (max(0, cx - cw / 2), max(0, top), min(width, cx + cw / 2), min(height, top + ch + ph))


def padded(box, width, height, rng, max_pad=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (
        int(max(0, x1 - rng.uniform(0, max_pad) * w)),
        int(max(0, y1 - rng.uniform(0, max_pad) * h)),
        int(min(width, x2 + rng.uniform(0, max_pad) * w)),
        int(min(height, y2 + rng.uniform(0, max_pad) * h)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle-model", default=str(ROOT / "weights" / "yolo11s.pt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "plate_det_v1"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-side", type=int, default=1280, help="downscale huge photos before cropping")
    args = ap.parse_args()

    from ultralytics import YOLO

    rng = random.Random(args.seed)
    out = Path(args.out)
    samples = load_sources()
    print(f"{len(samples)} labelled photos")

    # 1. load (downscaled) images and hash them for near-duplicate grouping
    images, hashes = [], []
    for s in samples:
        img = cv2.imread(str(s.image_path))
        scale = min(1.0, args.max_side / max(img.shape[:2]))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            s.plates = [tuple(v * scale for v in p) for p in s.plates]
        images.append(img)
        hashes.append(dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    groups = group_near_duplicates(hashes)
    group_key = {}
    for i, g in enumerate(groups):
        group_key.setdefault(g, str(samples[i].image_path.relative_to(RAW)))
    dup_groups = Counter(groups)
    print(f"{sum(1 for c in dup_groups.values() if c > 1)} near-duplicate groups "
          f"covering {sum(c for c in dup_groups.values() if c > 1)} photos")

    # 2. detect vehicles
    detector = YOLO(args.vehicle_model)
    for split in ("train", "val", "test", "test_small"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    rows, counts = [], Counter()
    for idx, (s, img) in enumerate(zip(samples, images)):
        split = split_of(group_key[groups[idx]], seed=args.seed)
        h, w = img.shape[:2]
        res = detector.predict(img, classes=list(COCO_VEHICLES), conf=0.25, verbose=False)[0]
        vehicles = [(tuple(b), COCO_VEHICLES[int(c)]) for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist())]
        for p_i, plate in enumerate(s.plates):
            holders = [(v, c) for v, c in vehicles if box_inside_fraction(plate, v) >= 0.9]
            if holders:
                vbox, vclass = min(holders, key=lambda vc: (vc[0][2] - vc[0][0]) * (vc[0][3] - vc[0][1]))
                kind = "vehicle"
            else:
                vbox, vclass, kind = context_box(plate, w, h, rng), "unknown", "context"
            x1, y1, x2, y2 = padded(vbox, w, h, rng)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # every labelled plate that falls inside this crop is a target, not only plate p_i
            labels = []
            for q in s.plates:
                if box_inside_fraction(q, (x1, y1, x2, y2)) >= 0.6:
                    labels.append((q[0] - x1, q[1] - y1, q[2] - x1, q[3] - y1))
            stem = f"{s.source}__{s.image_path.stem}__{p_i}"
            variants = [("near", crop, labels, split)]
            far_side = rng.randint(64, 224)
            f = far_side / max(crop.shape[:2])
            if f < 0.8:
                small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                far_split = "test_small" if split == "test" else split
                variants.append(("far", small, [tuple(v * f for v in b) for b in labels], far_split))
            for tag, im, labs, sp in variants:
                name = f"{stem}__{tag}"
                cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                ih, iw = im.shape[:2]
                (out / "labels" / sp / f"{name}.txt").write_text("\n".join(to_yolo_line(b, iw, ih) for b in labs) + "\n")
                pw = max(b[2] - b[0] for b in labs) if labs else 0
                rows.append({"file": f"{sp}/{name}.jpg", "split": sp, "source": s.source,
                             "source_image": str(s.image_path.relative_to(RAW)), "crop_kind": kind,
                             "vehicle_class": vclass, "scale": tag, "crop_w": iw, "crop_h": ih,
                             "plate_w": round(pw, 1), "n_plates": len(labs)})
                counts[(sp, kind, tag)] += 1

    with open(out / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: license_plate\n"
    )
    for key in sorted(counts):
        print(key, counts[key])
    print(f"wrote {len(rows)} crops to {out}")


if __name__ == "__main__":
    main()
