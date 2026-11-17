"""Train Model 2 (YOLO plate detector) on the vehicle-crop dataset built by
``build_plate_det_dataset.py``, then evaluate it on the near and far (small) test splits.

Outputs
* ``runs/plate_det/<name>/``             Ultralytics run: results.csv (curves), PR/F1 plots, confusion matrix
* ``weights/plate_<model>_v1.pt``        best weights
* ``experiments/plate_det/<name>.json``  test metrics + settings (small, safe to commit)

Usage:  python scripts/plate/train_plate_detector.py --model yolo11n.pt --epochs 100
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def metrics_dict(m) -> dict:
    return {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "speed_ms": {k: round(v, 2) for k, v in m.speed.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "plate_det_v1"))
    ap.add_argument("--model", default=str(ROOT / "weights" / "yolo11n.pt"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    data_dir = Path(args.data)
    stem = Path(args.model).stem
    name = args.name or f"plate_{stem}_v1"
    model = YOLO(args.model)
    model.train(
        data=str(data_dir / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True, patience=25, project=str(ROOT / "runs" / "plate_det"),
        name=name, exist_ok=True, fliplr=0.0,  # mirrored plates do not exist
        plots=True, verbose=False,
    )
    best = ROOT / "runs" / "plate_det" / name / "weights" / "best.pt"
    trained = YOLO(str(best))

    small_yaml = data_dir / "data_test_small.yaml"
    small_yaml.write_text(f"path: {data_dir}\ntrain: images/train\nval: images/val\ntest: images/test_small\n"
                          "names:\n  0: license_plate\n")
    results = {"name": name, "base_model": Path(args.model).name, "imgsz": args.imgsz, "epochs": args.epochs,
               "seed": args.seed, "data": str(data_dir.relative_to(ROOT)), "test": {}}
    for label, yaml in (("near", data_dir / "data.yaml"), ("far_small", small_yaml)):
        m = trained.val(data=str(yaml), split="test", imgsz=args.imgsz, batch=args.batch,
                        project=str(ROOT / "runs" / "plate_det"), name=f"{name}_test_{label}", exist_ok=True,
                        plots=True, verbose=False)
        results["test"][label] = metrics_dict(m)
        print(label, results["test"][label])

    out = ROOT / "experiments" / "plate_det"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(results, indent=2))
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "weights" / f"plate_{stem}_v1.pt")
    print("saved", ROOT / "weights" / f"plate_{stem}_v1.pt")


if __name__ == "__main__":
    main()
