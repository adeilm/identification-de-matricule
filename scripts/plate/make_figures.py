"""Presentation figures for the plate side, drawn from measured results only.

Reads  runs/ocr/*/history.csv, experiments/ocr/*.json, experiments/plate_det/*.json
Writes runs/figures/*.png  (images are not committed; regenerate with this script)

Usage:  python scripts/plate/make_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "figures"

# Validated categorical order (blue, orange, aqua, yellow), light surface; text in ink tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODES = ["real", "synth", "mixed"]
VARIANTS = ["plain", "enhanced", "rectified", "rectified+enhanced"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titleweight": "bold",
    "axes.titlesize": 12, "legend.frameon": False, "lines.linewidth": 2,
})


def read_history(run: Path) -> dict[str, np.ndarray]:
    with open(run / "history.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def ocr_curves() -> None:
    runs = {m: ROOT / "runs" / "ocr" / f"crnn_{m}_v1" for m in MODES}
    runs = {m: r for m, r in runs.items() if (r / "history.csv").exists()}
    if not runs:
        return
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    main = "mixed" if "mixed" in runs else next(iter(runs))
    h = read_history(runs[main])
    ax_loss.plot(h["epoch"], h["train_loss"], color=SERIES[0], label="train")
    ax_loss.plot(h["epoch"], h["val_loss"], color=SERIES[1], label="validation (real plates)")
    ax_loss.set(title=f"CTC loss, {main} training", xlabel="epoch", ylabel="loss")
    ax_loss.legend()
    for i, mode in enumerate(MODES):
        if mode in runs:
            h = read_history(runs[mode])
            ax_acc.plot(h["epoch"], 100 * h["val_exact"], color=SERIES[i], label=mode)
            ax_acc.annotate(mode, (h["epoch"][-1], 100 * h["val_exact"][-1]), xytext=(4, 0),
                            textcoords="offset points", color=INK2, va="center")
    ax_acc.set(title="Exact plate accuracy on real validation plates", xlabel="epoch", ylabel="% plates fully correct",
               ylim=(0, 100))
    ax_acc.legend(title="training data", loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "ocr_training_curves.png", dpi=160)
    plt.close(fig)


def ocr_ablation() -> None:
    results = {}
    for mode in MODES:
        p = ROOT / "experiments" / "ocr" / f"crnn_{mode}_v1.json"
        if p.exists():
            results[mode] = json.loads(p.read_text())["test"]
    if not results:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(VARIANTS))
    width = 0.8 / len(results)
    for i, (mode, test) in enumerate(results.items()):
        vals = [100 * test[v]["exact"] for v in VARIANTS]
        bars = ax.bar(x + (i - (len(results) - 1) / 2) * width, vals, width * 0.92, color=SERIES[MODES.index(mode)],
                      label=f"{mode} (n={test['plain']['n']})")
        best = int(np.argmax(vals))
        ax.annotate(f"{vals[best]:.1f}", (bars[best].get_x() + bars[best].get_width() / 2, vals[best]),
                    xytext=(0, 3), textcoords="offset points", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(x, ["no preprocessing", "enhancement", "rectification", "rectification\n+ enhancement"])
    ax.set(ylabel="% plates fully correct", ylim=(0, 100),
           title="OCR on held-out real plates: with vs without plate enhancement")
    ax.legend(title="OCR trained on", ncols=len(results), loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "ocr_enhancement_ablation.png", dpi=160)
    plt.close(fig)


def detector_near_far() -> None:
    files = sorted((ROOT / "experiments" / "plate_det").glob("*.json"))
    if not files:
        return
    res = json.loads(files[-1].read_text())
    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(metrics))
    for i, (split, label) in enumerate((("near", "near vehicles"), ("far_small", "far / small vehicles"))):
        vals = [100 * res["test"][split][m] for m in metrics]
        ax.bar(x + (i - 0.5) * 0.4, vals, 0.37, color=SERIES[i], label=label)
    ax.set_xticks(x, ["precision", "recall", "mAP@0.5", "mAP@0.5:0.95"])
    ax.set(ylabel="%", ylim=(0, 100), title=f"Plate detector ({res['name']}) on held-out vehicle crops")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "plate_detector_near_far.png", dpi=160)
    plt.close(fig)


def ocr_examples(n: int = 8) -> None:
    """Strip of test plates: original | rectified | enhanced, with prediction vs truth."""
    p = ROOT / "experiments" / "ocr" / "crnn_mixed_v1.json"
    weights = ROOT / "weights" / "ocr_crnn_mixed_v1.pt"
    if not p.exists() or not weights.exists():
        return
    import sys

    sys.path.insert(0, str(ROOT / "scripts" / "plate"))
    from train_ocr import load_real_splits  # noqa: E402

    from alpr.plate.enhancement import enhance_plate
    from alpr.plate.ocr import CRNNPlateReader
    from alpr.plate.rectification import rectify

    reader = CRNNPlateReader(weights)
    test = load_real_splits(seed=json.loads(p.read_text())["seed"])["test"]
    rng = np.random.default_rng(0)
    picks = [test[i] for i in rng.choice(len(test), size=min(n, len(test)), replace=False)]
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 1.1 * len(picks)))
    for row, (path, truth) in zip(axes, picks):
        img = cv2.imread(str(path))
        stages = [img, rectify(img), enhance_plate(rectify(img))]
        pred, conf = reader.read(stages[-1])
        for ax, im, title in zip(row, stages, ["plate crop", "rectified", "enhanced"]):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if row is axes[0]:
                ax.set_title(title)
        mark = "correct" if pred == truth else "WRONG"
        row[2].text(1.04, 0.5, f"read {pred} ({conf:.2f})\ntrue {truth}  {mark}", transform=row[2].transAxes,
                    va="center", color=INK if pred == truth else INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ocr_examples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (ocr_curves, ocr_ablation, detector_near_far, ocr_examples):
        fn()
    print("figures in", OUT, sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
