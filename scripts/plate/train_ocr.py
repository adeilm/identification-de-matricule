"""Train Model 3 (CRNN + CTC) on Tunisian plate crops.

Data
* real: Zindi "AI Hack Tunisia 2" recognition set, Kaggle mirror bechirbrahem/tunisian-licence-plates
  (902 crops, labels like ``117T3989``). Split by plate TEXT, so photos of the same plate never
  sit in both train and test. Only real plates are used for validation and test.
* synth: ``alpr.plate.synth`` plates, generated on the fly, training only.

``--mode real | synth | mixed`` trains on real only, synthetic only, or both, so the effect of
synthetic data can be measured on the same real test set.

Outputs
* ``runs/ocr/<name>/history.csv``  per-epoch loss / CER / exact accuracy (training curves)
* ``runs/ocr/<name>/best.pt``      best checkpoint on validation exact accuracy
* ``weights/ocr_<name>.pt``        copy of best.pt (default name crnn_<mode>_v1)
* ``experiments/ocr/<name>.json``  test metrics + settings (small, safe to commit)

Usage:  python scripts/plate/train_ocr.py --mode mixed --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from alpr.evaluation.text_metrics import text_metrics
from alpr.plate.datasets import read_ocr_csv, split_of
from alpr.plate.enhancement import enhance_plate
from alpr.plate.ocr import CRNN, CTCCodec, preprocess, save_checkpoint
from alpr.plate.rectification import rectify
from alpr.plate.synth import TunisianPlateSynth
from alpr.plate.validator import OCR_ALPHABET

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "raw" / "bechirbrahem_tunisian-licence-plates"


def load_real_splits(seed: int) -> dict[str, list[tuple[Path, str]]]:
    items = read_ocr_csv(REAL / "en_train_filtered" / "labels.csv", "zindi")
    items += read_ocr_csv(REAL / "en_val" / "labels.csv", "zindi")
    splits = {"train": [], "val": [], "test": []}
    skipped = 0
    for path, text in items:
        if not text or set(text) - set(OCR_ALPHABET) or not path.exists():
            skipped += 1
            continue
        splits[split_of(text, seed=seed)].append((path, text))
    print({k: len(v) for k, v in splits.items()}, "skipped", skipped)
    return splits


def light_augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    angle = rng.uniform(-4, 4)
    scale = rng.uniform(0.92, 1.05)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.06, 0.06) * h)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-30, 30))
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


class RealPlates(Dataset):
    def __init__(self, items, codec, augment=False, enhance=False, rectify_plates=False, repeat=1, seed=0):
        self.items, self.codec, self.augment = items * repeat, codec, augment
        self.enhance, self.rectify = enhance, rectify_plates
        self.seed, self.rng = seed, None
        self.images = {p: cv2.imread(str(p)) for p, _ in items}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.rng is None:  # one augmentation stream per worker process
            info = torch.utils.data.get_worker_info()
            self.rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        path, text = self.items[i]
        img = self.images[path]
        if self.rectify:
            img = rectify(img)
        if self.enhance:
            img = enhance_plate(img)
        if self.augment:
            img = light_augment(img, self.rng)
        return torch.from_numpy(preprocess(img)), text


class SynthPlates(Dataset):
    def __init__(self, length, seed=0):
        self.length, self.seed = length, seed
        self.synth = None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.synth is None:  # one generator per worker process
            info = torch.utils.data.get_worker_info()
            self.synth = TunisianPlateSynth(seed=self.seed * 1000 + (info.id if info else 0))
        img, text = self.synth.sample()
        return torch.from_numpy(preprocess(img)), text


def collate(codec):
    def fn(batch):
        images = torch.stack([b[0] for b in batch])
        texts = [b[1] for b in batch]
        targets = torch.tensor([k for t in texts for k in codec.encode(t)], dtype=torch.long)
        lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
        return images, targets, lengths, texts
    return fn


@torch.no_grad()
def evaluate(model, loader, codec, device, ctc):
    model.eval()
    preds, gts, losses = [], [], []
    for images, targets, lengths, texts in loader:
        log_probs = model(images.to(device))
        in_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
        losses.append(ctc(log_probs.cpu(), targets, in_len, lengths).item())
        preds += [p for p, _ in codec.decode_greedy(log_probs.cpu())]
        gts += texts
    m = text_metrics(preds, gts)
    m["loss"] = float(np.mean(losses))
    return m, preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synth", "mixed"], default="mixed")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--synth-per-epoch", type=int, default=8000)
    ap.add_argument("--real-repeat", type=int, default=8, help="oversample real crops per epoch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    name = args.name or f"crnn_{args.mode}_v1"
    run_dir = ROOT / "runs" / "ocr" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CTCCodec(OCR_ALPHABET)
    splits = load_real_splits(args.seed)

    train_parts = []
    if args.mode in ("real", "mixed"):
        train_parts.append(RealPlates(splits["train"], codec, augment=True, repeat=args.real_repeat, seed=args.seed))
    if args.mode in ("synth", "mixed"):
        train_parts.append(SynthPlates(args.synth_per_epoch, seed=args.seed))
    train_set = ConcatDataset(train_parts)
    col = collate(codec)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              collate_fn=col, persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(RealPlates(splits["val"], codec), batch_size=256, collate_fn=col)

    model = CRNN(codec.num_classes).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, device {device}")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader))

    history, best = [], (-1.0, 0.0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for images, targets, lengths, _ in train_loader:
            log_probs = model(images.to(device))
            in_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
            loss = ctc(log_probs, targets.to(device), in_len.to(device), lengths.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val, _, _ = evaluate(model, val_loader, codec, device, ctc)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val["loss"],
               "val_cer": val["cer"], "val_exact": val["exact"], "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        score = (val["exact"], -val["cer"])  # best exact accuracy, ties broken by lower CER
        if score > best:
            best = score
            save_checkpoint(run_dir / "best.pt", model, codec, {"mode": args.mode, "epoch": epoch, "seed": args.seed})

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    # final test on REAL held-out plates, with the best checkpoint, enhancement off and on
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    results = {"name": name, "mode": args.mode, "best_epoch": ckpt["epoch"], "seed": args.seed,
               "data": "Zindi AI Hack Tunisia 2 recognition set (Kaggle mirror bechirbrahem/tunisian-licence-plates)",
               "split": "sha1(seed:plate_text) 70/15/15", "split_sizes": {k: len(v) for k, v in splits.items()},
               "args": vars(args), "test": {}}
    for label, kwargs in {"plain": {}, "enhanced": {"enhance": True}, "rectified": {"rectify_plates": True},
                          "rectified+enhanced": {"rectify_plates": True, "enhance": True}}.items():
        loader = DataLoader(RealPlates(splits["test"], codec, **kwargs), batch_size=256, collate_fn=col)
        metrics, preds, gts = evaluate(model, loader, codec, device, ctc)
        metrics["errors"] = [{"pred": p, "true": t} for p, t in zip(preds, gts) if p != t][:30]
        results["test"][label] = metrics
        print(label, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items() if k != "errors"})

    exp_dir = ROOT / "experiments" / "ocr"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / f"{name}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy(run_dir / "best.pt", ROOT / "weights" / f"ocr_{name}.pt")
    print("saved", ROOT / "weights" / f"ocr_{name}.pt")


if __name__ == "__main__":
    main()
