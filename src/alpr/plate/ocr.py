"""Model 3: plate text recognition with a CRNN trained with CTC loss.

Why a CRNN and not per-character segmentation: Tunisian plates are one text line
(series, the word تونس, serial). A CRNN reads the whole line left to right, needs only
(plate image, text) pairs for training, and handles touching or dirty characters.

The output alphabet is ``0-9``, ``T`` (the word تونس) and ``N`` (the mark ن ت), matching the
dataset labels such as ``117T3989``. ``alpr.plate.validator`` turns that into ``117 TUN 3989``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from alpr.plate.validator import OCR_ALPHABET

IMG_H, IMG_W = 32, 128


class CTCCodec:
    """Index 0 is the CTC blank; characters start at 1."""

    def __init__(self, alphabet: str = OCR_ALPHABET):
        self.alphabet = alphabet
        self.index = {ch: i + 1 for i, ch in enumerate(alphabet)}

    @property
    def num_classes(self) -> int:
        return len(self.alphabet) + 1

    def encode(self, text: str) -> list[int]:
        return [self.index[ch] for ch in text]

    def decode_greedy(self, log_probs: torch.Tensor) -> list[tuple[str, float]]:
        """``log_probs``: (T, N, C). Returns (text, confidence) per sample.

        Confidence is the geometric mean of the best-path probability of the kept characters
        (1.0 is fully sure); 0.0 when nothing was read.
        """
        best_lp, best = log_probs.max(dim=2)  # (T, N)
        results = []
        for n in range(best.shape[1]):
            chars, lps, prev = [], [], 0
            for t in range(best.shape[0]):
                k = int(best[t, n])
                if k != 0 and k != prev:
                    chars.append(self.alphabet[k - 1])
                    lps.append(float(best_lp[t, n]))
                prev = k
            conf = float(np.exp(np.mean(lps))) if lps else 0.0
            results.append(("".join(chars), conf))
        return results


def _block(cin: int, cout: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """32x128 grayscale -> 32 time steps -> BiLSTM -> per-step class scores."""

    def __init__(self, num_classes: int, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            *_block(1, 32), nn.MaxPool2d(2, 2),  # 16 x 64
            *_block(32, 64), nn.MaxPool2d(2, 2),  # 8 x 32
            *_block(64, 128), *_block(128, 128), nn.MaxPool2d((2, 1), (2, 1)),  # 4 x 32
            *_block(128, 256), *_block(256, 256), nn.MaxPool2d((2, 1), (2, 1)),  # 2 x 32
            nn.Conv2d(256, 256, (2, 1), bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 1 x 32
            nn.Dropout(0.2),
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, dropout=0.2)
        self.head = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, 32, 128) in [-1, 1]. Returns log-probs (T=32, N, C) for ``nn.CTCLoss``."""
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)  # (T, N, 256)
        out, _ = self.rnn(f)
        return self.head(out).log_softmax(2)


def preprocess(plate_bgr: np.ndarray) -> np.ndarray:
    """BGR plate (any size) -> float32 (1, 32, 128) in [-1, 1]."""
    gray = plate_bgr if plate_bgr.ndim == 2 else cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA if gray.shape[0] > IMG_H else cv2.INTER_CUBIC)
    return ((gray.astype(np.float32) / 255.0 - 0.5) / 0.5)[None]


def save_checkpoint(path: str | Path, model: CRNN, codec: CTCCodec, extra: dict | None = None) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "alphabet": codec.alphabet, "img_h": IMG_H, "img_w": IMG_W,
         "hidden": model.rnn.hidden_size, **(extra or {})},
        path,
    )


class CRNNPlateReader:
    """Loads trained CRNN weights and reads plate images."""

    def __init__(self, weights: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights, map_location=self.device, weights_only=True)
        self.codec = CTCCodec(ckpt["alphabet"])
        self.model = CRNN(self.codec.num_classes, hidden=ckpt.get("hidden", 128)).to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])

    @torch.inference_mode()
    def read_batch(self, plates_bgr: list[np.ndarray]) -> list[tuple[str, float]]:
        if not plates_bgr:
            return []
        batch = torch.from_numpy(np.stack([preprocess(p) for p in plates_bgr])).to(self.device)
        return self.codec.decode_greedy(self.model(batch).cpu())

    def read(self, plate_bgr: np.ndarray) -> tuple[str, float]:
        return self.read_batch([plate_bgr])[0]
