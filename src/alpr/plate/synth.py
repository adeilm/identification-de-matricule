"""Synthetic Tunisian plates for OCR training.

Only ~900 real labelled plate crops exist, and many repeat the same plate. Synthetic plates add
digit combinations, fonts, angles, blur and lighting that the real set lacks. Final numbers are
always measured on REAL held-out plates only.

Layout copied from the real crops: white characters on black,
``<series 1-3 digits>  تونس  <serial 1-4 digits>`` or ``<5-6 digits>  ن ت`` (RS plates).
Labels use the OCR alphabet: ``117T3989``, ``103679N``.

Fonts are taken from the system (DejaVu / Liberation / Noto Arabic on Linux). Install them with
``apt install fonts-dejavu fonts-liberation fonts-noto-core`` if ``find_fonts`` finds none.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"]
DIGIT_FONT_NAMES = [
    "DejaVuSans-Bold.ttf", "DejaVuSansCondensed-Bold.ttf", "LiberationSans-Bold.ttf",
    "LiberationSansNarrow-Bold.ttf", "FreeSansBold.ttf", "NimbusSans-Bold.otf",
    "NimbusSansNarrow-Bold.otf", "NotoSans-Bold.ttf", "Lato-Bold.ttf",
]
ARABIC_FONT_NAMES = [
    "NotoNaskhArabic-Bold.ttf", "NotoNaskhArabic-Regular.ttf", "NotoKufiArabic-Bold.ttf",
    "NotoKufiArabic-Regular.ttf", "NotoSansArabic-Bold.ttf", "NotoSansArabic-Regular.ttf",
]


def find_fonts(names: list[str]) -> list[Path]:
    found = []
    for name in names:
        for d in FONT_DIRS:
            hits = list(d.rglob(name)) if d.exists() else []
            if hits:
                found.append(hits[0])
                break
    return found


@dataclass
class SynthConfig:
    rs_fraction: float = 0.06  # share of RS (ن ت) plates, ~0.6% in real data but we want enough examples
    blue_fraction: float = 0.05  # rental cars: white on blue
    height: int = 100


class TunisianPlateSynth:
    def __init__(self, seed: int = 0, config: SynthConfig = SynthConfig()):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.cfg = config
        self.digit_fonts = find_fonts(DIGIT_FONT_NAMES)
        self.arabic_fonts = find_fonts(ARABIC_FONT_NAMES)
        if not self.digit_fonts or not self.arabic_fonts:
            raise RuntimeError("no suitable fonts found; see the module docstring")

    # --- text -------------------------------------------------------------------------

    def random_label(self) -> str:
        r = self.rng
        if r.random() < self.cfg.rs_fraction:
            return f"{r.randint(10000, 999999)}N"
        series = r.choice([r.randint(1, 99), r.randint(100, 270), r.randint(100, 270), r.randint(100, 270)])
        serial = r.choice([r.randint(1, 999), r.randint(1000, 9999), r.randint(1000, 9999), r.randint(1000, 9999)])
        return f"{series}T{serial}"

    # --- rendering --------------------------------------------------------------------

    def _font(self, paths: list[Path], size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(self.rng.choice(paths)), size)

    def render_clean(self, label: str) -> np.ndarray:
        r, H = self.rng, self.cfg.height
        if label.endswith("N"):
            parts = [(label[:-1], "digit"), ("ن ت", "arabic")]
        else:
            series, serial = label.split("T")
            parts = [(series, "digit"), ("تونس", "arabic"), (serial, "digit")]

        digit_font = self._font(self.digit_fonts, int(H * r.uniform(0.62, 0.8)))
        arabic_font = self._font(self.arabic_fonts, int(H * r.uniform(0.38, 0.55)))
        gap = int(H * r.uniform(0.12, 0.35))
        pad_x = int(H * r.uniform(0.15, 0.45))

        widths = []
        for text, kind in parts:
            font = digit_font if kind == "digit" else arabic_font
            bbox = font.getbbox(text, direction="rtl" if kind == "arabic" else None)
            widths.append(bbox[2] - bbox[0])
        W = sum(widths) + gap * (len(parts) - 1) + 2 * pad_x

        blue = r.random() < self.cfg.blue_fraction
        bg = (r.randint(110, 160), r.randint(40, 70), r.randint(0, 30)) if blue else (r.randint(0, 45),) * 3
        fg = tuple(r.randint(200, 255) for _ in range(3))
        img = Image.new("RGB", (W, H), bg[::-1])  # PIL is RGB; keep BGR meaning for cv2 later
        draw = ImageDraw.Draw(img)
        if r.random() < 0.6:  # embossed border line
            inset = r.randint(2, 6)
            draw.rounded_rectangle([inset, inset, W - inset, H - inset], radius=r.randint(4, 12),
                                   outline=fg, width=r.randint(1, 3))
        if r.random() < 0.3:  # the small dots at both ends seen on many plates
            for x in (pad_x // 2, W - pad_x // 2):
                draw.ellipse([x - 3, H // 2 - 3, x + 3, H // 2 + 3], fill=fg)

        x = pad_x
        for (text, kind), w in zip(parts, widths):
            font = digit_font if kind == "digit" else arabic_font
            kw = {"direction": "rtl"} if kind == "arabic" else {}
            bbox = font.getbbox(text, **kw)
            y = (H - (bbox[3] - bbox[1])) // 2 - bbox[1] + r.randint(-3, 3)
            draw.text((x - bbox[0], y), text, font=font, fill=fg, **kw)
            x += w + gap
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # --- degradation ------------------------------------------------------------------

    def degrade(self, plate: np.ndarray) -> np.ndarray:
        r, n = self.rng, self.np_rng
        h, w = plate.shape[:2]
        # perspective: move each corner a little, then pad with a random "car body" colour
        margin = 0.12
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jitter = lambda s: r.uniform(-margin, margin) * s  # noqa: E731
        dst = np.float32([[jitter(w), jitter(h)], [w + jitter(w), jitter(h)],
                          [w + jitter(w), h + jitter(h)], [jitter(w), h + jitter(h)]])
        dst -= dst.min(axis=0)
        out_w, out_h = int(dst[:, 0].max()) + 1, int(dst[:, 1].max()) + 1
        body = tuple(int(v) for v in n.integers(0, 256, 3))
        warped = cv2.warpPerspective(plate, cv2.getPerspectiveTransform(src, dst), (out_w, out_h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=body)
        # loose or tight crop, like a detector box
        cx1, cy1 = r.randint(0, max(1, out_w // 20)), r.randint(0, max(1, out_h // 8))
        cx2, cy2 = out_w - r.randint(0, max(1, out_w // 20)), out_h - r.randint(0, max(1, out_h // 8))
        img = warped[cy1:cy2, cx1:cx2]
        # lighting
        img = cv2.convertScaleAbs(img, alpha=r.uniform(0.6, 1.3), beta=r.uniform(-40, 50))
        if r.random() < 0.3:  # shadow / gradient across the plate
            grad = np.linspace(r.uniform(0.4, 1.0), r.uniform(0.8, 1.2), img.shape[1])[None, :, None]
            img = np.clip(img * grad, 0, 255).astype(np.uint8)
        # resolution loss: far plates are only ~60-120 px wide
        target_w = r.randint(50, 220)
        f = target_w / img.shape[1]
        if f < 1:
            small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if r.random() < 0.5:
            k = r.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if r.random() < 0.2:  # motion blur
            k = r.choice([3, 5, 7])
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1 / k
            img = cv2.filter2D(img, -1, kernel)
        noise = n.normal(0, r.uniform(0, 12), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, r.randint(25, 95)])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def sample(self) -> tuple[np.ndarray, str]:
        label = self.random_label()
        return self.degrade(self.render_clean(label)), label
