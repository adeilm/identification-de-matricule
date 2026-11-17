# Plate side (Student B, Yessine): Model 2 plate detection + Model 3 OCR

This file records what the plate side does, which data it uses and why, how to reproduce it,
and what the presentation needs from it. Numbers appear here only after they are measured.

## 1. Where it plugs in

Dali's pipeline calls one object that follows `PlateRecognizer` in `src/alpr/contracts.py`:

```python
from alpr.plate.recognizer import TunisianPlateRecognizer

recognizer = TunisianPlateRecognizer(
    "weights/plate_yolo11n_v1.pt",     # Model 2
    "weights/ocr_crnn_mixed_v1.pt",    # Model 3
    enhance=False,                     # plate enhancement switch (UI toggle, ablation)
)
plates = recognizer.recognize(vehicle_roi)  # list[PlateResult]
```

For each vehicle crop (`VehicleROI.plate_search_image`):

```
vehicle crop ─► Model 2 YOLO plate detector ─► plate crop (bbox relative to the vehicle crop)
             ─► rectification (deskew + fixed height) ─► [plate enhancement] ─► Model 3 CRNN
             ─► Tunisian validator ─► PlateResult
```

| `PlateResult` field | Filled with |
|---|---|
| `bbox`, `detection_confidence` | Model 2 box in vehicle-crop pixels, YOLO score |
| `plate_crop` | original pixels of that box (never enhanced) |
| `ocr_input` | exact image given to OCR (after rectification / enhancement), for the UI |
| `raw_text`, `ocr_confidence` | CRNN output in the compact alphabet (e.g. `117T3989`), geometric mean of character probabilities |
| `text`, `status`, `plate_format` | validator output, e.g. `117 TUN 3989`, `valid`, `ordinary` |

| Module | Role |
|---|---|
| `src/alpr/plate/detector.py` | `PlateDetector`: Ultralytics YOLO wrapper, boxes as contract `BBox` |
| `src/alpr/plate/rectification.py` | skew estimate (Hough lines), rotation, resize to a fixed height |
| `src/alpr/plate/enhancement.py` | denoise (bilateral), CLAHE on luminance, unsharp mask, optional Otsu binarization |
| `src/alpr/plate/ocr.py` | CRNN + CTC model, greedy decoding, checkpoint loading |
| `src/alpr/plate/validator.py` | normalization and Tunisian syntax rules |
| `src/alpr/plate/synth.py` | synthetic Tunisian plates for OCR training |
| `src/alpr/plate/datasets.py` | dataset readers, near-duplicate grouping, deterministic splits |
| `src/alpr/evaluation/text_metrics.py` | CER, character accuracy, exact plate accuracy (shared folder, new file) |
| `scripts/plate/*.py` | download, dataset building, training entry points |

## 2. Tunisian plate format (what the validator accepts)

Checked on the real labelled crops (montage in section 3), not only on Wikipedia:

| Format | On the plate (left to right) | OCR label | Canonical `text` |
|---|---|---|---|
| Ordinary | `117  تونس  3989`: series 1–3 digits, the word تونس, serial 1–4 digits | `117T3989` | `117 TUN 3989` |
| RS (temporary / special regime) | `103679  ن ت`: 5–6 digits then ن ت | `103679N` | `103679 NT` |

White characters on black (blue for rental cars). The series is on the **left** in the image.
Rules: map Arabic-Indic digits and the Arabic words to the compact alphabet, fix only
unambiguous digit look-alikes (`O`→`0`, `B`→`8`, ...), never add or drop digits. Anything else is
`invalid_format` with the cleaned text kept. Government, diplomatic (CD/CMD), dealer and military
plates are out of scope for now: none appear in the labelled data.

## 3. Datasets

All downloaded on 2026-11-17 into `data/raw/` (Git-ignored) with `scripts/plate/download_data.py`.
Public Kaggle datasets download without an account via the versioned URL.

| Key | Source | Size | Licence | Used for |
|---|---|---|---|---|
| `det_prof_709` | Kaggle `achrafkhazri/labeled-licence-plates-dataset` v1 (the one in the brief) | 709 photos, 1 plate box each (VOC XML) | ODbL (database), contents © authors | Model 2 |
| `det_cvat_28` | Kaggle `mohamedalitrabelsi/tunisiania-license-plate-detection` v1 | 28 labelled photos (CVAT XML) | not stated | Model 2 |
| `ocr_zindi` | Kaggle `bechirbrahem/tunisian-licence-plates` v1: mirror of the Zindi *AI Hack Tunisia 2* recognition set (InstaDeep / Observatoire National de la Sécurité Routière) | 900 plate crops + text | not stated (competition data) | Model 3 |
| synthetic | `alpr.plate.synth`, generated on the fly | unlimited | ours | Model 3 training only |

Checked facts that shape the method:

- **Detection photos label one plate per photo**, even when other cars are visible. So a crop of
  another car is *not* a "no plate" example: those crops are skipped, never used as negatives.
- The 737 detection photos contain **19 pairs of near-duplicates** (same shot saved at two
  resolutions, e.g. `train/143.jpg` and `train/617.jpg`; confirmed by pixel difference). The group
  split keeps each pair on one side.
- Detection photos are mostly near, medium-distance shots: the plate is 11–31 % of the photo width
  (10th–90th percentile). Far/small vehicles are therefore added by downscaling (section 4).
- **OCR set: 900 crops but only 596 different plates**; 217 plates appear several times (up to 10).
  A random split would put the same plate in train and test, so the split is made **by plate text**.
- OCR crops are small: median 129 × 37 px, 10th percentile 85 px wide. Only 5 RS plates.
- No public dataset found has both full-car photos and plate text for the same images, so the
  end-to-end text accuracy needs a small hand-labelled test set (section 7).

Examined and not used: HF `kernelBits/tunisian_licence_plates` (1,600 crops, no text labels),
HF `Safe-Drive-TN/tunis-word-...` (boxes around the word تونس only), Kaggle `khazri/...` and
`achrafkhazri/licence-plate-digits-dataset` (isolated digits, superseded by the CRNN approach),
PGTLP / PGTLP-v2 (no public download found). Kaggle `hamedmani/tunisian-licence-plates`
(5,000 images, MIT) is being downloaded and inspected; it is added only if its labels fit.

## 4. Model 2 training data: vehicle crops

`scripts/plate/build_plate_det_dataset.py` → `data/processed/plate_det_v1/` (YOLO format) + `manifest.csv`.

1. A COCO-pretrained YOLO11s finds vehicles (car, motorcycle, bus, truck) in each photo.
2. Each labelled plate goes to the smallest vehicle box containing ≥ 90 % of it; the crop gets a
   random 0–8 % margin (Dali's crops may include padding).
3. No vehicle box around the plate (very close shots): a context crop around the plate is used,
   with the plate at 25–45 % of the crop width, and marked `crop_kind=context`.
4. **Near/far mix**: every crop is also saved downscaled to a longer side of 64–224 px (`scale=far`).
   In the test split, those go to `test_small`, so precision/recall are reported separately for
   near and far vehicles.
5. **Split 70/15/15 by groups of near-duplicate photos** (difference hash, Hamming ≤ 4), with
   `sha1(seed:group)` so the split is deterministic. Seed 42.

Result (`plate_det_v1`, built 2026-11-17): 737 photos → 723 vehicle crops + 14 context crops,
each also as a far copy = 1,474 images. Train 1,018 / val 190 / test 133 near + 133 far (`test_small`).
Vehicle classes of the near crops: car 622, truck 88, bus 13, motorcycle 1, unknown (context) 13.
Crop width: near median 702 px (10th–90th pct 439–1000), far median 144 px (79–209).

## 5. Models

**Model 2: YOLO11n plate detector** (`scripts/plate/train_plate_detector.py`). Starts from COCO
weights, one class `license_plate`, input 416 px (vehicle crops are small), horizontal flip off
(mirrored plates do not exist), early stopping after 25 epochs without improvement. Reports P, R,
mAP@0.5, mAP@0.5:0.95 on `test` (near) and `test_small` (far).

**Model 3: CRNN + CTC** (`scripts/plate/train_ocr.py`). Grayscale 32 × 128 input, 7 conv layers →
32 time steps → 2-layer BiLSTM → 13 classes (`0-9`, `T` = تونس, `N` = ن ت, blank); 2.05 M
parameters. Real crops split by plate text: 665 train / 115 validation / 120 test. Chosen over
per-character segmentation because it reads the whole line at once, needs only (image, text) pairs
and tolerates touching or dirty characters. 40 epochs, AdamW + one-cycle LR (max 1e-3), batch 128;
per epoch the real crops are seen 8 times (augmented) and 8,000 new synthetic plates are drawn. Three training modes on the
same real test set, to show what synthetic data brings:

| Mode | Training data |
|---|---|
| `real` | real train crops (augmented) |
| `synth` | synthetic plates only |
| `mixed` | both |

Validation and test use **real plates only**.

## 6. Enhancement ablation (presentation item "avec / sans amélioration")

Model weights and test set stay fixed; only the preprocessing changes.

| Exp. | Vehicle enhancement (Dali) | Plate enhancement (Yessine) | Where measured now |
|---|---|---|---|
| E1 | off | off | OCR test: `plain` / `rectified` |
| E2 | on | off | needs Dali's vehicle side |
| E3 | off | on | OCR test: `enhanced` / `rectified+enhanced` |
| E4 | on | on | needs Dali's vehicle side |

`train_ocr.py` evaluates the OCR test set in four variants (plain, enhanced, rectified,
rectified+enhanced) and stores them in `experiments/ocr/<run>.json`. Results are in section 7.
Neither plate enhancement nor deskewing improved OCR on this data, so both are **off by default**
in `TunisianPlateRecognizer` and stay available as switches for the UI and the ablation.

## 7. Results (measured 2026-11-17)

All on held-out data that was never used for training or model selection. Hardware: laptop
RTX 4050 6 GB.

### Model 2: plate detector (`experiments/plate_det/plate_yolo11n_v1.json`)

| Test split | Images | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference |
|---|---|---|---|---|---|---|
| near vehicle crops | 133 | 1.000 | 0.991 | 0.995 | 0.775 | 7.9 ms |
| far / small copies (64–224 px) | 133 | 0.999 | 0.940 | 0.969 | 0.703 | 2.4 ms |

Early-stopped at epoch 70 of 100 (no improvement for 25 epochs); 524 s of training on the GPU.
The same run on CPU took about 90 s per epoch (41 epochs in 3,690 s) versus 7.5 s per epoch on
the GPU. Small vehicles mostly cost recall (0.991 → 0.940): some far plates are only a few pixels
high. Caveat: test crops come from the same photo collection as training (different photos,
duplicates grouped), so a new camera or country region may score lower.

### Model 3: OCR, exact plate accuracy (CER) on 120 real test plates

| Trained on | no preprocessing | + enhancement | + rectification | + both |
|---|---|---|---|---|
| real only | 85.8 % (3.6 %) | 85.8 % (4.0 %) | 80.8 % (6.2 %) | 83.3 % (6.6 %) |
| synthetic only | 55.0 % (24.4 %) | 40.0 % (31.8 %) | 55.8 % (23.0 %) | 42.5 % (29.3 %) |
| **real + synthetic** | **88.3 % (3.4 %)** | 85.8 % (3.9 %) | 87.5 % (4.7 %) | 85.0 % (4.6 %) |

Reading it: one plate is 0.83 points, so gaps of 1–3 points are within noise. What is clear:
synthetic plates alone do not transfer to real photos (domain gap), but added to real plates they
help a little; enhancement and deskewing do not help a CRNN that was trained on raw crops, and
enhancement clearly hurts the synthetic-only model. Typical errors of the best model: one digit
confused with a similar one (`8`/`0`, `4`/`6`, `2`/`7`), a dropped edge digit on cut or blurred
plates, and garbage on unreadable crops. Deskew only corrects angles up to 20°, so strongly
tilted plates stay tilted.

### Plate side end to end

A visual spot-check of 12 held-out vehicle crops (not a formal measurement) found 10 read exactly right; the full plate side takes
about 8 ms per vehicle on the GPU. A motion-blurred crop was read as `5948N`; requiring 5–6 digits
for RS plates now rejects it. The proper measurement needs the checked sheet: run
`python scripts/plate/evaluate_plate_side.py --make-sheet` (done: 133 rows in
`experiments/plate_side/plate_text_test.csv`, contact sheets in `runs/labeling/`), correct
`true_text`, set `checked=yes`, then run the script without arguments.

### Figures

`python scripts/plate/make_figures.py` → `runs/figures/`: `ocr_training_curves.png`,
`ocr_enhancement_ablation.png`, `plate_detector_near_far.png`, `ocr_examples.png`. Ultralytics also
writes `runs/plate_det/plate_yolo11n_v1/results.png`, PR/F1 curves and confusion matrices.

## 8. Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# CPU PyTorch (~190 MB) is enough for these small models; the default GPU build pulls ~3 GB of CUDA libraries
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# or, for an NVIDIA GPU with a CUDA 13.2 driver (what was used here):
# pip install torch==2.15.0+cu132 torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e ".[plate,dev]"
python scripts/plate/download_data.py
python scripts/plate/build_plate_det_dataset.py
python scripts/plate/train_plate_detector.py --epochs 100
python scripts/plate/train_ocr.py --mode real
python scripts/plate/train_ocr.py --mode synth
python scripts/plate/train_ocr.py --mode mixed
python scripts/plate/make_figures.py
pytest -q
```

The scripts expect the COCO weights `weights/yolo11n.pt` and `weights/yolo11s.pt`
(https://github.com/ultralytics/assets/releases/tag/v8.3.0). Colab works the same way.

## 9. What the presentation needs from the plate side

- [x] Model 2 training curves (`runs/plate_det/plate_yolo11n_v1/results.png`), PR curve, P/R/mAP near vs far
- [x] Model 3 training curves and CER / exact accuracy for real / synth / mixed (`runs/figures/`)
- [x] Enhancement ablation on the plate side (section 7) and example strip crop → rectified → enhanced → text
- [x] Failure examples (`runs/figures/ocr_examples.png`, error list in `experiments/ocr/crnn_mixed_v1.json`)
- [x] Dataset facts: sources, sizes, near/far mix, split by plate text and by duplicate group (sections 3–4)
- [ ] E2/E4: same tables with Dali's vehicle enhancement on
- [ ] End-to-end accuracy on a small hand-labelled test set of full photos (to do with Dali)

## 10. Open items

- Check `experiments/plate_side/plate_text_test.csv` by hand (133 rows), then run the end-to-end evaluation.
- Run the plate side on Dali's real vehicle crops once Model 1 exists (padding/size may differ from ours).
- Cross-dataset test: the Roboflow export `itgateinternship/tunisian-license-plate` v2 (Kaggle
  `hamedmani/tunisian-licence-plates`, 4,994 images, MIT) contains ~1,700 photos from another source;
  ~1,180 of its images are near-copies of `det_prof_709`, so it must be de-duplicated before any use.
- Train OCR on rectified crops if deskewing is to be kept (today the model only saw raw crops).
- Optional baseline: a pretrained OCR (EasyOCR / PaddleOCR) on the same OCR test set, to compare with the trained CRNN.
- Perspective correction from plate corners (labels only give axis-aligned boxes today).

## 11. Work log

- 2026-11-17: compared Tunisian datasets (Kaggle, Hugging Face, Zindi, Roboflow, PGTLP); downloaded
  the ones above; checked plate layout and label format on real crops; wrote validator,
  rectification, enhancement, CRNN OCR, recognizer, synthetic generator, dataset builder, training
  scripts and tests. Built `plate_det_v1`, trained YOLO11n (GPU) and three CRNNs, measured the
  ablation, tightened RS validation after a visual check, made figures and the end-to-end check sheet.
