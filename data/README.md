# Datasets (not stored in Git)

Everything in this folder except this file is ignored by Git.
Each person downloads the datasets they need, locally or in Colab (Google Drive is the
easiest place to keep them between Colab sessions).

Suggested layout:

```
data/
  raw/<dataset-name>/        exactly as downloaded, never modified
  processed/<dataset-name>/  converted/cropped versions (e.g. YOLO format)
  splits/                    train/val/test file lists (small text files; may be committed later)
```

Candidate datasets from the assignment:

| Use | Dataset | Notes |
|---|---|---|
| Vehicles | UA-DETRAC, KITTI | Stanford Cars has no detection boxes: classification only |
| Plates (Tunisian) | Tunisian Licensed Plates Dataset (Kaggle, 709 images) | Plate boxes only, **no plate text**: OCR evaluation needs transcriptions |
| Plates (other) | OpenALPR benchmark, SSIG SegPlate, AOLP, CCPD | Non-Tunisian formats |
| OCR | SSIG, CCPD, synthetic plates (deep-anpr generator), EMNIST | Synthetic plates would need a Tunisian layout |

When a dataset is chosen, add a row here with: source URL, download date, licence,
how it was split, and any label conversion applied.

## Chosen datasets

| Folder in `data/raw/` | Source | Downloaded | Licence | Split | Conversion | Used by |
|---|---|---|---|---|---|---|
| `achrafkhazri_labeled-licence-plates-dataset` | https://www.kaggle.com/datasets/achrafkhazri/labeled-licence-plates-dataset (v1, 709 photos) | 2026-11-17 | ODbL, contents © authors | 70/15/15 by near-duplicate group, seed 42 | VOC → vehicle crops → YOLO (`scripts/plate/build_plate_det_dataset.py`) | Model 2 |
| `mohamedalitrabelsi_tunisiania-license-plate-detection` | https://www.kaggle.com/datasets/mohamedalitrabelsi/tunisiania-license-plate-detection (v1, 28 labelled photos) | 2026-11-17 | not stated | same as above | CVAT → vehicle crops → YOLO | Model 2 |
| `bechirbrahem_tunisian-licence-plates` | https://www.kaggle.com/datasets/bechirbrahem/tunisian-licence-plates (v1): mirror of the Zindi *AI Hack Tunisia 2* recognition set, 900 plate crops + text | 2026-11-17 | not stated (competition data) | 70/15/15 **by plate text** (596 distinct plates), seed 42 | none; labels `117T3989` | Model 3 |

Download them all with `python scripts/plate/download_data.py` (no Kaggle account needed).
Details and the reasons behind each choice: [docs/plate_side.md](../docs/plate_side.md).
