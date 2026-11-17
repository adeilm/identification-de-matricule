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
