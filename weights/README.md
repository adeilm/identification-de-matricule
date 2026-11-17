# Model weights (not stored in Git)

Weights are trained in Colab and used by the app on our laptops.
Everything in this folder except this file is ignored by Git.

## Naming

`<stage>_<model>_v<N>.pt`, for example `vehicle_yolo11n_v1.pt`, `plate_yolo11n_v2.pt`.
Never overwrite a version that has been used for reported results; make `v<N+1>`.

## Moving weights from Colab to a laptop

- Colab CLI (from the laptop): `colab download <remote-path>/best.pt weights/plate_yolo11n_v1.pt`
- Colab web UI: save to Google Drive, then download from Drive.

## Sharing weights between us

Publish each version as a GitHub Release asset so both of us download the same file:

```bash
gh release create weights-v1 weights/vehicle_yolo11n_v1.pt weights/plate_yolo11n_v1.pt \
  --title "Weights v1" --notes "What data/settings produced these"
gh release download weights-v1 --dir weights/
```

| File | Trained by | Data / split | Notes |
|---|---|---|---|
| `plate_yolo11n_v1.pt` | Yessine (laptop RTX 4050) | `data/processed/plate_det_v1` (vehicle crops from Kaggle achrafkhazri 709 + mohamedalitrabelsi 28), split by duplicate group, seed 42 | Model 2; results in `experiments/plate_det/plate_yolo11n_v1.json` |
| `ocr_crnn_mixed_v1.pt` | Yessine (laptop RTX 4050) | Zindi recognition crops (Kaggle bechirbrahem) split by plate text, seed 42, + synthetic plates | Model 3, **default**; `experiments/ocr/crnn_mixed_v1.json` |
| `ocr_crnn_real_v1.pt` | Yessine | same real split, no synthetic | Model 3 comparison |
| `ocr_crnn_synth_v1.pt` | Yessine | synthetic plates only | Model 3 comparison |
| `yolo11n.pt`, `yolo11s.pt` | Ultralytics (COCO) | https://github.com/ultralytics/assets/releases/tag/v8.3.0 | starting point for Model 2; YOLO11s makes the vehicle crops for the plate dataset |
