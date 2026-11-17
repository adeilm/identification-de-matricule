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
| *(none yet)* | | | |
