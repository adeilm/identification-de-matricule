# Tunisian License Plate Recognition (ALPR)

Course project: *Système de reconnaissance automatique des plaques d'immatriculation*,
Computer Vision and Applications, Faculté des Sciences de Sfax, I3, 2026–2027.

Given an image or a video, the system finds each vehicle, finds the plate on each
vehicle, reads the plate text, and checks it against Tunisian plate formats.

```
frame ─► [Model 1] vehicle detector ─► vehicle crops ─► (optional enhancement)
      ─► [Model 2] plate detector ─► plate crop ─► rectify / (optional enhancement)
      ─► [Model 3] OCR ─► Tunisian format validation ─► result + UI
```

The three models are separate stages, each measured on its own.

## Team and ownership

| Area | Owner | Code |
|---|---|---|
| Input (image, video, webcam), vehicle detection, cropping, vehicle enhancement, tracking, Streamlit UI, orchestration | **Student A: Dali** | `src/alpr/vehicle/`, `src/alpr/ui/`, `src/alpr/pipeline.py` |
| Plate detection, rectification, plate enhancement, OCR, Tunisian validation | **Student B: Yessine** | `src/alpr/plate/` |
| Interface between the two halves | **Both** (agree before changing) | `src/alpr/contracts.py`, `tests/test_contracts.py` |
| Metrics, datasets/splits, final report | **Both** | `src/alpr/evaluation/` |

**Start by reading [`src/alpr/contracts.py`](src/alpr/contracts.py).** It defines exactly
what Student A hands to Student B (`VehicleROI`) and what comes back (`PlateResult`),
plus the image and coordinate conventions. Then read [CONTRIBUTING.md](CONTRIBUTING.md)
for how we use Git.

## Where things run

- **Google Colab (GPU):** building, training and testing the models. Notebooks live in `notebooks/`.
- **Laptops (CPU):** writing code, unit tests, and the Streamlit demo using the exported weights.

## Setup on a laptop

Python 3.12 is recommended (ML libraries often lag the newest Python release).

```bash
git clone https://github.com/adeilm/identification-de-matricule.git
cd identification-de-matricule
python3.12 -m venv .venv && source .venv/bin/activate   # or: uv venv --python 3.12
pip install -e ".[dev]"            # contract + tests only (fast)
pip install -e ".[vehicle,ui,dev]" # Student A's full set
pip install -e ".[plate,dev]"      # Student B's full set
pytest -q
```

## Setup in Colab

First cell of every notebook:

```python
!git clone -q https://github.com/adeilm/identification-de-matricule.git
%cd identification-de-matricule
!pip install -q -e ".[vehicle]"   # or ".[plate]"
```

Colab machines are wiped when the session ends: save checkpoints to Google Drive
while training, and move final weights as described in [weights/README.md](weights/README.md).

## Repository layout

```
src/alpr/
  contracts.py      shared types and conventions (both)
  pipeline.py       runs vehicle side then plate side on one frame (A)
  vehicle/          Model 1, crops, vehicle enhancement, tracking (A)
  plate/            Model 2, rectification, plate enhancement, Model 3, validation (B)
  evaluation/       metrics (both)
  ui/               Streamlit app (A)
tests/              contract + unit tests (run before every push)
notebooks/          Colab notebooks for training/evaluation
data/               datasets, NOT in Git (see data/README.md)
weights/            trained models, NOT in Git (see weights/README.md)
docs/               assignment and design notes
```

More folders (`scripts/`, `configs/`, `experiments/`) are added when first needed.

## Results

No results yet. Metrics will be added here only once measured on the held-out test set.

- **Plate side (Model 2 + Model 3):** measured on held-out data, see
  [docs/plate_side.md § Results](docs/plate_side.md#7-results-measured-2026-11-17).
