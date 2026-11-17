# Notes for coding agents

Tunisian ALPR course project. Read README.md (ownership, layout) and CONTRIBUTING.md (Git rules) first.

- Two owners: Student A (Dali) = `src/alpr/vehicle/`, `ui/`, `pipeline.py`; Student B (Yessine) = `src/alpr/plate/`. Only change the other owner's folder when asked.
- `src/alpr/contracts.py` is the agreement between both halves. Do not change it as a side effect of other work; a contract change is its own PR and needs the other student's approval.
- Follow its conventions: BGR uint8 images; integer `(x1, y1, x2, y2)` boxes with exclusive x2/y2; plate boxes are relative to the vehicle crop; never overwrite original pixels with enhanced ones.
- Models are trained and evaluated in Colab; laptops are CPU-only. Never commit datasets, weights, videos or secrets.
- Run `pytest -q` before proposing a commit.
- Never report a metric that was not measured, and say which data/weights produced it.
