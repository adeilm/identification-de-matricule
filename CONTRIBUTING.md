# How we work together

Two people, one repo. These rules exist so we never overwrite each other's work
or end up with two versions of the same thing.

## 1. Never work directly on `main`

`main` always holds working code that both of us have seen.
All work happens on a branch and arrives in `main` through a pull request (PR).

```bash
git switch main
git pull                              # get the latest main
git switch -c dali/vehicle-detector   # new branch: <your-name>/<topic>
# ... work, commit often ...
git push -u origin dali/vehicle-detector
# then open a PR on GitHub (or: gh pr create)
```

Branch names: `dali/<topic>` or `yessine/<topic>`, e.g. `yessine/plate-ocr-baseline`.

Before opening a PR, bring your branch up to date with `main`:

```bash
git fetch origin
git rebase origin/main   # or: git merge origin/main
pytest -q
```

## 2. Stay in your own folders

| Path | Owner |
|---|---|
| `src/alpr/vehicle/`, `src/alpr/ui/`, `src/alpr/pipeline.py`, `notebooks/A_*` | Dali (Student A) |
| `src/alpr/plate/`, `notebooks/B_*` | Yessine (Student B) |
| `src/alpr/contracts.py`, `tests/test_contracts.py`, `src/alpr/evaluation/`, `pyproject.toml`, docs | Shared |

If you need a change in the other person's folder, ask them or open a PR and let them review it.

## 3. The contract is an agreement

`src/alpr/contracts.py` is how the two halves talk. Any change to it:

- goes in its **own small PR**, with a sentence explaining why;
- is merged **only after the other person approves**;
- updates `tests/test_contracts.py` in the same PR.

## 4. Pull requests

- Small and focused: one feature or fix per PR.
- Tests must pass (GitHub runs them automatically on every PR).
- The other person looks at it before merging when it touches shared files.
  For changes only inside your own folder, a quick self-merge after tests pass is fine.

## 5. Never commit

- Datasets, images and videos (except tiny test fixtures in `tests/fixtures/`)
- Model weights (`*.pt`, `*.onnx`, ...) and training output folders (`runs/`)
- Secrets: `kaggle.json`, API keys, tokens

`.gitignore` blocks the common cases, but check `git status` before every commit.
Anything committed stays in Git history forever, even if deleted later.

## 6. Notebooks

- Name them by owner: `notebooks/A_vehicle_eval.ipynb`, `notebooks/B_plate_train.ipynb`.
- Never edit the other person's notebook; copy it instead.
- Keep logic in `src/alpr/` and call it from the notebook, so it can be tested and reused by the app.
- Clear outputs before committing (Colab: *Edit → Clear all outputs*); big outputs make huge diffs and merge conflicts.

## 7. Results and numbers

- Record which dataset, split, weights and settings produced each number.
- Never put example or guessed numbers in the report or README as if they were measured.
