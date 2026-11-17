# How we work together

Two people, one repo, one branch: we both commit directly to `main`.
That only works if we pull often, push small commits, and stay in our own folders.

## 1. Daily routine

One-time setup, so `git pull` replays your commits on top of the other person's
instead of creating merge commits:

```bash
git config pull.rebase true
```

Every time you start working, and again right before you push:

```bash
git pull            # get the other person's latest commits
pytest -q           # make sure everything still works
git push
```

- Commit and push small pieces often (at least once per work session).
  The longer you wait, the bigger the conflicts.
- If `git pull` reports a conflict: fix the files it lists, then
  `git add <file>` and `git rebase --continue`. If unsure, stop and ask the other person.
- Never force-push (`git push --force`); GitHub blocks it on `main` anyway.

## 2. Stay in your own folders

| Path | Owner |
|---|---|
| `src/alpr/vehicle/`, `src/alpr/ui/`, `src/alpr/pipeline.py`, `notebooks/A_*` | Dali (Student A) |
| `src/alpr/plate/`, `notebooks/B_*` | Yessine (Student B) |
| `src/alpr/contracts.py`, `tests/test_contracts.py`, `src/alpr/evaluation/`, `pyproject.toml`, docs | Shared |

Working in separate folders is what keeps one shared branch conflict-free.
If you need a change in the other person's folder, tell them first.

## 3. The contract is an agreement

`src/alpr/contracts.py` is how the two halves talk. Before changing it:

- agree on the change with the other person first;
- commit it **on its own**, with a message that says what changed and why;
- update `tests/test_contracts.py` in the same commit.

## 4. Never commit

- Datasets, images and videos (except tiny test fixtures in `tests/fixtures/`)
- Model weights (`*.pt`, `*.onnx`, ...) and training output folders (`runs/`)
- Secrets: `kaggle.json`, API keys, tokens

`.gitignore` blocks the common cases, but check `git status` before every commit.
Anything pushed stays in Git history forever, even if deleted later.

## 5. Notebooks

- Name them by owner: `notebooks/A_vehicle_eval.ipynb`, `notebooks/B_plate_train.ipynb`.
- Never edit the other person's notebook; copy it instead.
- Keep logic in `src/alpr/` and call it from the notebook, so it can be tested and reused by the app.
- Clear outputs before committing (Colab: *Edit → Clear all outputs*); big outputs make huge diffs and conflicts.

## 6. Results and numbers

- Record which dataset, split, weights and settings produced each number.
- Never put example or guessed numbers in the report or README as if they were measured.
