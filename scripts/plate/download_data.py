"""Download the public datasets used by the plate side into ``data/raw/<name>/``.

Public Kaggle datasets can be downloaded without an account through the versioned download URL,
so no ``kaggle.json`` is needed. Versions are pinned so everyone gets the same files.

Usage:  python scripts/plate/download_data.py            # all
        python scripts/plate/download_data.py ocr_zindi  # one
"""

from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# key: (kaggle ref, version, folder, use)
DATASETS = {
    "det_prof_709": ("achrafkhazri/labeled-licence-plates-dataset", 1,
                     "achrafkhazri_labeled-licence-plates-dataset", "Model 2: 709 photos, VOC plate boxes"),
    "det_cvat_28": ("mohamedalitrabelsi/tunisiania-license-plate-detection", 1,
                    "mohamedalitrabelsi_tunisiania-license-plate-detection", "Model 2: 28 photos, CVAT plate boxes"),
    "ocr_zindi": ("bechirbrahem/tunisian-licence-plates", 1,
                  "bechirbrahem_tunisian-licence-plates", "Model 3: 900 plate crops + text"),
}


def download(ref: str, version: int, dest: Path, retries: int = 30) -> None:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}?datasetVersionNumber={version}"
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "archive.zip"
    for attempt in range(1, retries + 1):
        have = archive.stat().st_size if archive.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as resp, open(archive, "ab" if have else "wb") as f:
                if have and resp.status != 206:  # server ignored the range: start over
                    f.truncate(0)
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            break
        except Exception as exc:  # network drop or laptop sleep: resume from what we have
            print(f"  attempt {attempt} failed ({exc}); resuming")
            time.sleep(5)
    else:
        raise RuntimeError(f"could not download {ref}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    archive.unlink()


def main() -> None:
    keys = sys.argv[1:] or list(DATASETS)
    for key in keys:
        ref, version, folder, use = DATASETS[key]
        dest = RAW / folder
        if dest.exists() and any(p.name != "archive.zip" for p in dest.iterdir()):
            print(f"{key}: already in {dest}")
            continue
        print(f"{key}: {ref} v{version} -> {dest}  ({use})")
        download(ref, version, dest)


if __name__ == "__main__":
    main()
