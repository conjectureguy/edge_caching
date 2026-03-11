from __future__ import annotations

import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import URLError

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_100K_DIRNAME = "ml-100k"


def download_movielens_100k(data_root: str | Path) -> Path:
    """Download and extract MovieLens 100K under data_root if needed."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    extracted_dir = data_root / MOVIELENS_100K_DIRNAME
    if extracted_dir.exists():
        return extracted_dir

    zip_path = data_root / "ml-100k.zip"
    if not zip_path.exists():
        try:
            with urllib.request.urlopen(MOVIELENS_100K_URL, timeout=60) as response, zip_path.open("wb") as f:
                shutil.copyfileobj(response, f)
        except URLError as exc:
            raise RuntimeError(
                "Failed to download MovieLens 100K. "
                f"Please verify network access or manually place ml-100k.zip in {data_root}."
            ) from exc

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)

    if not extracted_dir.exists():
        raise FileNotFoundError(f"Expected extracted directory not found: {extracted_dir}")
    return extracted_dir


def load_ratings(dataset_dir: str | Path) -> list[dict[str, int]]:
    """Load u.data ratings file as a list of dictionaries."""
    dataset_dir = Path(dataset_dir)
    ratings_path = dataset_dir / "u.data"
    rows: list[dict[str, int]] = []

    with ratings_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for user_id, item_id, rating, timestamp in reader:
            rows.append(
                {
                    "user_id": int(user_id),
                    "item_id": int(item_id),
                    "rating": int(rating),
                    "timestamp": int(timestamp),
                }
            )
    return rows
