"""Download the public datasets used by the TEST examples.

Sources:

* UCI Adult (income) — https://archive.ics.uci.edu/ml/datasets/adult
* freMTPL2 (French motor TPL insurance) — public academic dataset;
  CSV mirror hosted on Hugging Face (mabilton/fremtpl2), originally from
  the CASdatasets R package / Kaggle mirror.

The full freMTPL2freq file (~36 MB, 678K policies) is kept in data/raw/
(not committed); a reproducible 100K-row sample is written to data/.

Usage:
    python data/download_data.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
DATA = BASE / "data"

URLS = {
    "adult.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    ),
    "freMTPL2freq.csv": (
        "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/"
        "freMTPL2freq.csv"
    ),
    "freMTPL2sev.csv": (
        "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/"
        "freMTPL2sev.csv"
    ),
}


def download(name: str) -> Path:
    """Download (once) and return the raw file path."""
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    url = URLS[name]
    print(f"Downloading {name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    print(f"  saved {dest.stat().st_size / 1e6:.1f} MB", flush=True)
    return dest


def make_samples() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    # UCI Adult: keep the full file (3.8 MB, standard benchmark).
    adult = download("adult.data")
    (DATA / "adult.data").write_bytes(adult.read_bytes())

    # freMTPL2 frequency table: commit a reproducible 100K sample.
    freq_raw = download("freMTPL2freq.csv")
    freq = pd.read_csv(freq_raw)
    sample = (
        freq.sample(n=min(100_000, len(freq)), random_state=42)
        .sort_values("IDpol")
        .reset_index(drop=True)
    )
    sample.to_csv(DATA / "freMTPL2freq_sample.csv", index=False)
    print(f"Wrote {len(sample):,} policy rows to data/freMTPL2freq_sample.csv")

    # freMTPL2 severity table: small, keep the full file.
    sev = download("freMTPL2sev.csv")
    (DATA / "freMTPL2sev.csv").write_bytes(sev.read_bytes())


if __name__ == "__main__":
    make_samples()
