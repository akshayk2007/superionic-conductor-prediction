"""Fetch and validate the official splits without changing their rows."""

import hashlib
import re
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import pandas as pd

REVISION = "4eaac889809dd489e3bae468c45d5c0e02b9147a"
DOWNLOAD_URL = f"https://raw.githubusercontent.com/NRC-Mila/OBELiX/{REVISION}/data/downloads"
CHECKSUMS = {
    "train.csv": "01d87779862807e24bc45bb2c759852f892390b99f9340f13906c54805aa8776",
    "test.csv": "b2c266eb74e22f403743d52959bea7c43b1531a850994870a70dd909200e5297",
    "train_cifs.zip": "b6f6e6a75dbee9b449c185c9b7d906f3885d6ba53f49c93733a6971f2cbeb751",
    "test_cifs.zip": "c6f4a26a7fc12d9bd97781ffd8195ba45d2429f0c1e9d03bc9be897666a4d86c",
}

TARGET = "Ionic conductivity (S cm-1)"
THRESHOLD = Decimal("1e-4")
READING = re.compile(r"(?P<relation><=|<|≤)?\s*(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def verify_file(path: Path, content: bytes) -> None:
    if hashlib.sha256(content).hexdigest() != CHECKSUMS[path.name]:
        raise ValueError(f"SHA-256 mismatch for {path}. Use an empty data directory to refetch.")


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in CHECKSUMS:
        path = data_dir / name
        if path.exists():
            verify_file(path, path.read_bytes())
            continue
        with urlopen(f"{DOWNLOAD_URL}/{name}", timeout=60) as response:
            content = response.read()
        verify_file(path, content)
        with NamedTemporaryFile(dir=data_dir, delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            temp_path.write_bytes(content)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def official_path(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run the download command first.")
    verify_file(path, path.read_bytes())
    return path


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = official_path(data_dir, f"{split}.csv")
    return pd.read_csv(path, dtype=str, keep_default_na=False).map(str.strip)


def parse_conductivity(raw: str) -> tuple[str, Decimal] | None:
    """Return an exact measurement or its upper bound, expressed in S/cm."""
    if not raw.strip():
        return None
    match = READING.fullmatch(raw.strip())
    if match is None:
        raise ValueError(f"Invalid conductivity value {raw!r}")
    return match["relation"] or "=", Decimal(match["value"])


def binary_label(raw: str) -> int | None:
    reading = parse_conductivity(raw)
    if reading is None:
        return None
    relation, value = reading
    if relation == "=":
        return int(value >= THRESHOLD)
    if value < THRESHOLD or (relation == "<" and value == THRESHOLD):
        return 0
    return None


def split_labels(rows: pd.DataFrame) -> pd.Series:
    labels = rows[TARGET].map(binary_label)
    if labels.isna().any():
        ids = ", ".join(rows.loc[labels.isna(), "ID"])
        raise ValueError(f"Missing or threshold-ambiguous labels for official rows: {ids}")
    return labels.astype(int).rename("fast_conductor")
