"""Summarize the two official CSVs and the actual CIF archive membership."""

from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd

from .data import REVISION, TARGET, THRESHOLD, official_path, parse_conductivity, read_split, split_labels
from .features import composition_key


def repeated_groups(keys: pd.Series) -> str:
    counts = keys.value_counts()
    repeated = counts[counts.gt(1)]
    return f"{len(repeated)} groups / {repeated.sum()} rows"


def split_audit(rows: pd.DataFrame, archive_path: Path) -> dict[str, str | int]:
    labels = split_labels(rows)
    readings = rows[TARGET].map(parse_conductivity).tolist()
    bounds = Counter(f"{relation}{value}" for relation, value in readings if relation != "=")
    exact = [value for relation, value in readings if relation == "="]
    reduced = rows["Reduced Composition"]
    true = rows["True Composition"]
    reduced_keys = reduced.map(composition_key)
    true_keys = true.map(composition_key)
    with ZipFile(archive_path) as archive:
        cifs = [PurePosixPath(name).stem for name in archive.namelist() if name.lower().endswith(".cif")]
    cif_ids = set(cifs)
    row_ids = set(rows["ID"])
    marked = rows["Cif ID"].isin(["done", "available"])
    promised_without_file = sorted(set(rows.loc[marked, "ID"]) - cif_ids)
    close_match = rows["close match"].str.casefold().eq("yes")
    return {
        "Rows": len(rows),
        "Columns": len(rows.columns),
        "Unique entry IDs": rows["ID"].nunique(),
        "Slow labels (0)": labels.eq(0).sum(),
        "Fast labels (1)": labels.eq(1).sum(),
        "Exact values at the threshold": sum(value == THRESHOLD for value in exact),
        "Exact conductivity range (S/cm)": f"{min(exact)} to {max(exact)}",
        "Upper bounds (S/cm)": ", ".join(f"`{bound}` × {count}" for bound, count in sorted(bounds.items())),
        "Repeated reduced formula strings": repeated_groups(reduced),
        "Repeated unit-cell formula strings": repeated_groups(true),
        "Repeated normalized reduced compositions": repeated_groups(reduced_keys),
        "Repeated normalized unit-cell compositions": repeated_groups(true_keys),
        "Normalized unit-cell compositions with both labels": labels.groupby(true_keys).nunique().gt(1).sum(),
        "CIF members in archive": len(cifs),
        "CSV rows with an archived CIF": len(cif_ids & row_ids),
        "Archive CIFs without a matching CSV ID": len(cif_ids - row_ids),
        "CIF archive entries marked as close matches": rows["ID"].isin(cif_ids).mul(close_match).sum(),
        "CIF flags": ", ".join(
            f"`{status or 'blank'}` × {count}" for status, count in sorted(rows["Cif ID"].value_counts().items())
        ),
        "Positive CIF flags with no archived file": ", ".join(promised_without_file) or "None",
    }


def audit(data_dir: Path) -> str:
    train, test = (read_split(data_dir, split) for split in ("train", "test"))
    train_summary, test_summary = (
        split_audit(rows, official_path(data_dir, f"{split}_cifs.zip"))
        for split, rows in (("train", train), ("test", test))
    )
    lines = [
        "# OBELiX data audit",
        "",
        f"Official train and test CSVs and CIF archives at revision [`{REVISION[:12]}`]"
        f"(https://github.com/NRC-Mila/OBELiX/tree/{REVISION}/data/downloads).",
        "The audit counts blank or whitespace-only fields as missing.",
        "",
        "## Rows, labels, and CIF coverage",
        "",
        "| Measure | Train | Test |",
        "| --- | ---: | ---: |",
        *(f"| {name} | {value} | {test_summary[name]} |" for name, value in train_summary.items()),
        "",
        "The target is the primary experimental room-temperature ionic conductivity in S/cm.",
        "The auxiliary total and bulk conductivity columns also use S/cm.",
        "Cell lengths use angstroms, angles use degrees, and `Z` counts formula units per cell.",
        "The reported space group number is dimensionless.",
        "",
        "## Cross-split overlap",
        "",
        "| Grouping | Shared values |",
        "| --- | ---: |",
    ]
    for column in ("ID", "DOI", "Reduced Composition", "True Composition"):
        shared = set(train[column]) & set(test[column])
        lines.append(f"| Exact `{column}` | {len(shared)} |")
    for column in ("Reduced Composition", "True Composition"):
        shared = set(train[column].map(composition_key)) & set(test[column].map(composition_key))
        lines.append(f"| Normalized `{column}` | {len(shared)} |")
    lines += [
        "",
        "Normalization parses the formulas and compares elemental fractions rounded to eight decimal places.",
        "Repeated-group counts include every row in each repeated group.",
        "",
        "## Missing values by column",
        "",
        "| Column | Train blanks | Test blanks |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| `{column}` | {train[column].eq('').sum()} | {test[column].eq('').sum()} |"
                 for column in train.columns)
    return "\n".join(lines) + "\n"
