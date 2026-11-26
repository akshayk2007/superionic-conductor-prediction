"""Training-only composition neighbors and archived structure identity checks."""

import hashlib
import math
import re
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from zipfile import ZipFile

import numpy as np
from pymatgen.core.structure_matcher import StructureMatcher

from .data import TARGET, official_path, read_split
from .features import (
    CRYSTAL_SYSTEMS, ELEMENTS, FRACTION_FEATURES, SYSTEM_FEATURES,
    CifInputError, composition_key, inputs_from_cif, parse_composition,
)
from .models import CifPrediction

NEIGHBOR_LIMIT = 3
COMPOSITION_WEIGHT = 0.85
NEIGHBOR_DISCLAIMER = (
    "These conductivity values were measured for the known materials, not for your uploaded material. "
    "Similarity does not validate the prediction."
)
NEIGHBOR_METHOD = (
    "OBELiX training CSV metadata only. Ranking uses 85% elemental-fraction overlap and 15% crystal-system "
    "agreement. References with missing symmetry use composition alone. Cell volume is omitted because the "
    "audit found differences between reported and CIF-derived cell bases."
)
PROVENANCE_LABELS = {
    "close_match": "OBELiX flags a close-match CIF",
    "unflagged_archive": "Archived CIF · no close-match flag",
    "csv_only": "Reported CSV metadata · no archived CIF",
    "archive_unavailable": "Reported CSV metadata · archive unavailable",
}


def doi_link(value: str) -> str | None:
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I)
    if re.fullmatch(r"10\.\d{4,9}/[^\s\x00-\x1f\x7f]+", doi, flags=re.I):
        return "https://doi.org/" + quote(doi, safe="/:;()")
    return None


def crystal_system(raw: str) -> str | None:
    try:
        number = float(raw)
    except ValueError:
        return None
    if not number.is_integer() or not 1 <= number <= 230:
        return None
    return next(system for system, upper in CRYSTAL_SYSTEMS if number <= upper)


@dataclass
class Reference:
    formula: str
    reported_conductivity: str | None
    doi_url: str | None
    crystal_system: str | None
    provenance: str


class NeighborIndex:
    """Use measured properties only when rendering already-ranked references."""

    def __init__(self, data_dir: Path):
        rows = read_split(data_dir, "train")
        self.references = []
        self.archive_available = False
        self.byte_matches = defaultdict(list)
        self.structures_by_composition = defaultdict(list)
        fractions, positions = [], {}
        for _, row in rows.iterrows():
            try:
                comp = parse_composition(row["True Composition"])
            except (ValueError, KeyError):
                continue
            if not comp.num_atoms or not math.isfinite(comp.num_atoms):
                continue
            vector = [comp[element] / comp.num_atoms for element in ELEMENTS]
            if not all(math.isfinite(value) and value >= 0 for value in vector):
                continue
            positions[row["ID"]] = len(self.references)
            fractions.append(vector)
            self.references.append(Reference(
                row["Reduced Composition"] or comp.reduced_formula,
                row[TARGET] or None,
                doi_link(row["DOI"]),
                crystal_system(row["Space group #"]),
                "close_match" if row["close match"].casefold() == "yes" else "archive_unavailable",
            ))
        self.fractions = np.asarray(fractions).reshape(-1, len(ELEMENTS))
        self.systems = np.asarray([reference.crystal_system for reference in self.references], dtype=object)
        try:
            path = official_path(data_dir, "train_cifs.zip")
        except FileNotFoundError:
            return
        self.archive_available = True
        for reference in self.references:
            if reference.provenance == "archive_unavailable":
                reference.provenance = "csv_only"
        with ZipFile(path) as archive, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name in archive.namelist():
                if not name.lower().endswith(".cif"):
                    continue
                position = positions.get(PurePosixPath(name).stem)
                if position is None:
                    continue
                if self.references[position].provenance == "csv_only":
                    self.references[position].provenance = "unflagged_archive"
                content = archive.read(name)
                self.byte_matches[hashlib.sha256(content).digest()].append(position)
                try:
                    record = inputs_from_cif(content)
                except CifInputError:
                    continue
                counts = record.materials.iloc[0]["True Composition"]
                self.structures_by_composition[composition_key(counts)].append((position, record.structure))

    def exact_matches(self, prediction: CifPrediction, content: bytes) -> dict[int, str]:
        matches = {position: "same_file" for position in self.byte_matches.get(hashlib.sha256(content).digest(), ())}
        composition = prediction.cif.materials.iloc[0]["True Composition"]
        structure = prediction.cif.structure
        candidates = self.structures_by_composition.get(composition_key(composition), ())
        if candidates:
            matcher = StructureMatcher(ltol=0.005, stol=0.01, angle_tol=0.5, scale=False)
            for position, reference in candidates:
                if position not in matches and matcher.fit(reference, structure):
                    matches[position] = "equivalent_structure"
        return matches

    def describe(self, position: int) -> dict:
        reference = self.references[position]
        return {**asdict(reference), "provenance_label": PROVENANCE_LABELS[reference.provenance]}

    def find(self, prediction: CifPrediction, content: bytes) -> dict:
        features = prediction.features.iloc[0]
        composition = features.loc[list(FRACTION_FEATURES)].to_numpy()
        system = features.loc[list(SYSTEM_FEATURES)].idxmax().removeprefix("crystal_system_")
        shared = np.minimum(self.fractions, composition)
        overlap = shared.sum(axis=1).clip(0, 1)
        same_system = self.systems == system
        similarity = np.where(
            np.equal(self.systems, None), overlap,
            COMPOSITION_WEIGHT * overlap + (1 - COMPOSITION_WEIGHT) * same_system,
        )
        exact = self.exact_matches(prediction, content)
        neighbors = []
        for position in np.argsort(-similarity, kind="stable"):
            if position in exact:
                continue
            # Skip chemically unrelated rows even if their crystal system agrees.
            if overlap[position] <= 0:
                continue
            element_order = np.argsort(-shared[position], kind="stable")
            elements = [ELEMENTS[i].symbol for i in element_order[:4] if shared[position, i] > 0]
            neighbors.append({
                **self.describe(position),
                "composition_overlap": float(overlap[position]),
                "shared_elements": elements,
                "same_crystal_system": None if self.systems[position] is None else bool(same_system[position]),
                "similarity": float(similarity[position]),
            })
            if len(neighbors) == NEIGHBOR_LIMIT:
                break
        return {
            "neighbors": neighbors,
            "exact_matches": [
                {**self.describe(position), "match_type": kind}
                for position, kind in exact.items()
            ][:NEIGHBOR_LIMIT],
            "exact_match_count": len(exact),
            "archive_available": self.archive_available,
        }
