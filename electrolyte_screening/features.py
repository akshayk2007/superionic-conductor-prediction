"""Features shared by the CSV benchmark and single-CIF prediction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetryUndeterminedError
from sklearn.base import BaseEstimator, TransformerMixin

LATTICE_COLUMNS = ("a", "b", "c", "alpha", "beta", "gamma")
INPUT_COLUMNS = ("True Composition", *LATTICE_COLUMNS, "Space group #")
ELEMENTS = tuple(Element.from_Z(number) for number in range(1, 119))
ATOMIC_NUMBERS = np.asarray([element.Z for element in ELEMENTS])
ATOMIC_MASSES = np.asarray([float(element.atomic_mass) for element in ELEMENTS])
CRYSTAL_SYSTEMS = (
    ("triclinic", 2),
    ("monoclinic", 15),
    ("orthorhombic", 74),
    ("tetragonal", 142),
    ("trigonal", 167),
    ("hexagonal", 194),
    ("cubic", 230),
)
FRACTION_FEATURES = tuple(f"fraction_{element.symbol}" for element in ELEMENTS)
SYSTEM_FEATURES = tuple(f"crystal_system_{system}" for system, _ in CRYSTAL_SYSTEMS)
VOLUME_FEATURE = "volume_per_atom_angstrom3"
# Allow a small rounding excess while keeping pymatgen's physical occupancy check.
MAX_SITE_OCCUPANCY = 1.002
FEATURE_NAMES = (
    *FRACTION_FEATURES,
    "element_count",
    "mean_atomic_number",
    "std_atomic_number",
    "mean_atomic_mass_amu",
    "std_atomic_mass_amu",
    VOLUME_FEATURE,
    *SYSTEM_FEATURES,
)


class CifInputError(ValueError):
    """The uploaded CIF cannot be used as a screening input."""


@dataclass
class CifInputs:
    materials: pd.DataFrame
    has_partial_occupancies: bool
    parser_warnings: list[str]
    structure: Structure


def parse_composition(formula: str | dict) -> Composition:
    if isinstance(formula, str):
        # The official reduced formulas also use bracketed groups and an adduct separator.
        parts = formula.replace("[", "(").replace("]", ")").split("*")
        composition = sum((Composition(part, strict=True) for part in parts), Composition({}))
    else:
        composition = Composition(formula, strict=True)
    return composition.element_composition


def composition_key(formula: str) -> tuple[tuple[str, float], ...]:
    fractions = parse_composition(formula).fractional_composition
    return tuple(sorted((element.symbol, round(amount, 8)) for element, amount in fractions.items()))


def select_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[:, list(INPUT_COLUMNS)]


def cif_parse_details(error: Exception) -> str:
    cause = error.__cause__ or error
    if isinstance(cause, KeyError):
        return f"The CIF is missing a required field ({cause})."
    detail = " ".join(str(error).split())
    if not detail or detail == "Invalid CIF file with no structures!":
        return "Check for a data_ block with unit-cell parameters and an atom-site loop."
    if "exceeded tolerance" in detail:
        return f"{detail} Correct the atom-site occupancies. Only sums up to {MAX_SITE_OCCUPANCY} can be normalized."
    return detail


def inputs_from_cif(content: bytes) -> CifInputs:
    """Extract a single structure from the contents of an uploaded CIF."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CifInputError("The CIF is not valid UTF-8 text. Save it as UTF-8 and upload it again.") from error
    try:
        parser = CifParser.from_str(text, occupancy_tolerance=MAX_SITE_OCCUPANCY)
        structures = parser.parse_structures(primitive=False, on_error="raise")
    except (ValueError, KeyError, IndexError, TypeError, ArithmeticError) as error:
        raise CifInputError(f"Could not parse a crystal structure from the CIF. {cif_parse_details(error)}") from error
    if len(structures) != 1:
        raise CifInputError("Provide a CIF containing a single crystal structure.")
    structure = structures[0]
    composition = structure.composition.get_el_amt_dict()
    if composition.get("Li", 0) <= 0:
        raise CifInputError("The CIF structure must contain lithium.")
    try:
        symmetry = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5)
        group = symmetry.get_space_group_number()
    except (ValueError, SymmetryUndeterminedError) as error:
        raise CifInputError("Could not determine the crystal symmetry from the CIF.") from error
    materials = pd.DataFrame(
        [(composition, *structure.lattice.parameters, group)],
        columns=INPUT_COLUMNS,
    )
    return CifInputs(materials, not structure.is_ordered, list(parser.warnings), structure)


def material_features(composition_value: str | dict, *lattice_and_symmetry: str | float) -> list[float]:
    composition = parse_composition(composition_value)
    if composition["Li"] <= 0:
        raise ValueError("The composition must contain lithium.")
    fractions = np.asarray([composition[element] / composition.num_atoms for element in ELEMENTS])
    mean_number = fractions @ ATOMIC_NUMBERS
    mean_mass = fractions @ ATOMIC_MASSES
    lengths_angles = np.asarray(lattice_and_symmetry[:-1], dtype=float)
    lengths, angles = lengths_angles[:3], lengths_angles[3:]
    if not np.isfinite(lengths_angles).all() or any(lengths <= 0) or any((angles <= 0) | (angles >= 180)):
        raise ValueError("Expected positive lattice lengths and angles between 0 and 180 degrees.")
    volume = Lattice.from_parameters(*lengths_angles).volume
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Expected a positive cell volume.")
    group = float(lattice_and_symmetry[-1])
    if not group.is_integer() or not 1 <= group <= 230:
        raise ValueError("Expected an international space group number from 1 to 230.")
    system_index = next(index for index, (_, upper) in enumerate(CRYSTAL_SYSTEMS) if group <= upper)
    return [
        *fractions,
        len(composition),
        mean_number,
        np.sqrt(fractions @ (ATOMIC_NUMBERS - mean_number) ** 2),
        mean_mass,
        np.sqrt(fractions @ (ATOMIC_MASSES - mean_mass) ** 2),
        volume / composition.num_atoms,
        *(float(index == system_index) for index in range(len(CRYSTAL_SYSTEMS))),
    ]


class MaterialFeatures(TransformerMixin, BaseEstimator):
    """Deterministic featurization with an explicit allowlist."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        records = select_inputs(X).itertuples(index=False, name=None)
        return pd.DataFrame(
            [material_features(*record) for record in records],
            columns=FEATURE_NAMES,
            index=X.index,
            dtype=float,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(FEATURE_NAMES, dtype=object)
