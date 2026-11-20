from pathlib import Path
import io
import warnings

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifParser


app = FastAPI(title="LiScreen | Solid Electrolyte Screening")
BASE_DIR = Path(__file__).resolve().parent

# Automatically locate the trained model.
MODEL_CANDIDATES = ["rf_model.joblib", "superionic_sendek5_rf.joblib"]
model_path = next(
    (BASE_DIR / name for name in MODEL_CANDIDATES if (BASE_DIR / name).exists()),
    None,
)
if model_path is None:
    joblib_files = sorted(BASE_DIR.glob("*.joblib"))
    if not joblib_files:
        raise RuntimeError(f"No .joblib model file found in {BASE_DIR}")
    model_path = joblib_files[0]

loaded = joblib.load(model_path)
model = loaded["pipeline"] if isinstance(loaded, dict) else loaded

# Keep this order exactly aligned with the trained model.
FEATURES = ["LASD", "SBI", "AFC", "LLSD", "LLB"]
MAX_FILE_SIZE = 10 * 1024 * 1024
BOND_CUTOFF = 4.0
OCCUPANCY_TOLERANCE = 1.01


class CrystalInput(BaseModel):
    LASD: float
    SBI: float
    AFC: float
    LLSD: float
    LLB: float


def get_probability_class_index(class_label=1):
    """Return the probability column for the requested model class."""
    classes = list(model.classes_)
    if class_label not in classes:
        raise ValueError(f"The trained model does not contain class {class_label}.")
    return classes.index(class_label)


def parse_cif_text(cif_text: str) -> Structure:
    """
    Parse CIF files using pymatgen's CIF parser.

    A small occupancy tolerance handles rounding such as 0.759 + 0.242 = 1.001.
    It does not repair substantially invalid occupancies or missing structural data.
    """
    try:
        parser = CifParser(
            io.StringIO(cif_text),
            occupancy_tolerance=OCCUPANCY_TOLERANCE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structures = parser.parse_structures(primitive=False)

        if not structures:
            raise ValueError("No crystal structures were found in this CIF.")
        return structures[0]
    except Exception as exc:
        raise ValueError(f"Could not read CIF structure: {exc}") from exc


def _site_species(site):
    """Return [(element_symbol, occupancy, electronegativity), ...] for a site."""
    components = []
    for specie, occupancy in site.species.items():
        element = getattr(specie, "element", specie)
        symbol = element.symbol
        electronegativity = Element(symbol).X
        if electronegativity is None:
            raise ValueError(
                f"Pauling electronegativity is unavailable for {symbol}."
            )
        components.append((symbol, float(occupancy), float(electronegativity)))
    return components


def extract_sendek_descriptors(structure: Structure) -> dict:
    """
    Calculate occupancy-aware approximations of the five project descriptors.

    Ordered CIFs follow the same geometric approach as the previous implementation.
    For disordered CIFs, species occupancies are used as weights for site and
    pair contributions. This is an approximation; it is not guaranteed to be
    identical to the descriptor-generation procedure used for model training.
    """
    if len(structure) == 0:
        raise ValueError("The CIF contains no atomic sites.")

    site_components = [_site_species(site) for site in structure]
    if not any(symbol == "Li" and occ > 0 for comps in site_components for symbol, occ, _ in comps):
        raise ValueError("No lithium (Li) sites were found in this CIF.")

    # A site can contain more than one element. Keep its fractional occupancies.
    li_occupancy = []
    non_li_components = []
    anion_occupancy = []
    highest_non_li_en = max(
        (
            en
            for comps in site_components
            for symbol, occ, en in comps
            if symbol != "Li" and occ > 0
        ),
        default=None,
    )
    if highest_non_li_en is None:
        raise ValueError("The structure contains lithium only; no framework exists.")

    for comps in site_components:
        li_occ = sum(occ for symbol, occ, _ in comps if symbol == "Li")
        li_occupancy.append(li_occ)

        non_li = [
            (symbol, occ, en)
            for symbol, occ, en in comps
            if symbol != "Li" and occ > 0
        ]
        non_li_components.append(non_li)

        # Preserve the previous project's anion definition: the most
        # electronegative non-Li element(s) in the structure.
        anion_occ = sum(
            occ for symbol, occ, en in non_li
            if np.isclose(en, highest_non_li_en)
        )
        anion_occupancy.append(anion_occ)

    li_indices = [i for i, occ in enumerate(li_occupancy) if occ > 0]
    anion_indices = [i for i, occ in enumerate(anion_occupancy) if occ > 0]
    if not anion_indices:
        raise ValueError("Could not identify anion sites from the CIF.")

    lattice = structure.lattice
    cart_coords = np.asarray(structure.cart_coords, dtype=float)
    translations = [
        np.asarray([a, b, c], dtype=float) @ lattice.matrix
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
    ]

    # Store each crystallographic site and its periodic images.
    periodic_positions = []
    for image in translations:
        for j, position in enumerate(cart_coords):
            periodic_positions.append((j, image, position + image))

    def site_distances(center_index, candidate_indices):
        center = cart_coords[center_index]
        found = []
        for j, image, position in periodic_positions:
            if j not in candidate_indices:
                continue
            if j == center_index and np.allclose(image, 0.0):
                continue
            distance = float(np.linalg.norm(position - center))
            if distance > 1e-8:
                found.append((j, distance))
        return found

    def weighted_average(values, weights, description):
        valid = [
            (float(value), float(weight))
            for value, weight in zip(values, weights)
            if weight > 0 and np.isfinite(value)
        ]
        total_weight = sum(weight for _, weight in valid)
        if total_weight <= 0:
            raise ValueError(f"Could not calculate {description}.")
        return sum(value * weight for value, weight in valid) / total_weight

    # LASD: mean nearest distance from Li-bearing sites to anion-bearing sites.
    lasd_values, lasd_weights = [], []
    for i in li_indices:
        candidates = site_distances(i, set(anion_indices))
        if not candidates:
            raise ValueError("Could not find periodic anion neighbors for lithium sites.")
        lasd_values.append(min(distance for _, distance in candidates))
        lasd_weights.append(li_occupancy[i])
    lasd = weighted_average(lasd_values, lasd_weights, "LASD")

    # LLSD: mean nearest Li-Li distance, weighted by Li occupancy at each center.
    llsd_values, llsd_weights = [], []
    for i in li_indices:
        candidates = site_distances(i, set(li_indices))
        if not candidates:
            raise ValueError("Could not find another periodic lithium neighbor.")
        llsd_values.append(min(distance for _, distance in candidates))
        llsd_weights.append(li_occupancy[i])
    llsd = weighted_average(llsd_values, llsd_weights, "LLSD")

    # LLB: expected number of Li neighbors within the project's 4 Å cutoff.
    llb_values, llb_weights = [], []
    for i in li_indices:
        candidates = site_distances(i, set(li_indices))
        expected_count = sum(
            li_occupancy[j] for j, distance in candidates
            if distance <= BOND_CUTOFF
        )
        llb_values.append(expected_count)
        llb_weights.append(li_occupancy[i])
    llb = weighted_average(llb_values, llb_weights, "LLB")

    # SBI: occupancy-weighted mean electronegativity difference for non-Li
    # atom pairs within 4 Å. Species on the same crystallographic site are
    # not treated as bonded to one another.
    sbi_numerator = 0.0
    sbi_denominator = 0.0
    for i, components_i in enumerate(non_li_components):
        center = cart_coords[i]
        for j, image, position in periodic_positions:
            if j == i and np.allclose(image, 0.0):
                continue
            distance = float(np.linalg.norm(position - center))
            if not (1e-8 < distance <= BOND_CUTOFF):
                continue
            for symbol_i, occ_i, en_i in components_i:
                for symbol_j, occ_j, en_j in non_li_components[j]:
                    pair_weight = occ_i * occ_j
                    if pair_weight <= 0:
                        continue
                    sbi_numerator += abs(en_i - en_j) * pair_weight
                    sbi_denominator += pair_weight
    if sbi_denominator <= 0:
        raise ValueError(
            "No non-lithium bonds were found within the 4 Å cutoff; SBI cannot be calculated."
        )
    sbi = sbi_numerator / sbi_denominator

    # AFC: occupancy-weighted count of anion neighbors within 1 Å of each
    # anion site's nearest anion-neighbor distance.
    afc_values, afc_weights = [], []
    for i in anion_indices:
        candidates = site_distances(i, set(anion_indices))
        if not candidates:
            raise ValueError("Could not find periodic anion neighbors to calculate AFC.")
        nearest = min(distance for _, distance in candidates)
        expected_count = sum(
            anion_occupancy[j]
            for j, distance in candidates
            if nearest - 1e-8 <= distance <= nearest + 1.0 + 1e-8
        )
        afc_values.append(expected_count)
        afc_weights.append(anion_occupancy[i])
    afc = weighted_average(afc_values, afc_weights, "AFC")

    values = {
        "LASD": float(lasd),
        "SBI": float(sbi),
        "AFC": float(afc),
        "LLSD": float(llsd),
        "LLB": float(llb),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Descriptor extraction produced a non-finite value.")
    return values


def make_prediction(descriptors: dict) -> dict:
    input_df = pd.DataFrame(
        [[descriptors[name] for name in FEATURES]],
        columns=FEATURES,
    )
    probabilities = model.predict_proba(input_df)[0]
    class_index = get_probability_class_index(1)
    probability = float(probabilities[class_index])
    is_superionic = bool(probability >= 0.5)
    return {
        "is_superionic": is_superionic,
        "probability": round(probability * 100, 2),
        "verdict": "Superionic conductor" if is_superionic else "Poor conductor",
    }


# --- Static file endpoints ---

@app.get("/")
def serve_ui():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/style.css")
def serve_css():
    return FileResponse(BASE_DIR / "style.css")


@app.get("/script.js")
def serve_js():
    return FileResponse(BASE_DIR / "script.js")


@app.get("/logo.png")
def serve_logo():
    logo_path = BASE_DIR / "logo.png"
    if logo_path.exists():
        return FileResponse(logo_path)
    return Response(status_code=204)


# --- API endpoints ---

@app.post("/predict")
def predict(data: CrystalInput):
    """Manual descriptor screening endpoint."""
    descriptors = data.model_dump()
    return make_prediction(descriptors)


@app.post("/predict-cif")
async def predict_cif(file: UploadFile = File(...)):
    """Parse CIF, calculate descriptors, and run the trained model."""
    filename = file.filename or "uploaded.cif"
    if not filename.lower().endswith(".cif"):
        raise HTTPException(status_code=400, detail="Please upload a valid .cif file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded CIF file is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum size is 10 MB.",
        )

    try:
        cif_text = contents.decode("utf-8-sig", errors="strict")
        structure = parse_cif_text(cif_text)
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="The CIF file is not valid UTF-8 text.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        descriptors = extract_sendek_descriptors(structure)
        prediction = make_prediction(descriptors)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CIF processing failed: {exc}",
        ) from exc

    has_disorder = any(not site.is_ordered for site in structure)
    response = {
        "filename": filename,
        "formula": structure.composition.reduced_formula,
        "number_of_sites": len(structure),
        "lattice_parameters": {
            "a": round(float(structure.lattice.a), 4),
            "b": round(float(structure.lattice.b), 4),
            "c": round(float(structure.lattice.c), 4),
        },
        "descriptors": {
            key: round(float(value), 6)
            for key, value in descriptors.items()
        },
        "prediction": prediction,
        "descriptor_method": (
            "occupancy_weighted_approximation" if has_disorder else "ordered_structure"
        ),
        "warnings": (
            [
                "This CIF contains partial or mixed occupancies. "
                "Descriptors use an occupancy-weighted approximation. "
                "The trained model may have been trained on descriptors generated "
                "with a different procedure; validate this prediction before "
                "treating it as scientifically confirmed."
            ]
            if has_disorder else []
        ),
    }
    return response


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    """Batch CSV screening endpoint for multiple candidate rows."""
    filename = file.filename or "uploaded.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid .csv file.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 10 MB.",
            )
        df = pd.read_csv(io.BytesIO(contents))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV file: {exc}",
        ) from exc

    # Normalize headers so LASD, lasd, and " LASD " are treated alike.
    normalized_columns = {
        str(column).strip().upper(): column for column in df.columns
    }
    required_cols = ["LASD", "SBI", "AFC", "LLSD", "LLB"]
    missing_cols = [
        column for column in required_cols
        if column not in normalized_columns
    ]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required columns: {', '.join(missing_cols)}. "
                "Expected: LASD, SBI, AFC, LLSD, LLB"
            ),
        )

    results = []
    successful = 0
    failed = 0
    material_column = normalized_columns.get("MATERIAL")

    for idx, row in df.iterrows():
        material_name = (
            str(row[material_column])
            if material_column is not None
            else f"Row {idx + 1}"
        )
        try:
            features = {
                name: float(row[normalized_columns[name]])
                for name in required_cols
            }
            if not all(np.isfinite(value) for value in features.values()):
                raise ValueError("Contains non-numeric, NaN, or infinite values.")

            prediction = make_prediction(features)
            results.append({
                "row": idx + 1,
                "material": material_name,
                "is_superionic": prediction["is_superionic"],
                "verdict": prediction["verdict"],
                "probability": prediction["probability"],
                "error": None,
            })
            successful += 1
        except Exception as row_err:
            results.append({
                "row": idx + 1,
                "material": material_name,
                "is_superionic": False,
                "verdict": "Error processing row",
                "probability": None,
                "error": str(row_err),
            })
            failed += 1

    return {
        "filename": filename,
        "total_rows": len(df),
        "successful_predictions": successful,
        "failed_rows": failed,
        "results": results,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
        "features": FEATURES,
    }
