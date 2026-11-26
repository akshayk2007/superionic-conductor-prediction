"""Local web interface for the verified screening pipeline."""

import hashlib
import json
import logging
import re
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.datastructures import UploadFile
from starlette.requests import ClientDisconnect

from .data import REVISION, THRESHOLD
from .features import LATTICE_COLUMNS, MAX_SITE_OCCUPANCY, SYSTEM_FEATURES, CifInputError
from .models import CifPrediction, score_cif
from .neighbors import NEIGHBOR_DISCLAIMER, NEIGHBOR_METHOD, NeighborIndex

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
MAX_FILES = 20
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_FILES * MAX_FILE_BYTES + 1024 * 1024
MAX_PENDING_UPLOADS = 4
VIEW_SITE_LIMIT = 2500
LOG = logging.getLogger(__name__)
LIMITATIONS = [
    "The fast-class probability is an uncalibrated model estimate. Use it for screening and prioritization.",
    "The CIF evaluation covers 67 of 121 official test entries. Some CIFs describe close matches to the measured material.",
    "The official train and test sets share two normalized compositions. Prospective performance remains unmeasured.",
    "The features omit local coordination, defects, grain boundaries, and measurement conditions.",
]

BATCH_ERROR = f"Select at most {MAX_FILES} CIF files per batch."
BODY_ERROR = "This upload exceeds the batch size limit. Split it into smaller batches."
FILE_GUIDE = "Select one or more CIF files using the files field."
NO_NEIGHBORS = "Known materials are unavailable. Prepare the local OBELiX training data to enable neighbors."
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


class UploadGuard:
    """Bound upload streams before multipart files are buffered."""

    def __init__(self, app):
        self.app = app
        self.pending = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/api/predict":
            await self.app(scope, receive, send)
            return
        length = dict(scope["headers"]).get(b"content-length", b"")
        # Bound conversion as well as the number of bytes the server will read.
        if length.isdigit() and (len(length) > 12 or int(length) > MAX_REQUEST_BYTES):
            await JSONResponse({"detail": BODY_ERROR}, status_code=413)(scope, receive, send)
            return
        if self.pending >= MAX_PENDING_UPLOADS:
            response = JSONResponse({"detail": "IonLab is busy. Please try your upload again shortly."}, status_code=503)
            await response(scope, receive, send)
            return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_REQUEST_BYTES:
                raise HTTPException(413, detail=BODY_ERROR)
            return message

        self.pending += 1
        try:
            await self.app(scope, limited_receive, send)
        finally:
            self.pending -= 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    verification = json.loads((ROOT / "reports/upload-verification.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "reports/benchmark.json").read_text(encoding="utf-8"))
    artifact_path = ROOT / "artifacts/fast_conductor.joblib"
    if not artifact_path.is_file():
        raise RuntimeError(f"Place the existing verified model at {artifact_path} before starting the app.")
    artifact = artifact_path.read_bytes()
    if hashlib.sha256(artifact).hexdigest() != verification["saved_artifact_sha256"]:
        raise RuntimeError("The saved model does not match the upload verification report. Restore the verified artifact.")
    pipeline = joblib.load(BytesIO(artifact))
    classifier = pipeline.steps[-1][1]
    app.state.pipeline = pipeline
    app.state.scoring_lock = Lock()
    try:
        app.state.neighbor_index = NeighborIndex(ROOT / "data/raw")
    except (OSError, ValueError):
        LOG.warning("OBELiX training references could not be loaded.", exc_info=True)
        app.state.neighbor_index = None
    feature_names = pipeline.named_steps["materialfeatures"].get_feature_names_out()
    importance = sorted(zip(feature_names, classifier.feature_importances_), key=lambda pair: -pair[1])
    cif = verification["cif_verification"]
    app.state.model = {
        "name": "Random forest",
        "classifier": type(classifier).__name__,
        "trees": classifier.n_estimators,
        "dataset": "NRC-Mila OBELiX",
        "dataset_url": "https://github.com/NRC-Mila/OBELiX",
        "dataset_revision": REVISION,
        "artifact_sha256": verification["saved_artifact_sha256"],
        "threshold_s_per_cm": float(THRESHOLD),
        "feature_count": len(feature_names),
        "limits": {"files_per_batch": MAX_FILES, "file_bytes": MAX_FILE_BYTES},
        "neighbor_method": NEIGHBOR_METHOD,
        "neighbor_disclaimer": NEIGHBOR_DISCLAIMER,
        "limitations": LIMITATIONS,
        "training_coverage_profile": verification["training_coverage_profile"],
        "csv_benchmark": benchmark,
        "cif_evaluation": {
            "counts": cif["evaluated"],
            "metrics": cif["cif_metrics"],
            "paired_csv_metrics": cif["csv_metrics_on_same_rows"],
        },
        "global_feature_importance": [
            {"feature": str(name), "importance": float(value)} for name, value in importance
        ],
    }
    yield


app = FastAPI(title="IonLab CIF screening", lifespan=lifespan)
app.add_middleware(UploadGuard)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(
        STATIC / "index.html",
        headers={
            "Content-Security-Policy": CONTENT_SECURITY_POLICY,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-cache",
        },
    )


@app.get("/api/model")
def model_details(request: Request) -> dict:
    return request.app.state.model


def describe_structure(prediction: CifPrediction) -> dict:
    metadata = prediction.cif.materials.iloc[0]
    features = prediction.features.iloc[0]
    structure = prediction.cif.structure
    # These markers are for inspection only. Inference has already used every parsed site.
    site_step = max(1, (len(structure) + VIEW_SITE_LIMIT - 1) // VIEW_SITE_LIMIT)
    sites = [
        {
            "fractional": site.frac_coords.tolist(),
            "species": site.species.element_composition.get_el_amt_dict(),
        }
        for site in structure.sites[::site_step]
    ]
    system = features.loc[list(SYSTEM_FEATURES)].idxmax().removeprefix("crystal_system_")
    return {
        "formula": structure.composition.element_composition.reduced_formula,
        "unit_cell_composition": metadata["True Composition"],
        "lattice": dict(zip(LATTICE_COLUMNS, map(float, structure.lattice.parameters))),
        "space_group_number": int(metadata["Space group #"]),
        "crystal_system": system,
        "features": {name: float(value) for name, value in features.items()},
        "structure": {
            "cell": structure.lattice.matrix.tolist(),
            "sites": sites,
            "site_count": len(structure),
        },
        **prediction.result,
    }


def public_cif_error(error: CifInputError) -> str:
    message = str(error)
    if not message.startswith("Could not parse a crystal structure from the CIF."):
        return message
    # Only technical details with a known shape leave the parser boundary.
    occupancy = re.search(r"Occupancy (\d+(?:\.\d+)?(?:e[+-]?\d+)?) exceeded tolerance", message, re.I)
    if occupancy:
        return (
            f"Occupancy {occupancy.group(1)} exceeded tolerance. Correct the atom-site occupancies. "
            f"Only sums up to {MAX_SITE_OCCUPANCY} can be normalized."
        )
    missing = re.search(r"missing a required field \('(_[a-zA-Z0-9_.]{1,90})'\)", message)
    if missing:
        return f"The CIF is missing a required field ('{missing.group(1)}')."
    return (
        "Could not parse a crystal structure from the CIF. Check for a data_ block with numeric "
        "unit-cell parameters and a complete atom-site loop."
    )


def screen_files(request: Request, files: list[UploadFile]) -> dict:
    results = []
    for uploaded in files:
        filename = (uploaded.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        result = {"filename": filename or "Unnamed file"}
        try:
            if not filename.lower().endswith(".cif"):
                raise CifInputError("Choose a file with a .cif extension.")
            content = uploaded.file.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise CifInputError("This CIF exceeds the 2 MiB file limit.")
            if not content:
                raise CifInputError("The CIF file is empty.")
            with request.app.state.scoring_lock:
                prediction = score_cif(request.app.state.pipeline, content)
            result.update(ok=True, **describe_structure(prediction))
        except CifInputError as error:
            result.update(ok=False, error=public_cif_error(error))
        except Exception:
            LOG.exception("Failed to screen a CIF")
            result.update(ok=False, error="IonLab could not process this CIF. Try uploading a different file.")
        else:
            index = request.app.state.neighbor_index
            result["known_materials"] = {"unavailable": NO_NEIGHBORS}
            if index is not None:
                try:
                    result["known_materials"] = index.find(prediction, content)
                except Exception:
                    LOG.exception("Could not look up training references")
        results.append(result)
    return {"results": results}


@app.post("/api/predict")
async def upload(request: Request) -> JSONResponse:
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise HTTPException(422, detail=FILE_GUIDE)
    try:
        async with request.form(max_files=MAX_FILES, max_fields=0) as form:
            files = form.getlist("files")
            if not files or len(files) != len(form.multi_items()) or not all(isinstance(file, UploadFile) for file in files):
                raise HTTPException(422, detail=FILE_GUIDE)
            batch = await run_in_threadpool(screen_files, request, files)
    except ClientDisconnect as error:
        raise HTTPException(400, detail="The upload was interrupted. Please try again.") from error
    except StarletteHTTPException as error:
        if error.status_code != 400:
            raise
        if str(error.detail).startswith("Too many files"):
            raise HTTPException(413, detail=BATCH_ERROR) from error
        raise HTTPException(400, detail="The upload form was incomplete or malformed. Select the files again.") from error
    return JSONResponse(batch, headers={"Cache-Control": "no-store"})
