import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pytest
import numpy as np

pytest.importorskip("httpx2")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from electrolyte_screening.data import TARGET, official_path, read_split
from electrolyte_screening.features import CRYSTAL_SYSTEMS, ELEMENTS, parse_composition
from electrolyte_screening.models import predict_cif, score_cif
from electrolyte_screening.neighbors import NEIGHBOR_DISCLAIMER, NeighborIndex
from electrolyte_screening.web import MAX_FILE_BYTES, MAX_FILES, ROOT, app

FIXTURES = Path(__file__).parent / "fixtures"
MODEL = ROOT / "artifacts/fast_conductor.joblib"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def upload(filename: str, fixture: str):
    return "files", (filename, (FIXTURES / fixture).read_bytes(), "chemical/x-cif")


def test_app_serves_static_assets_and_verified_results(client):
    home = client.get("/")
    assert home.status_code == 200 and "Screen lithium" in home.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    response = client.get("/api/model")
    assert response.status_code == 200
    model = response.json()
    assert model["artifact_sha256"] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert model["threshold_s_per_cm"] == 1e-4
    assert model["csv_benchmark"]["test"]["rows"] == 121
    assert len(model["csv_benchmark"]["test_scores"]) == 4
    assert model["cif_evaluation"]["counts"]["rows"] == 67
    assert model["cif_evaluation"]["metrics"]["recall"] == 0.50
    assert model["neighbor_disclaimer"] == NEIGHBOR_DISCLAIMER
    assert len(model["global_feature_importance"]) == model["feature_count"] == 131


def test_single_unseen_cif_matches_pipeline_independent_of_filename(client):
    response = client.post("/api/predict", files=[upload("8po.cif", "unseen_lii.cif")])
    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["ok"] is True
    expected = predict_cif(MODEL, FIXTURES / "unseen_lii.cif")
    assert result["fast_conductor"] == expected["fast_conductor"]
    assert result["fast_conductor_probability"] == expected["fast_conductor_probability"]
    assert result["unit_cell_composition"] == {"Li": 4, "I": 4}
    assert result["crystal_system"] == "cubic"
    assert result["lattice"]["a"] == 6
    assert result["features"]["volume_per_atom_angstrom3"] == pytest.approx(27)
    assert len(result["structure"]["sites"]) == 8


def test_mixed_batch_keeps_valid_occupancies_and_returns_file_errors(client):
    response = client.post("/api/predict", files=[
        upload("lithium-free.cif", "lithium_free.cif"),
        upload("partial.cif", "partial_occupancy.cif"),
        upload("broken.cif", "invalid.cif"),
        upload("unseen.CIF", "unseen_lii.cif"),
    ])
    assert response.status_code == 200
    no_lithium, partial, bad, unseen = response.json()["results"]
    assert no_lithium["ok"] is False and "must contain lithium" in no_lithium["error"]
    assert bad["ok"] is False and "Could not parse" in bad["error"]
    assert unseen["ok"] is True
    assert partial["ok"] is True
    assert partial["has_partial_occupancies"] is True
    assert partial["features"]["fraction_Li"] == pytest.approx(1 / 3)
    assert {"Li": 0.5} in [site["species"] for site in partial["structure"]["sites"]]


def test_cod_1545083_upload_keeps_both_elements_on_the_mixed_site(client):
    response = client.post("/api/predict", files=[upload("1545083.cif", "1545083.cif")])
    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["ok"], result
    expected_site = {"Zr": 0.759 / 1.001, "Ta": 0.242 / 1.001}
    mixed_sites = [site["species"] for site in result["structure"]["sites"] if set(site["species"]) == {"Zr", "Ta"}]
    assert len(mixed_sites) == 16
    for species in mixed_sites:
        assert species == pytest.approx(expected_site)
        assert sum(species.values()) == pytest.approx(1)
    composition = result["unit_cell_composition"]
    assert composition == pytest.approx({"Li": 51.36, "La": 24, "O": 96, **{
        element: len(mixed_sites) * occupancy for element, occupancy in expected_site.items()
    }})
    for element in expected_site:
        assert result["features"][f"fraction_{element}"] == pytest.approx(composition[element] / sum(composition.values()))
    assert result["has_partial_occupancies"] is True
    assert any("1.001" in message for message in result["cif_parser_warnings"])
    predicted = predict_cif(MODEL, FIXTURES / "1545083.cif")
    assert result["fast_conductor"] == predicted["fast_conductor"]
    assert result["fast_conductor_probability"] == predicted["fast_conductor_probability"]


def test_parse_errors_explain_how_to_correct_the_cif(client):
    excessive = (FIXTURES / "1545083.cif").read_bytes().replace(b"ZrTa 0.2420", b"ZrTa 0.3000")
    files = [
        ("files", ("excessive.cif", excessive)),
        ("files", ("missing-sites.cif", b"data_bad\n_cell_length_a 4\n")),
        ("files", ("encoding.cif", b"\xff")),
        upload("no-data-block.cif", "invalid.cif"),
    ]
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    results = response.json()["results"]
    assert all(not result["ok"] for result in results)
    assert "Occupancy 1.059 exceeded tolerance" in results[0]["error"]
    assert "Only sums up to 1.002 can be normalized" in results[0]["error"]
    assert "missing a required field ('_atom_site_label')" in results[1]["error"]
    assert "Save it as UTF-8" in results[2]["error"]
    assert "Check for a data_ block" in results[3]["error"]


def test_coverage_and_file_validation(client):
    unseen = (FIXTURES / "unseen_lii.cif").read_bytes().replace(b" 6.0", b" 60.0")
    response = client.post("/api/predict", files=[
        ("files", ("large-cell.cif", unseen)),
        ("files", ("oversize.cif", b"a" * (MAX_FILE_BYTES + 1))),
        ("files", ("empty.cif", b"")),
        upload("wrong.txt", "unseen_lii.cif"),
    ])
    assert response.status_code == 200
    warning, large, empty, wrong = response.json()["results"]
    assert warning["ok"] and warning["training_coverage"]["warning"]
    assert "volume_outside_training_range" in {item["code"] for item in warning["training_coverage"]["reasons"]}
    assert large["ok"] is False and "2 MiB" in large["error"]
    assert empty["ok"] is False and "empty" in empty["error"]
    assert wrong["ok"] is False and ".cif extension" in wrong["error"]
    assert client.post("/api/predict", files=[upload("x.cif", "invalid.cif")] * (MAX_FILES + 1)).status_code == 413
    assert client.post("/api/predict").status_code == 422


def test_all_archived_cifs_match_the_verified_scores(client):
    data_dir = ROOT / "data/raw"
    if not (data_dir / "test_cifs.zip").is_file():
        pytest.skip("Download the official data to exercise all 67 archived CIFs.")
    report = json.loads((ROOT / "reports/upload-verification.json").read_text(encoding="utf-8"))
    verified = report["cif_verification"]["rows"]
    assert len(verified) == 67
    with ZipFile(official_path(data_dir, "test_cifs.zip")) as archive:
        for first in range(0, len(verified), MAX_FILES):
            batch = verified[first:first + MAX_FILES]
            files = [
                ("files", (PurePosixPath(row["archive_member"]).name, archive.read(row["archive_member"])))
                for row in batch
            ]
            response = client.post("/api/predict", files=files)
            assert response.status_code == 200
            for actual, row in zip(response.json()["results"], batch, strict=True):
                assert actual["ok"], actual
                expected = row["cif_prediction"]
                assert actual["fast_conductor"] == expected["fast_conductor"]
                assert actual["fast_conductor_probability"] == expected["fast_conductor_probability"]
                assert actual["training_coverage"] == expected["training_coverage"]


def test_neighbors_match_independent_training_ranking_and_measurements(client):
    data_dir = ROOT / "data/raw"
    train = read_split(data_dir, "train")
    response = client.post("/api/predict", files=[upload("garnet.cif", "1545083.cif")])
    (result,) = response.json()["results"]
    assert result["ok"]
    query = np.asarray([result["features"][f"fraction_{element.symbol}"] for element in ELEMENTS])
    expected = []
    for row in train.to_dict("records"):
        material = parse_composition(row["True Composition"]).fractional_composition
        vector = np.asarray([material[element] for element in ELEMENTS])
        overlap = 1 - np.abs(query - vector).sum() / 2
        system = next(name for name, upper in CRYSTAL_SYSTEMS if int(row["Space group #"]) <= upper)
        expected.append((0.85 * overlap + 0.15 * (system == result["crystal_system"]), overlap, row))
    expected.sort(key=lambda entry: -entry[0])
    known = result["known_materials"]
    assert known["exact_matches"] == []
    assert len(known["neighbors"]) == 3
    for neighbor, (score, overlap, row) in zip(known["neighbors"], expected[:3], strict=True):
        assert neighbor["formula"] == row["Reduced Composition"]
        assert neighbor["reported_conductivity"] == row[TARGET]
        assert neighbor["doi_url"] == "https://doi.org/" + row["DOI"]
        assert neighbor["similarity"] == pytest.approx(score)
        assert neighbor["composition_overlap"] == pytest.approx(overlap)
        assert (neighbor["provenance"] == "close_match") == (row["close match"].lower() == "yes")
    assert [n["provenance"] for n in known["neighbors"]] == ["close_match", "close_match", "csv_only"]


def test_archived_training_identity_is_separate_from_neighbors(client):
    with ZipFile(official_path(ROOT / "data/raw", "train_cifs.zip")) as archive:
        raw = archive.read("train_randomized_cifs/jqc.cif")
        proxy = archive.read("train_randomized_cifs/bq9.cif")
    response = client.post("/api/predict", files=[
        ("files", ("renamed.cif", raw)),
        ("files", ("reserialized.cif", b"# Alternate serialization\n" + raw)),
        ("files", ("close-match.cif", proxy)),
    ])
    assert response.status_code == 200
    same_file, equivalent, close_match = response.json()["results"]
    for result, match_type in ((same_file, "same_file"), (equivalent, "equivalent_structure")):
        known = result["known_materials"]
        (exact,) = known["exact_matches"]
        assert exact["formula"] == "Li7BiO6" and exact["match_type"] == match_type
        assert exact["provenance"] == "unflagged_archive"
        assert all(n["formula"] != exact["formula"] for n in known["neighbors"])
    assert any(match["provenance"] == "close_match" for match in close_match["known_materials"]["exact_matches"])


def test_neighbor_index_ignores_targets_and_handles_missing_symmetry_and_fields(client, monkeypatch):
    original = read_split(ROOT / "data/raw", "train").iloc[:7].copy()
    original.loc[original.index[0], ["Space group #", TARGET, "DOI"]] = ""
    original.loc[original.index[-1], "True Composition"] = ""
    original.loc[original.index[1], TARGET] = "<1E-10"
    metadata = original.copy()

    def fake_read(data_dir, split):
        assert split == "train"
        return metadata

    def missing_archive(*_):
        raise FileNotFoundError

    monkeypatch.setattr("electrolyte_screening.neighbors.read_split", fake_read)
    monkeypatch.setattr("electrolyte_screening.neighbors.official_path", missing_archive)
    content = (FIXTURES / "unseen_lii.cif").read_bytes()
    prediction = score_cif(app.state.pipeline, content)
    index = NeighborIndex(ROOT / "data/raw")
    assert len(index.references) == 6
    assert index.references[0].doi_url is None and index.references[0].crystal_system is None
    assert index.references[1].reported_conductivity == "<1E-10"
    before = index.find(prediction, content)
    metadata[[TARGET, "DOI", "ID"]] = ["0.99", "https://elsewhere.example/not-a-doi", "irrelevant"]
    after = NeighborIndex(ROOT / "data/raw").find(prediction, content)
    assert not before["archive_available"] and not after["archive_available"]
    keys = ("formula", "similarity", "composition_overlap", "same_crystal_system")
    assert [[r[key] for key in keys] for r in before["neighbors"]] == [
        [r[key] for key in keys] for r in after["neighbors"]
    ]
    assert all(r["doi_url"] is None for r in after["neighbors"])


def test_batch_stream_limit_and_malformed_form(client, monkeypatch):
    monkeypatch.setattr("electrolyte_screening.web.MAX_REQUEST_BYTES", 256)
    headers = {"Content-Type": "multipart/form-data; boundary=ionlab"}
    oversize = client.post("/api/predict", content=b"A" * 300, headers=headers)
    assert oversize.status_code == 413
    prefix = b'--ionlab\r\nContent-Disposition: form-data; name="files"; filename="a.cif"\r\n\r\n'
    streamed = client.post(
        "/api/predict", content=iter([prefix, b"A" * 180, b"A" * 180]), headers=headers,
    )
    assert streamed.status_code == 413
    assert "batch size limit" in streamed.json()["detail"]
    invalid = client.post("/api/predict", content=b"not a multipart body", headers=headers)
    assert invalid.status_code == 400
    assert "malformed" in invalid.json()["detail"]


def test_private_files_and_unexpected_errors_are_not_exposed(client, monkeypatch):
    for path in (
        "/artifacts/fast_conductor.joblib",
        "/data/raw/train.csv",
        "/reports/upload-verification.json",
        "/static/%2e%2e/%2e%2e/data/raw/train.csv",
    ):
        assert client.get(path).status_code == 404

    def fail(*_):
        raise RuntimeError("Internal path example-private-file.txt")

    monkeypatch.setattr("electrolyte_screening.web.score_cif", fail)
    response = client.post("/api/predict", files=[upload("user.cif", "unseen_lii.cif")])
    assert response.status_code == 200 and response.json()["results"][0]["ok"] is False
    assert "example-private-file" not in response.text and "Traceback" not in response.text
