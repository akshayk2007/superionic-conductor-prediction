from pathlib import Path

import joblib
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

from electrolyte_screening.data import read_split, split_labels
from electrolyte_screening.features import CifInputError, VOLUME_FEATURE, composition_key, select_inputs
from electrolyte_screening.models import candidates, predict_cif, save_screening_model, score_cif

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    rows = read_split(Path("data/raw"), "train")
    inputs = select_inputs(rows)
    pipeline = candidates()["RandomForestClassifier"]
    path = tmp_path_factory.mktemp("upload") / "model.joblib"
    with threadpool_limits(limits=1):
        pipeline.fit(inputs, split_labels(rows))
        save_screening_model(pipeline, inputs, path)
    return path, inputs


def test_unseen_lithium_cif_uses_extracted_features(trained_model):
    artifact, inputs = trained_model
    assert composition_key("LiI") not in set(inputs["True Composition"].map(composition_key))
    path = FIXTURES / "unseen_lii.cif"
    pipeline = joblib.load(artifact)
    prediction = score_cif(pipeline, path.read_bytes())
    features = prediction.features.iloc[0]
    assert features["fraction_Li"] == features["fraction_I"] == 0.5
    assert features[VOLUME_FEATURE] == pytest.approx(27)
    assert features["crystal_system_cubic"] == 1
    assert prediction.result["fast_conductor"] == pipeline[1:].predict(prediction.features)[0]
    assert prediction.result["fast_conductor_probability"] == pipeline[1:].predict_proba(prediction.features)[0, 1]
    assert prediction.result == predict_cif(artifact, path)


@pytest.mark.parametrize(
    ("filename", "message"),
    [("lithium_free.cif", "must contain lithium"), ("invalid.cif", "Could not parse")],
)
def test_invalid_uploads_get_a_clear_error(trained_model, filename, message):
    artifact, _ = trained_model
    with pytest.raises(CifInputError, match=message):
        predict_cif(artifact, FIXTURES / filename)


def test_partial_occupancy_and_header_changes(trained_model):
    artifact, _ = trained_model
    pipeline = joblib.load(artifact)
    content = (FIXTURES / "partial_occupancy.cif").read_bytes()
    original = score_cif(pipeline, content)
    assert original.result["has_partial_occupancies"] is True
    assert original.features["fraction_Li"].iloc[0] == pytest.approx(1 / 3)
    assert original.features[VOLUME_FEATURE].iloc[0] == pytest.approx(64 / 1.5)
    altered = content.replace(b"data_partial_occupancy", b"data_8po").replace(b"Li0.5 Cl1", b"Na1 Cl1")
    with pytest.warns(UserWarning, match="composition"):
        renamed = score_cif(pipeline, altered)
    assert_frame_equal(original.features, renamed.features)
    assert original.result["fast_conductor_probability"] == renamed.result["fast_conductor_probability"]
    full_occupancy = content.replace(b"0.0 0.0 0.0 0.5", b"0.0 0.0 0.0 1").replace(b"Li0.5", b"Li1")
    full = score_cif(pipeline, full_occupancy)
    assert full.result["has_partial_occupancies"] is False
    assert full.features["fraction_Li"].iloc[0] == pytest.approx(0.5)
    assert full.features[VOLUME_FEATURE].iloc[0] == pytest.approx(32)


def test_coverage_marks_unseen_elements_and_out_of_range_structure(trained_model):
    artifact, _ = trained_model
    pipeline = joblib.load(artifact)
    content = (FIXTURES / "unseen_lii.cif").read_bytes()
    altered = content.replace(b"I4", b"Tl4").replace(b" I ", b" Tl ").replace(b" 6.0", b" 60.0")
    result = score_cif(pipeline, altered).result["training_coverage"]
    assert result["warning"] is True
    reasons = {reason["code"]: reason for reason in result["reasons"]}
    assert reasons["unseen_elements"]["elements"] == ["Tl"]
    assert "volume_outside_training_range" in reasons
    assert "distant_composition" in reasons
    assert result["nearest_composition_distance"] > result["composition_distance_threshold"]


def test_coverage_matches_training_features(trained_model):
    artifact, inputs = trained_model
    pipeline = joblib.load(artifact)
    features = pipeline.named_steps["materialfeatures"].transform(inputs).iloc[0]
    result = pipeline.training_coverage_.check(features)
    assert result["warning"] is False
    np.testing.assert_allclose(result["nearest_composition_distance"], 0)
