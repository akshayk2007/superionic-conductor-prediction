from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

from electrolyte_screening.data import TARGET, official_path, read_split, split_labels
from electrolyte_screening.features import INPUT_COLUMNS, MaterialFeatures, composition_key, select_inputs
from electrolyte_screening.models import candidates, predict_cif, training_folds

DATA_DIR = Path("data/raw")


def test_measurements_and_provenance_cannot_change_features():
    rows = read_split(DATA_DIR, "train").head(4)
    materials = select_inputs(rows)
    contaminated = rows.copy()
    for column in contaminated.columns.difference(INPUT_COLUMNS):
        contaminated[column] = "overwritten"
    contaminated["manufactured_leak"] = [1, 2, 3, 4]
    assert TARGET not in INPUT_COLUMNS
    transformer = MaterialFeatures().fit(materials)
    assert_frame_equal(transformer.transform(materials), transformer.transform(contaminated))


def test_training_folds_keep_equivalent_compositions_together():
    rows = read_split(DATA_DIR, "train")
    inputs, labels = select_inputs(rows), split_labels(rows)
    keys = inputs["True Composition"].map(composition_key)
    all_validation_rows = []
    for fitting, validation in training_folds(inputs, labels):
        assert set(keys.iloc[fitting]).isdisjoint(keys.iloc[validation])
        assert labels.iloc[fitting].nunique() == labels.iloc[validation].nunique() == 2
        all_validation_rows.extend(validation)
    assert sorted(all_validation_rows) == list(range(len(rows)))


def test_scaler_is_fitted_on_training_materials_only():
    rows = read_split(DATA_DIR, "train")
    inputs, labels = select_inputs(rows), split_labels(rows)
    fitting, validation = training_folds(inputs, labels)[0]
    model = candidates()["LogisticRegression"]
    with threadpool_limits(limits=1):
        model.fit(inputs.iloc[fitting], labels.iloc[fitting])
        trained_mean = model.named_steps["standardscaler"].mean_.copy()
        held_out = inputs.iloc[validation].assign(a=100)
        model.predict(held_out)
    expected = MaterialFeatures().transform(inputs.iloc[fitting]).mean()
    np.testing.assert_allclose(trained_mean, expected)
    np.testing.assert_array_equal(model.named_steps["standardscaler"].mean_, trained_mean)


def test_saved_pipeline_predicts_an_official_valid_cif(tmp_path):
    archive_path = official_path(DATA_DIR, "train_cifs.zip")
    cif = tmp_path / "sample.cif"
    with ZipFile(archive_path) as archive:
        cif.write_bytes(archive.read("train_randomized_cifs/jqc.cif"))
    rows = read_split(DATA_DIR, "train").head(100)
    model = candidates()["LogisticRegression"]
    with threadpool_limits(limits=1):
        model.fit(select_inputs(rows), split_labels(rows))
    artifact = tmp_path / "model.joblib"
    joblib.dump(model, artifact)
    result = predict_cif(artifact, cif)
    assert result["fast_conductor"] in (0, 1)
    assert 0 <= result["fast_conductor_probability"] <= 1
    assert result["threshold_s_per_cm"] == 1e-4
    assert result["training_coverage"]["reasons"] == [{"code": "coverage_profile_missing"}]
