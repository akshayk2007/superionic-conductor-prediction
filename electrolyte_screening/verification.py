"""Verify the CIF upload path against the official test data and CSV benchmark."""

import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from .data import REVISION, TARGET, official_path, read_split, split_labels
from .features import (
    FRACTION_FEATURES,
    SYSTEM_FEATURES,
    VOLUME_FEATURE,
    CifInputError,
    composition_key,
    select_inputs,
)
from .models import MAJORITY_BASELINE, candidates, class_counts, score, score_cif

FRACTION_TOLERANCE = 0.0001
VOLUME_RELATIVE_TOLERANCE = 0.01
SUMMARY_RELATIVE_TOLERANCE = 0.0001
SUMMARY_ABSOLUTE_TOLERANCE = 0.000001
SUMMARY_FEATURES = ("mean_atomic_number", "std_atomic_number", "mean_atomic_mass_amu", "std_atomic_mass_amu")


def compare_features(csv: pd.Series, cif: pd.Series) -> dict:
    csv_fractions, cif_fractions = csv.loc[list(FRACTION_FEATURES)], cif.loc[list(FRACTION_FEATURES)]
    fraction_difference = (cif_fractions - csv_fractions).abs()
    changed_fractions = fraction_difference.gt(FRACTION_TOLERANCE) | (
        csv_fractions.gt(0) != cif_fractions.gt(0)
    )
    csv_volume, cif_volume = float(csv[VOLUME_FEATURE]), float(cif[VOLUME_FEATURE])
    volume_change = abs(cif_volume - csv_volume) / csv_volume
    csv_system = csv.loc[list(SYSTEM_FEATURES)].idxmax().removeprefix("crystal_system_")
    cif_system = cif.loc[list(SYSTEM_FEATURES)].idxmax().removeprefix("crystal_system_")
    summary_changes = [
        field for field in SUMMARY_FEATURES
        if not np.isclose(cif[field], csv[field], rtol=SUMMARY_RELATIVE_TOLERANCE,
                          atol=SUMMARY_ABSOLUTE_TOLERANCE)
    ]
    return {
        "composition_mismatch": bool(changed_fractions.any()),
        "max_atomic_fraction_difference": float(fraction_difference.max()),
        "changed_atomic_fractions": {
            field.removeprefix("fraction_"): [float(csv[field]), float(cif[field])]
            for field in changed_fractions.index[changed_fractions]
        },
        "cell_volume_per_atom_angstrom3": [csv_volume, cif_volume],
        "cell_volume_per_atom_absolute_difference_angstrom3": abs(cif_volume - csv_volume),
        "cell_volume_per_atom_relative_difference": volume_change,
        "cell_volume_per_atom_mismatch": volume_change > VOLUME_RELATIVE_TOLERANCE,
        "crystal_system": [csv_system, cif_system],
        "crystal_system_mismatch": csv_system != cif_system,
        "element_count": [int(csv["element_count"]), int(cif["element_count"])],
        "element_count_mismatch": bool(csv["element_count"] != cif["element_count"]),
        "changed_weighted_summaries": {field: [float(csv[field]), float(cif[field])] for field in summary_changes},
    }


def verify_archived_cifs(
    pipeline, test_rows: pd.DataFrame, test_y: pd.Series, csv_predictions: pd.Series, archive_path: Path,
) -> dict:
    csv_features = pipeline.named_steps["materialfeatures"].transform(select_inputs(test_rows))
    compared, rejected = [], []
    with ZipFile(archive_path) as archive:
        cif_files = [name for name in archive.namelist() if name.lower().endswith(".cif")]
        names = {PurePosixPath(name).stem: name for name in cif_files}
        if len(names) != len(cif_files) or set(names) - set(test_rows.index):
            raise ValueError("The official test archive contains duplicate or unmatched CIF IDs.")
        for entry_id in test_rows.index.intersection(names, sort=False):
            try:
                prediction = score_cif(pipeline, archive.read(names[entry_id]))
            except CifInputError as error:
                rejected.append({"id": entry_id, "error": str(error)})
                continue
            csv_meta = test_rows.loc[entry_id]
            cif_meta = prediction.cif.materials.iloc[0]
            csv_group, cif_group = int(csv_meta["Space group #"]), int(cif_meta["Space group #"])
            comparison = compare_features(csv_features.loc[entry_id], prediction.features.iloc[0])
            compared.append({
                "id": entry_id,
                "archive_member": names[entry_id],
                "official_close_match_flag": csv_meta["close match"].casefold() == "yes",
                "label": int(test_y.loc[entry_id]),
                "csv_prediction": int(csv_predictions.loc[entry_id]),
                "cif_prediction": prediction.result,
                "csv_unit_cell_composition": csv_meta["True Composition"],
                "cif_unit_cell_element_counts": cif_meta["True Composition"],
                "space_group_number": [csv_group, cif_group],
                "space_group_number_mismatch": csv_group != cif_group,
                **comparison,
            })

    paired_ids = [row["id"] for row in compared]
    if not paired_ids:
        raise ValueError("No archived test CIFs could be scored.")
    labels = test_y.loc[paired_ids]
    cif_predictions = np.asarray([row["cif_prediction"]["fast_conductor"] for row in compared])
    paired_csv_predictions = csv_predictions.loc[paired_ids].to_numpy()
    comparison_fields = (
        "composition_mismatch",
        "cell_volume_per_atom_mismatch",
        "crystal_system_mismatch",
        "space_group_number_mismatch",
        "element_count_mismatch",
        "changed_weighted_summaries",
    )
    return {
        "archived_test_cifs": len(cif_files),
        "evaluated": class_counts(labels),
        "rejected": rejected,
        "cif_metrics": score(labels, cif_predictions),
        "csv_metrics_on_same_rows": score(labels, paired_csv_predictions),
        "prediction_disagreements": int((cif_predictions != paired_csv_predictions).sum()),
        "test_cifs_with_partial_occupancies": sum(row["cif_prediction"]["has_partial_occupancies"] for row in compared),
        "test_cifs_with_parser_warnings": sum(bool(row["cif_prediction"]["cif_parser_warnings"]) for row in compared),
        "test_cifs_outside_coverage": sum(row["cif_prediction"]["training_coverage"]["warning"] for row in compared),
        "comparison_tolerances": {
            "max_absolute_atomic_fraction_change": FRACTION_TOLERANCE,
            "flag_any_element_added_or_removed": True,
            "relative_cell_volume_per_atom_change": VOLUME_RELATIVE_TOLERANCE,
            "weighted_summary_relative": SUMMARY_RELATIVE_TOLERANCE,
            "weighted_summary_absolute": SUMMARY_ABSOLUTE_TOLERANCE,
            "discrete_features": "exact",
            "paired_feature_value_order": ["csv", "cif"],
        },
        "mismatch_counts": {field: sum(bool(row[field]) for row in compared) for field in comparison_fields},
        "rows_with_any_model_feature_mismatch": sum(
            any(row[field] for field in comparison_fields if field != "space_group_number_mismatch")
            for row in compared
        ),
        "rows": compared,
    }


def overlap_groups(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    train_keys, test_keys = (rows["True Composition"].map(composition_key) for rows in (train_rows, test_rows))
    shared = sorted(set(train_keys) & set(test_keys))
    groups = []
    fields = ["ID", "Reduced Composition", "True Composition", TARGET]
    for key in shared:
        groups.append({
            "normalized_atomic_fractions": dict(key),
            "train": train_rows.loc[train_keys.map(lambda item: item == key), fields].to_dict("records"),
            "test": test_rows.loc[test_keys.map(lambda item: item == key), fields].to_dict("records"),
        })
    return groups, test_keys.isin(shared)


@threadpool_limits.wrap(limits=1)
def verify_uploads(data_dir: Path, artifact_path: Path) -> dict:
    train_rows, test_rows = (read_split(data_dir, split).set_index("ID", drop=False) for split in ("train", "test"))
    train_X, train_y = select_inputs(train_rows), split_labels(train_rows)
    test_X, test_y = select_inputs(test_rows), split_labels(test_rows)
    pipeline = joblib.load(artifact_path)
    selected = type(pipeline.steps[-1][1]).__name__
    comparators = candidates()
    if selected not in comparators or not hasattr(pipeline, "training_coverage_"):
        raise ValueError("Use a model artifact produced by the current train command.")
    comparators[MAJORITY_BASELINE] = DummyClassifier(strategy="most_frequent")
    coverage = pipeline.training_coverage_
    groups, overlapping = overlap_groups(train_rows, test_rows)
    full_csv, filtered_csv, predictions_by_model = {}, {}, {}
    for name, estimator in comparators.items():
        fitted = pipeline if name == selected else estimator.fit(train_X, train_y)
        predictions = pd.Series(fitted.predict(test_X), index=test_X.index)
        predictions_by_model[name] = predictions
        full_csv[name] = score(test_y, predictions)
        filtered_csv[name] = score(test_y[~overlapping], predictions[~overlapping])

    return {
        "obelix_revision": REVISION,
        "saved_model": selected,
        "saved_artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "training_coverage_profile": {
            "distinct_compositions": len(coverage.compositions),
            "composition_distance_threshold": coverage.distance_threshold,
            "cell_volume_per_atom_range_angstrom3": coverage.volume_range,
            "crystal_systems": [system.removeprefix("crystal_system_") for system in coverage.crystal_systems],
        },
        "confusion_matrix_class_order": [0, 1],
        "cif_verification": verify_archived_cifs(
            pipeline, test_rows, test_y, predictions_by_model[selected], official_path(data_dir, "test_cifs.zip"),
        ),
        "csv_overlap_sensitivity": {
            "grouping": "True Composition atomic fractions rounded to eight decimal places",
            "fit_policy": "Used the saved model and fit the other candidates and baseline on the official train set.",
            "overlap_groups": groups,
            "overlapping_test_predictions": {
                name: predictions[overlapping].to_dict() for name, predictions in predictions_by_model.items()
            },
            "official_test": {"counts": class_counts(test_y), "metrics": full_csv},
            "test_without_overlap": {"counts": class_counts(test_y[~overlapping]), "metrics": filtered_csv},
        },
    }
