"""Compare CIF and CSV training features on the same archived cohort."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .data import REVISION, THRESHOLD, official_path, read_split, split_labels
from .features import CifInputError, composition_key, select_inputs
from .models import N_FOLDS, SEED, candidates, class_counts, score, score_cif, write_report

REPORT_PATH = Path("reports/cif-model-comparison.json")
PROTECTED_PATHS = (
    Path("artifacts/fast_conductor.joblib"),
    Path("reports/benchmark.json"),
    Path("reports/upload-verification.json"),
)

# Other estimator settings match the existing candidate definitions.
GRID = {
    "LogisticRegression": {"logisticregression__C": [0.1, 1.0, 10.0]},
    "RandomForestClassifier": {"randomforestclassifier__min_samples_leaf": [1, 2, 4]},
    "HistGradientBoostingClassifier": {"histgradientboostingclassifier__max_leaf_nodes": [7, 15, 31]},
}
CV_METRICS = {
    "fast_f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    "fast_f1": "f1",
    "fast_recall": "recall",
    "fast_precision": "precision",
    "accuracy": "accuracy",
}


@dataclass
class ArchivedCohort:
    rows: pd.DataFrame
    labels: pd.Series
    features: pd.DataFrame
    cif_compositions: list
    deployed_predictions: pd.DataFrame
    rejected: dict


def load_archived_cohort(data_dir: Path, split: str, deployed, expected_rows: int) -> ArchivedCohort:
    rows = read_split(data_dir, split).set_index("ID", drop=False)
    ids, features, compositions, deployed_predictions = [], [], [], []
    rejected = {}
    with ZipFile(official_path(data_dir, f"{split}_cifs.zip")) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".cif")]
        members = {PurePosixPath(name).stem: name for name in names}
        if len(members) != len(names) or set(members) - set(rows.index):
            raise ValueError(f"The official {split} archive has duplicate or unpaired CIF IDs.")
        for entry_id in rows.index.intersection(members, sort=False):
            try:
                prediction = score_cif(deployed, archive.read(members[entry_id]))
            except CifInputError as error:
                rejected[entry_id] = str(error)
                continue
            ids.append(entry_id)
            features.append(prediction.features)
            compositions.append(composition_key(prediction.cif.materials.iloc[0]["True Composition"]))
            deployed_predictions.append(prediction.result)
    if len(ids) != expected_rows:
        raise ValueError(f"Expected {expected_rows} accepted {split} CIFs, got {len(ids)}.")
    accepted = rows.loc[ids]
    return ArchivedCohort(
        accepted, split_labels(accepted), pd.concat(features).set_axis(ids), compositions,
        pd.DataFrame(deployed_predictions, index=ids), rejected,
    )


def composition_groups(csv_keys: list, cif_keys: list) -> np.ndarray:
    """Keep rows connected by any exact normalized composition in the same fold."""
    parent = list(range(len(csv_keys)))
    first_row_for_key = {}

    def root(row: int) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    for index, keys in enumerate(zip(csv_keys, cif_keys, strict=True)):
        for key in keys:
            other = first_row_for_key.setdefault(key, index)
            parent[root(index)] = root(other)
    return np.asarray([root(index) for index in range(len(parent))])


def select_cif_model(features: pd.DataFrame, labels: pd.Series, folds: list):
    estimators = {name: pipeline[1:] for name, pipeline in candidates().items()}
    best_by_family = {}
    results = {}
    for name, estimator in estimators.items():
        final_step = estimator.steps[-1][0]
        parameter_grid = GRID[name] | {f"{final_step}__class_weight": [None, "balanced"]}
        search = GridSearchCV(
            estimator, parameter_grid, scoring=CV_METRICS, cv=folds, refit=False,
            error_score="raise", n_jobs=1,
        )
        search.fit(features, labels)
        trials = []
        for index, settings in enumerate(search.cv_results_["params"]):
            trials.append({
                "settings": settings,
                **{f"mean_{metric}": float(search.cv_results_[f"mean_test_{metric}"][index]) for metric in CV_METRICS},
                "fold_fast_f2": [
                    float(search.cv_results_[f"split{fold}_test_fast_f2"][index]) for fold in range(N_FOLDS)
                ],
            })
        results[name] = trials
        best_by_family[name] = max(trials, key=selection_key)
    selected = max(best_by_family, key=lambda name: selection_key(best_by_family[name]))
    chosen = estimators[selected].set_params(**best_by_family[selected]["settings"])
    return chosen, {
        "selection_rule": "Highest mean fast-class F2, then F1, then recall. Exact ties favor the first listed trial.",
        "selected_classifier": selected,
        "selected_parameters": chosen.steps[-1][1].get_params(deep=False),
        "best_by_family": best_by_family,
        "all_trials": results,
    }


def selection_key(trial: dict) -> tuple:
    return trial["mean_fast_f2"], trial["mean_fast_f1"], trial["mean_fast_recall"]


def evaluate(labels: pd.Series, predicted: pd.Series, close_match: pd.Series) -> dict:
    metrics = {}
    for group, mask in (
        ("all", pd.Series(True, index=labels.index)), ("close_match", close_match), ("unflagged", ~close_match),
    ):
        true, prediction = labels[mask], predicted[mask]
        metrics[group] = {
            **class_counts(true),
            **score(true, prediction.to_numpy()),
            "fast_f2": fbeta_score(true, prediction, beta=2, zero_division=0),
            "missed_fast_ids": true.index[true.eq(1) & prediction.eq(0)].tolist(),
            "false_positive_ids": true.index[true.eq(0) & prediction.eq(1)].tolist(),
        }
    return metrics


def protected_hashes() -> dict:
    return {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in PROTECTED_PATHS}


@threadpool_limits.wrap(limits=1)
def run_experiment(data_dir: Path = Path("data/raw")) -> dict:
    before = protected_hashes()
    deployed = joblib.load(PROTECTED_PATHS[0])
    train = load_archived_cohort(data_dir, "train", deployed, expected_rows=247)
    csv_keys = train.rows["True Composition"].map(composition_key).tolist()
    groups = composition_groups(csv_keys, train.cif_compositions)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(train.features, train.labels, groups))
    fold_counts = []
    for fit, validation in folds:
        if set(groups[fit]) & set(groups[validation]) or train.labels.iloc[validation].nunique() != 2:
            raise ValueError("Expected disjoint composition groups and both labels in each validation fold.")
        fold_counts.append(class_counts(train.labels.iloc[validation]))

    selected, selection = select_cif_model(train.features, train.labels, folds)
    cif_model = clone(selected).fit(train.features, train.labels)
    csv_features = deployed.named_steps["materialfeatures"].transform(select_inputs(train.rows))
    csv_control = clone(selected).fit(csv_features, train.labels)
    print(f"Training-only selection: {selection['selected_classifier']}", flush=True)

    # Read the test archive and its labels only once the two experimental fits are fixed.
    test = load_archived_cohort(data_dir, "test", deployed, expected_rows=67)
    close_match = test.rows["close match"].str.casefold().eq("yes")
    predictions = {}
    probabilities = {}
    for name, model in (("cif_trained", cif_model), ("matched_csv_control", csv_control)):
        predictions[name] = pd.Series(model.predict(test.features), index=test.rows.index)
        probabilities[name] = pd.Series(model.predict_proba(test.features)[:, 1], index=test.rows.index)
    predictions["saved_full_csv"] = test.deployed_predictions["fast_conductor"]
    probabilities["saved_full_csv"] = test.deployed_predictions["fast_conductor_probability"]
    metrics = {name: evaluate(test.labels, predictions[name], close_match)
               for name in ("saved_full_csv", "cif_trained", "matched_csv_control")}

    verified = json.loads(PROTECTED_PATHS[2].read_text(encoding="utf-8"))
    expected = verified["cif_verification"]["cif_metrics"]
    if any(metrics["saved_full_csv"]["all"][metric] != value for metric, value in expected.items()):
        raise ValueError("The deployed model did not reproduce the verified official CIF benchmark.")
    if protected_hashes() != before:
        raise RuntimeError("A deployed artifact or benchmark changed during the experiment.")

    train_close = train.rows["close match"].str.casefold().eq("yes")
    return {
        "obelix_revision": REVISION,
        "conductivity_threshold_s_per_cm": float(THRESHOLD),
        "protected_sha256": before,
        "train_counts": class_counts(train.labels),
        "train_close_match_counts": class_counts(train.labels[train_close]),
        "train_rejected_cifs": train.rejected,
        "test_counts": class_counts(test.labels),
        "cross_validation": {
            "folds": N_FOLDS, "seed": SEED, "distinct_groups": int(np.unique(groups).size),
            "grouping": "Union of equal normalized CIF or CSV elemental fractions rounded to eight decimals.",
            "validation_fold_counts": fold_counts,
            **selection,
        },
        "csv_control_policy": "Same rows, labels, selected classifier, and settings as the CIF-trained model.",
        "class_prediction_policy": "Standard classifier predict rule for every model.",
        "confusion_matrix_class_order": [0, 1],
        "test_metrics": metrics,
        "test_predictions": [
            {
                "id": entry_id, "fast_label": int(test.labels.loc[entry_id]),
                "close_match": bool(close_match.loc[entry_id]),
                **{
                    name: {"class": int(prediction.loc[entry_id]), "fast_probability": float(probabilities[name].loc[entry_id])}
                    for name, prediction in predictions.items()
                },
            }
            for entry_id in test.rows.index
        ],
    }


if __name__ == "__main__":
    write_report(REPORT_PATH, run_experiment())
    print(f"Wrote {REPORT_PATH}")
