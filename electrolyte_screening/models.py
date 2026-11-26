"""Select using the training split and score once on the official test split."""

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .coverage import TrainingCoverage
from .data import REVISION, THRESHOLD, read_split, split_labels
from .features import CifInputError, CifInputs, MaterialFeatures, composition_key, inputs_from_cif, select_inputs

SEED = 42
N_FOLDS = 5
MAJORITY_BASELINE = "Majority (most frequent)"


@dataclass
class CifPrediction:
    cif: CifInputs
    features: pd.DataFrame
    result: dict


def candidates() -> dict:
    return {
        "LogisticRegression": make_pipeline(
            MaterialFeatures(), StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
        ),
        "RandomForestClassifier": make_pipeline(
            MaterialFeatures(),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=1),
        ),
        "HistGradientBoostingClassifier": make_pipeline(
            MaterialFeatures(),
            HistGradientBoostingClassifier(
                max_iter=150, max_leaf_nodes=15, min_samples_leaf=12, learning_rate=0.06,
                l2_regularization=1, early_stopping=False, random_state=SEED,
            ),
        ),
    }


def training_folds(inputs: pd.DataFrame, labels: pd.Series) -> list:
    keys = inputs["True Composition"].map(composition_key)
    groups = pd.factorize(keys, sort=False)[0]
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(splitter.split(inputs, labels, groups))


def class_counts(labels: pd.Series) -> dict:
    return {"rows": len(labels), "slow": int(labels.eq(0).sum()), "fast": int(labels.eq(1).sum())}


def score(labels: pd.Series, predictions: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


@threadpool_limits.wrap(limits=1)
def benchmark(data_dir: Path, artifact_path: Path) -> dict:
    train_rows = read_split(data_dir, "train")
    train_X, train_y = select_inputs(train_rows), split_labels(train_rows)
    folds = training_folds(train_X, train_y)
    estimators = candidates()
    cv = {}
    for name, estimator in estimators.items():
        fold_f1 = cross_val_score(estimator, train_X, train_y, cv=folds, scoring="f1", error_score="raise")
        cv[name] = {"fold_f1": fold_f1.tolist(), "mean_f1": float(fold_f1.mean())}
    # dict insertion order resolves an exact tie in favor of the earlier candidate.
    selected = max(cv, key=lambda name: cv[name]["mean_f1"])

    estimators[MAJORITY_BASELINE] = DummyClassifier(strategy="most_frequent")
    for estimator in estimators.values():
        estimator.fit(train_X, train_y)

    test_rows = read_split(data_dir, "test")
    test_X, test_y = select_inputs(test_rows), split_labels(test_rows)
    evaluation = {name: score(test_y, estimator.predict(test_X)) for name, estimator in estimators.items()}
    save_screening_model(estimators[selected], train_X, artifact_path)

    return {
        "obelix_revision": REVISION,
        "threshold_s_per_cm": float(THRESHOLD),
        "train": class_counts(train_y),
        "test": class_counts(test_y),
        "selection": {
            "metric": "mean positive-class F1",
            "folds": N_FOLDS,
            "group": "normalized True Composition",
            "seed": SEED,
            "scores": cv,
            "selected": selected,
        },
        "confusion_matrix_class_order": [0, 1],
        "test_scores": evaluation,
    }


def save_screening_model(pipeline, train_inputs: pd.DataFrame, artifact_path: Path) -> None:
    train_features = pipeline.named_steps["materialfeatures"].transform(train_inputs)
    pipeline.training_coverage_ = TrainingCoverage.fit(train_features)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=artifact_path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        joblib.dump(pipeline, temp_path, compress=3)
        temp_path.replace(artifact_path)
    finally:
        temp_path.unlink(missing_ok=True)


@threadpool_limits.wrap(limits=1)
def score_cif(pipeline, content: bytes) -> CifPrediction:
    """Score the same uploaded bytes and return the exact features used."""
    cif = inputs_from_cif(content)
    try:
        features = pipeline.named_steps["materialfeatures"].transform(cif.materials)
    except (ValueError, KeyError, ArithmeticError) as error:
        raise CifInputError("Could not extract material features from the CIF structure.") from error
    classifier = pipeline[1:]
    coverage = getattr(pipeline, "training_coverage_", None)
    result = {
        "fast_conductor": int(classifier.predict(features)[0]),
        "fast_conductor_probability": float(classifier.predict_proba(features)[0, 1]),
        "threshold_s_per_cm": float(THRESHOLD),
        "has_partial_occupancies": cif.has_partial_occupancies,
        "cif_parser_warnings": cif.parser_warnings,
        "training_coverage": coverage.check(features.iloc[0]) if coverage else {
            "warning": True,
            "reasons": [{"code": "coverage_profile_missing"}],
        },
    }
    return CifPrediction(cif, features, result)


def predict_cif(artifact_path: Path, cif_path: Path) -> dict:
    try:
        content = cif_path.read_bytes()
    except OSError as error:
        raise CifInputError(f"Could not read the CIF file at {cif_path}.") from error
    return score_cif(joblib.load(artifact_path), content).result


def write_report(path: Path | None, content: str | dict) -> None:
    output = content if isinstance(content, str) else json.dumps(content, indent=2) + "\n"
    if path is None:
        print(output, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
