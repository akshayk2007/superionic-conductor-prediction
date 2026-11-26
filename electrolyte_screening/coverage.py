"""Describe the training feature coverage used to flag unfamiliar inputs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .features import ELEMENTS, FRACTION_FEATURES, SYSTEM_FEATURES, VOLUME_FEATURE


@dataclass
class TrainingCoverage:
    compositions: np.ndarray
    distance_threshold: float
    volume_range: tuple[float, float]
    crystal_systems: tuple[str, ...]

    @classmethod
    def fit(cls, features: pd.DataFrame) -> "TrainingCoverage":
        fractions = features.loc[:, list(FRACTION_FEATURES)].to_numpy()
        compositions = np.unique(fractions.round(8), axis=0)
        threshold = 0.0
        if len(compositions) > 1:
            distances = cdist(compositions, compositions, metric="cityblock") / 2
            np.fill_diagonal(distances, np.inf)
            threshold = float(np.quantile(distances.min(axis=1), 0.95))
        volumes = features[VOLUME_FEATURE]
        systems = tuple(name for name in SYSTEM_FEATURES if features[name].any())
        return cls(compositions, threshold, (float(volumes.min()), float(volumes.max())), systems)

    def check(self, features: pd.Series) -> dict:
        fractions = features.loc[list(FRACTION_FEATURES)].to_numpy()
        distance = float(cdist([fractions.round(8)], self.compositions, metric="cityblock").min() / 2)
        unseen = (fractions > 0) & ~self.compositions.any(axis=0)
        volume = float(features[VOLUME_FEATURE])
        crystal_system = features.loc[list(SYSTEM_FEATURES)].idxmax()
        reasons = []
        if unseen.any():
            reasons.append({
                "code": "unseen_elements",
                "elements": [element.symbol for element, absent in zip(ELEMENTS, unseen) if absent],
            })
        if distance > self.distance_threshold:
            reasons.append({"code": "distant_composition"})
        if not self.volume_range[0] <= volume <= self.volume_range[1]:
            reasons.append({
                "code": "volume_outside_training_range",
                "volume_per_atom_angstrom3": volume,
                "training_range_angstrom3": self.volume_range,
            })
        if crystal_system not in self.crystal_systems:
            reasons.append({
                "code": "unseen_crystal_system",
                "crystal_system": crystal_system.removeprefix("crystal_system_"),
            })
        return {
            "warning": bool(reasons),
            "reasons": reasons,
            "nearest_composition_distance": distance,
            "composition_distance_threshold": self.distance_threshold,
        }
