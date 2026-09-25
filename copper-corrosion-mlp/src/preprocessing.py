"""Train-only feature standardization. Never fit on validation/test data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScalerTrainOnly:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    constant_mask_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "StandardScalerTrainOnly":
        self.mean_ = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        self.constant_mask_ = std < 1e-8
        std = np.where(self.constant_mask_, 1.0, std)  # avoid /0; constant features -> 0 after transform
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("scaler not fit")
        z = (X - self.mean_) / self.std_
        z[:, self.constant_mask_] = 0.0
        return z

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "constant_mask": self.constant_mask_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScalerTrainOnly":
        return cls(
            mean_=np.array(d["mean"]),
            std_=np.array(d["std"]),
            constant_mask_=np.array(d["constant_mask"], dtype=bool),
        )
