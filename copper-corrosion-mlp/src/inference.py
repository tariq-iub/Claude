"""Lightweight CPU inference pipeline: load the saved artifacts once, then
classify single Lab vectors or batches, with calibrated probabilities and
uncertainty abstention. No re-fitting of any preprocessing at inference
time (task brief §16, "never estimate new normalization parameters from
an inference image").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import build_model
from .preprocessing import StandardScalerTrainOnly
from .features import build_features
from .calibration import apply_temperature


class CorrosionMLPPredictor:
    def __init__(self, models_dir: str):
        models_dir = Path(models_dir)
        cfg = json.load(open(models_dir / "preprocessing_and_config.json"))
        self.cfg = cfg
        self.class_order = cfg["class_order"]
        self.feature_set = cfg["feature_set"]
        self.scaler = StandardScalerTrainOnly.from_dict(cfg["scaler"])
        self.temperature = cfg["temperature"]
        self.tau = cfg["abstention_threshold"]
        self.prototypes = np.array(cfg["prototypes"]) if cfg["prototypes"] is not None else None

        in_dim = len(cfg["feature_names"])
        model_cfg = {"kind": cfg["model_kind"], "hidden_dims": cfg["hidden_dims"],
                     "activation": cfg["activation"], "dropout": cfg["dropout"], "batchnorm": cfg["batchnorm"]}
        self.model = build_model(in_dim, len(self.class_order), model_cfg)
        state = torch.load(models_dir / "best_model.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_lab(self, lab: np.ndarray) -> dict:
        """lab: (N,3) CIELAB array. Returns calibrated probabilities, the
        argmax class, and an uncertainty-aware label ('Uncertain' below tau)."""
        lab = np.atleast_2d(lab).astype(np.float64)
        X, _ = build_features(lab, self.feature_set, self.prototypes)
        Xs = self.scaler.transform(X)
        with torch.no_grad():
            logits = self.model(torch.tensor(Xs, dtype=torch.float32)).numpy()
        proba = apply_temperature(logits, self.temperature)
        conf = proba.max(axis=1)
        pred_idx = proba.argmax(axis=1)
        labels = [self.class_order[i] if c >= self.tau else "Uncertain" for i, c in zip(pred_idx, conf)]
        return {"probabilities": proba, "predicted_class": labels, "confidence": conf,
                "class_order": self.class_order}
