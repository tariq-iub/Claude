"""CPU deployment validation: dynamic INT8 quantization vs. the FP32
model, compared on the held-out test set (task brief §22). Uses PyTorch's
built-in dynamic quantization (torch.ao.quantization.quantize_dynamic) -
appropriate for a Linear-dominated MLP - rather than a full ONNX Runtime
pipeline, since no ONNX Runtime is available in this environment; the
quantized module is still evaluated end-to-end for accuracy parity before
being accepted, exactly as the task brief requires.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_labels, CANONICAL_CLASSES
from src.features import build_features
from src.inference import CorrosionMLPPredictor
from src.evaluator import compute_metrics


def run(cfg_path=str(ROOT / "configs/mlp_config.yaml")):
    cfg = yaml.safe_load(open(cfg_path))
    predictor = CorrosionMLPPredictor(str(ROOT / "models"))
    ds = load_labels(str(ROOT / cfg["data"]["labels_path"]), cfg["data"]["invalid_row_policy"])

    from src.splits import train_val_test_split
    y = ds.label_ids
    _, _, test_idx = train_val_test_split(y, seed=cfg["seeds_for_repeat"][0],
                                           train_frac=cfg["validation"]["train_frac"],
                                           val_frac=cfg["validation"]["val_frac"])
    lab_test = ds.lab[test_idx]
    y_test = y[test_idx]
    n_classes = len(CANONICAL_CLASSES)

    fp32_out = predictor.predict_lab(lab_test)
    fp32_pred = fp32_out["probabilities"].argmax(axis=1)
    fp32_metrics = compute_metrics(y_test, fp32_pred, n_classes, CANONICAL_CLASSES)

    q_model = torch.ao.quantization.quantize_dynamic(predictor.model, {torch.nn.Linear}, dtype=torch.qint8)

    X, _ = build_features(lab_test, predictor.feature_set, predictor.prototypes)
    Xs = predictor.scaler.transform(X)
    with torch.no_grad():
        q_logits = q_model(torch.tensor(Xs, dtype=torch.float32)).numpy()
    from src.calibration import apply_temperature
    q_proba = apply_temperature(q_logits, predictor.temperature)
    q_pred = q_proba.argmax(axis=1)
    q_metrics = compute_metrics(y_test, q_pred, n_classes, CANONICAL_CLASSES)

    agreement = float((q_pred == fp32_pred).mean())
    mean_abs_prob_diff = float(np.mean(np.abs(q_proba - fp32_out["probabilities"])))
    macro_f1_drop = fp32_metrics["macro_f1"] - q_metrics["macro_f1"]

    import io
    buf_fp32, buf_int8 = io.BytesIO(), io.BytesIO()
    torch.save(predictor.model.state_dict(), buf_fp32)
    torch.save(q_model.state_dict(), buf_int8)

    def latency(m, n=50):
        x = torch.randn(1, Xs.shape[1])
        with torch.no_grad():
            for _ in range(5):
                m(x)
            ts = []
            for _ in range(n):
                t0 = time.perf_counter(); m(x); ts.append(time.perf_counter() - t0)
        return float(np.mean(ts) * 1000)

    result = {
        "fp32_macro_f1": fp32_metrics["macro_f1"], "int8_macro_f1": q_metrics["macro_f1"],
        "macro_f1_drop": macro_f1_drop, "top1_agreement": agreement, "mean_abs_prob_diff": mean_abs_prob_diff,
        "fp32_size_bytes": buf_fp32.getbuffer().nbytes, "int8_size_bytes": buf_int8.getbuffer().nbytes,
        "fp32_latency_ms_bs1": latency(predictor.model), "int8_latency_ms_bs1": latency(q_model),
        "max_macro_f1_drop_tolerance": 0.01,
        "accepted_for_deployment": macro_f1_drop <= 0.01,
    }
    print(json.dumps(result, indent=2))
    json.dump(result, open(ROOT / "results" / "quantization_comparison.json", "w"), indent=2)
    return result


if __name__ == "__main__":
    run()
