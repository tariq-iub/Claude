import numpy as np

from cucorr.evaluation.agreement import calculate_inter_annotator_agreement, cohen_kappa_from_cm, fleiss_kappa
from cucorr.evaluation.metrics import binary_scores, boundary_scores, evaluate_segmentation
from cucorr.evaluation.statistics import compare_approaches, holm, select_best
from cucorr.schema import UNCERTAIN


def test_perfect_and_ignored():
    ref = np.zeros((20, 20), np.uint8); ref[5:10, 5:10] = 2; ref[0, :] = UNCERTAIN
    pred = ref.copy(); pred[0, :] = 4   # predictions on ignored pixels must not matter
    m = evaluate_segmentation(ref, pred)
    assert m["per_class_iou"][0] == 1 and m["per_class_iou"][2] == 1 and np.isnan(m["per_class_iou"][4])
    assert m["binary_dice"] == 1 and m["boundary_f1"] == 1


def test_abstention_counts_as_error_not_healthy():
    ref = np.ones((10, 10), np.uint8)
    pred = np.full_like(ref, UNCERTAIN)
    m = evaluate_segmentation(ref, pred)
    assert m["abstention_rate"] == 1 and m["per_class_recall"][1] == 0
    assert binary_scores(ref, pred)["sensitivity"] == 0


def test_boundary_shift():
    ref = np.zeros((30, 30), np.uint8); ref[:, 15:] = 1
    pred = np.zeros_like(ref); pred[:, 16:] = 1
    assert boundary_scores(ref, pred, 2)["boundary_f1"] == 1
    assert boundary_scores(ref, pred, 0)["boundary_f1"] < 1


def test_kappa():
    assert cohen_kappa_from_cm(np.diag([10, 10])) == 1
    assert abs(cohen_kappa_from_cm(np.array([[25, 25], [25, 25]]))) < 1e-9


def test_iaa_and_fleiss():
    rng = np.random.default_rng(0)
    a = [rng.integers(0, 6, (20, 20)).astype(np.uint8) for _ in range(4)]
    b = [x.copy() for x in a]; b[0][:5] = (b[0][:5] + 1) % 6
    rep = calculate_inter_annotator_agreement({"r1": a, "r2": b, "r3": a}, iters=50)
    assert 0.8 < rep["pairs"]["r1|r2"]["cohen_kappa"] < 1
    assert rep["pairs"]["r1|r3"]["cohen_kappa"] == 1
    assert fleiss_kappa([a, a]) == 1


def test_stats():
    x = np.linspace(0.5, 0.9, 40); y = x - 0.1
    r = compare_approaches({"a": x, "b": y}, iters=200)
    k = r["pairwise"]["a vs b"]
    assert abs(k["mean_diff"] - 0.1) < 1e-9 and k["ci"][0] > 0 and r["power_note"] is None
    assert holm({"p": 0.01, "q": 0.04})["q"] == 0.04 and holm({"p": 0.01, "q": 0.04})["p"] == 0.02
    assert select_best({"a": {"macro_miou": 0.8}, "b": {"macro_miou": 0.5}})[0][0] == "a"
