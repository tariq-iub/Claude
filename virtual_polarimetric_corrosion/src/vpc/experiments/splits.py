"""Group-level (cartridge / acquisition-session) stratified splitting and leakage checks. Never split pixels or views."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def group_stratified_split(groups: Sequence, strata: Sequence, ratios=(0.70, 0.15, 0.15), seed: int = 0) -> Dict[str, np.ndarray]:
    """Assign each *group* to train/val/test, stratifying on one label per group (e.g. severity bin of the cartridge).
    ``groups`` and ``strata`` are per-sample; a group's stratum is its mode. Returns dict split -> array of sample indices."""
    groups = np.asarray(groups)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    g_stratum = {}
    for g in ug:
        v, c = np.unique(strata[groups == g], return_counts=True)
        g_stratum[g] = v[c.argmax()]
    assign: Dict = {}
    names = ("train", "val", "test")
    carry = {"val": 0.0, "test": 0.0}   # carry rounding remainders across strata so global proportions stay on target
    for s_ in np.unique(list(g_stratum.values())):
        gs = np.array([g for g in ug if g_stratum[g] == s_])
        rng.shuffle(gs)
        n = len(gs)
        take = {}
        for name, r in (("test", ratios[2]), ("val", ratios[1])):
            carry[name] += r * n
            take[name] = int(np.floor(carry[name] + 0.5))
            carry[name] -= take[name]
        take["test"], take["val"] = min(take["test"], n), min(take["val"], n - min(take["test"], n))
        for i, g in enumerate(gs):
            assign[g] = "test" if i < take["test"] else ("val" if i < take["test"] + take["val"] else "train")
    out = {k: np.where(np.array([assign[g] == k for g in groups]))[0] for k in names}
    assert_no_group_leakage({k: groups[v] for k, v in out.items()})
    return out


def assert_no_group_leakage(split_groups: Dict[str, Iterable]) -> None:
    sets = {k: set(np.asarray(list(v)).tolist()) for k, v in split_groups.items()}
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = sets[keys[i]] & sets[keys[j]]
            if inter:
                raise AssertionError(f"group leakage between {keys[i]} and {keys[j]}: {sorted(inter)[:5]}")


def near_duplicate_check(features: np.ndarray, split_index: Dict[str, np.ndarray], thresh: float = 0.995) -> Dict[str, float]:
    """Guards against *cross-group* near-duplicates (e.g. same cartridge photographed in two sessions under different IDs):
    max cosine similarity between each test sample and the train set; flag if above ``thresh``."""
    f = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-12)
    sim = f[split_index["test"]] @ f[split_index["train"]].T
    mx = sim.max(1)
    return {"max_sim": float(mx.max()), "n_flagged": int((mx > thresh).sum())}
