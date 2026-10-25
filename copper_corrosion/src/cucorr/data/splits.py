"""Group-aware (specimen-level) splitting to prevent leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_split(image_ids, group_ids, fractions=(0.6, 0.2, 0.2), seed: int = 0) -> pd.DataFrame:
    """Assign whole groups to train/val/test. All images of a specimen share a split."""
    df = pd.DataFrame({"image_id": list(image_ids), "group_id": list(group_ids)})
    groups = df["group_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_tr = max(1, int(round(fractions[0] * n)))
    n_va = max(1, int(round(fractions[1] * n))) if n >= 3 else 0
    split = {g: "train" for g in groups[:n_tr]}
    split.update({g: "val" for g in groups[n_tr:n_tr + n_va]})
    split.update({g: "test" for g in groups[n_tr + n_va:]})
    df["split"] = df["group_id"].map(split)
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    per = df.groupby("group_id")["split"].nunique()
    bad = per[per > 1]
    if len(bad):
        raise ValueError(f"groups spanning several splits (leakage): {list(bad.index)}")
