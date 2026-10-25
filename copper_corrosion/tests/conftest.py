import numpy as np
import pytest

from cucorr.config import load_config
from cucorr.data.synthetic import synthetic_dataset


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={"approach_b": {"n_estimators": 30, "pixels_per_image": 1500},
                                  "evaluation": {"bootstrap_iterations": 100},
                                  "approach_d": {"epochs": 2, "tile_size": 64, "mc_dropout_samples": 2,
                                                 "pretrained": False}})


@pytest.fixture(scope="session")
def dataset():
    return synthetic_dataset(8, seed=1, h=64, w=80)
