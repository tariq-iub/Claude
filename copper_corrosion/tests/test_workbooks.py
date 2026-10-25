import pytest

from cucorr.config import resolve
from cucorr.data.workbooks import audit, load_manifest, load_settings, parse_rgb, prototypes
from cucorr.schema import CLASS_NAMES, class_id


def test_parse_rgb():
    assert parse_rgb("148,106,53\n") == (148, 106, 53)
    assert parse_rgb(" 1, 2 ,3 ") == (1, 2, 3)
    assert parse_rgb(None) is None and parse_rgb("") is None and parse_rgb(float("nan")) is None
    with pytest.raises(ValueError):
        parse_rgb("1,2")
    with pytest.raises(ValueError):
        parse_rgb("1,2,300")


def test_class_mapping():
    assert class_id("cucl2") == 4
    with pytest.raises(KeyError):
        class_id("Cu(OH)2")


def test_real_workbooks(cfg):
    s = load_settings(resolve(cfg, "settings_xlsx"))
    assert [r.name for r in s] == list(CLASS_NAMES)
    assert all(len(r.all_colors) == 4 for r in s)
    m = load_manifest(resolve(cfg, "labels_xlsx"))
    assert len(m) == 660 and m["class_id"].between(0, 5).all()
    lab, rgb, ids = prototypes(s, m)
    assert lab.shape == (684, 3) and set(ids) == set(range(6))
    rep = audit(resolve(cfg, "settings_xlsx"), resolve(cfg, "labels_xlsx"))
    assert rep["class_names_compatible"] and rep["manifest"]["cluster_id_duplicates"] == 0
