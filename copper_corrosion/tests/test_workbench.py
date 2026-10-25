"""Headless test of the workbench app: synthetic data -> run all -> compare tabs render."""
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from cucorr.config import PROJECT_ROOT  # noqa: E402


def test_workbench_end_to_end():
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "workbench.py"), default_timeout=600)
    at.run()
    assert not at.exception
    next(b for b in at.button if b.label == "Generate synthetic set").click().run()
    assert not at.exception and len(at.session_state["items"]) == 24
    for key in ("B · SVM",):
        at.checkbox(key=f"sel_{key}").uncheck()
    at.toggle[1].set_value(False).run()   # robustness off for speed
    next(b for b in at.button if b.label == "⏩ Run all remaining").click().run()
    assert not at.exception, at.exception
    res = at.session_state["results"]
    assert {"A · CIEDE2000", "B · Random Forest", "C · Superpixels", "F · Hybrid"} <= set(res)
    assert all(r.error is None for r in res.values())
    assert all(r.metrics for r in res.values())
