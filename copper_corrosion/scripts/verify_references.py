"""Resolve every DOI in docs/references.md via doi.org (HEAD request). Network required."""
import re
import sys
import urllib.request
from pathlib import Path

text = (Path(__file__).resolve().parents[1] / "docs" / "references.md").read_text()
dois = sorted({d.rstrip(".,") for d in re.findall(r"doi:(10\.[^\s|]+)", text)})
bad = 0
for d in dois:
    try:
        req = urllib.request.Request(f"https://doi.org/{d}", method="HEAD")
        code = urllib.request.urlopen(req, timeout=20).status
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "code", str(e))
    ok = code in (200, 301, 302, 403)  # some publishers refuse HEAD after redirect
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {code} {d}")
sys.exit(1 if bad else 0)
