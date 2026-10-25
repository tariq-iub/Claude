"""Thin wrapper: equivalent to `cucorr predict ...`. See README for arguments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cucorr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
