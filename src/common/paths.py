"""Repo-root paths. The one place that knows how deep src/ nests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
