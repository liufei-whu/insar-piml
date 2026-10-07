"""Frozen 50k pure-ML baseline experiment."""
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/"src"/"train_baseline.py"),run_name="__main__")
