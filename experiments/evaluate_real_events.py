"""Evaluate a selected frozen model on all three real events."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
sys.argv=[str(ROOT/"src"/"evaluate_real.py"),"--event","all"]+sys.argv[1:]
runpy.run_path(str(ROOT/"src"/"evaluate_real.py"),run_name="__main__")
