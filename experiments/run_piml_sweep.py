"""Physics-weight sweep used in the project."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
trainer=ROOT/"src"/"train_piml.py"
for lam in (0.0,0.01,0.1,1.0):
    print(f"\n=== lambda_phys={lam} ===",flush=True)
    subprocess.run([sys.executable,str(trainer),"--lambda-phys",str(lam)],check=True)
