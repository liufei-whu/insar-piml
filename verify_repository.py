from pathlib import Path
import csv,subprocess,sys
R=Path(__file__).resolve().parent
O=Path.home()/"Documents"/"PIML"
C=O/"results"/"seismic_ml"/"global_adaptive_fov_single_np_360_realistic_xy5_no_depth_50k"/"best_model.pt"
F=R/"results"/"summary"/"real_baseline.csv"
subprocess.run([sys.executable,str(R/"src"/"evaluate_real.py"),"--event","all","--model","baseline","--checkpoint",str(C),"--data-root",str(O),"--output",str(F)],check=True)
W={("jishishan","NP1"):12.8610,("jishishan","NP2"):13.9028,("albania","NP1"):12.1636,("albania","NP2"):11.0389,("morocco","NP1"):38.8018,("morocco","NP2"):44.6525}
with F.open(newline="",encoding="utf-8") as f:
    for r in csv.DictReader(f):
        k=(r["event"],r["plane"]); assert abs(float(r["equal_rmse_mm"])-W[k])<=5e-4,(k,r["equal_rmse_mm"],W[k])
print("[PASS] all six frozen baseline real-event RMSEs reproduced")
