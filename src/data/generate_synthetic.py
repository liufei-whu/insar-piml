#!/usr/bin/env python3
"""Generate full-360 single-NP global adaptive-FOV clean 9-D dataset."""
from pathlib import Path
import json, sys, numpy as np, torch
SCRIPT_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=SCRIPT_DIR.parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from physics.okada import okada_forward

OUT_DIR=PROJECT_ROOT/"data"/"clean_9d"/"global_adaptive_fov_single_np_360"
OUT_DIR.mkdir(parents=True,exist_ok=True)
OUT_FILE=OUT_DIR/"dataset_9d_adaptive_fov_single_np_360.npz"
N_TRAIN,N_VAL,N_TEST=50000,5000,1000
GRID_N=64; MU=30e9
LOS_ASC=np.array([-0.558875709446653,-0.12902662410396473,0.8191520442889918],dtype=np.float64)
LOS_DES=np.array([0.5588757094466531,-0.1290266241039646,0.8191520442889918],dtype=np.float64)
BOUNDS={"x0":(-30.,30.),"y0":(-30.,30.),"depth":(5.,40.),"slip":(.1,3.5),
        "length":(10.,45.),"width":(5.,25.),"dip":(30.,80.),"strike":(0.,360.),"rake":(45.,135.)}
PARAM_NAMES=list(BOUNDS)
SIGMA_X,SIGMA_Y,SIGMA_DEPTH,SIGMA_MW=2.,2.,4.,.10
SIGMA_STRIKE,SIGMA_DIP,SIGMA_RAKE=8.,8.,12.
SEEDS={"train":85001,"val":85002,"test":85003}

def moment_magnitude(slip,length,width):
    m0=MU*(length*1000.)*(width*1000.)*slip
    return (2./3.)*(np.log10(m0)-9.1)

def choose_half_width(mw_s,depth_s):
    return 40. if (mw_s<=6.8 and depth_s<=20.) else 60.

def sample_valid_source(rng):
    while True:
        p={k:rng.uniform(lo,hi) for k,(lo,hi) in BOUNDS.items()}
        top=p["depth"]-.5*p["width"]*np.sin(np.deg2rad(p["dip"]))
        if top>.05: return p

def make_seismic_prior(p,rng):
    mw=moment_magnitude(p["slip"],p["length"],p["width"])
    s=np.array([p["x0"]+rng.normal(0,SIGMA_X),p["y0"]+rng.normal(0,SIGMA_Y),
        p["depth"]+rng.normal(0,SIGMA_DEPTH),mw+rng.normal(0,SIGMA_MW),
        (p["strike"]+rng.normal(0,SIGMA_STRIKE))%360.,
        np.clip(p["dip"]+rng.normal(0,SIGMA_DIP),0.,90.),
        p["rake"]+rng.normal(0,SIGMA_RAKE)],dtype=np.float64)
    return s,mw

def generate_split(n,seed,label):
    rng=np.random.default_rng(seed)
    X=np.empty((n,2,GRID_N,GRID_N),np.float32); y=np.empty((n,9),np.float32)
    seismic=np.empty((n,8),np.float32); hs=np.empty(n,np.float32); mws=np.empty(n,np.float32)
    for i in range(n):
        p=sample_valid_source(rng); s7,mw=make_seismic_prior(p,rng); h=choose_half_width(s7[3],s7[2])
        axis=torch.linspace(-h,h,GRID_N,dtype=torch.float64)
        yg,xg=torch.meshgrid(axis,axis,indexing="ij")
        kw=dict(x0=p["x0"],y0=p["y0"],depth=p["depth"],slip=p["slip"],x_grid=xg,y_grid=yg,
                length=p["length"],width=p["width"],dip=p["dip"],strike=p["strike"],rake=p["rake"])
        X[i,0]=okada_forward(**kw,los_vector=LOS_ASC).detach().cpu().numpy().astype(np.float32)
        X[i,1]=okada_forward(**kw,los_vector=LOS_DES).detach().cpu().numpy().astype(np.float32)
        y[i]=np.array([p[k] for k in PARAM_NAMES],np.float32)
        seismic[i,:7]=s7.astype(np.float32); seismic[i,7]=h; hs[i]=h; mws[i]=mw
        if (i+1)%1000==0 or i+1==n: print(f"{label}: {i+1}/{n}",flush=True)
    print(f"{label}: H40={np.mean(hs==40):.3f}, H60={np.mean(hs==60):.3f}, Mw={mws.min():.3f}..{mws.max():.3f}")
    return X,y,seismic,hs,mws

def main():
    print("="*78); print("FULL-360 SINGLE-NP GLOBAL ADAPTIVE-FOV CLEAN DATASET"); print("="*78)
    print("Only one seismic plane is supplied: the noisy generating-fault plane.")
    print("Strike source range: [0,360); seismic strike noise wraps modulo 360.")
    arrays={}
    for split,n in [("train",N_TRAIN),("val",N_VAL),("test",N_TEST)]:
        X,y,S,H,Mw=generate_split(n,SEEDS[split],split)
        arrays[f"X_{split}"]=X; arrays[f"y_{split}"]=y; arrays[f"seismic_{split}"]=S
        arrays[f"H_{split}"]=H; arrays[f"Mw_true_{split}"]=Mw
    meta={"description":"Full-360 single-NP global adaptive-FOV clean Okada dataset",
          "grid_n":GRID_N,"fov_rule":"H=40 km if Mw_s<=6.8 and depth_s<=20 km, otherwise H=60 km",
          "seismic_vector":["x_s","y_s","depth_s","Mw_s","strike_s","dip_s","rake_s","H_km"],
          "seismic_sigmas":{"x_km":SIGMA_X,"y_km":SIGMA_Y,"depth_km":SIGMA_DEPTH,"Mw":SIGMA_MW,
                             "strike_deg":SIGMA_STRIKE,"dip_deg":SIGMA_DIP,"rake_deg":SIGMA_RAKE},
          "bounds":{k:list(v) for k,v in BOUNDS.items()},"parameter_names":PARAM_NAMES,"seeds":SEEDS,
          "los_asc":LOS_ASC.tolist(),"los_des":LOS_DES.tolist(),
          "note":"One noisy seismic plane only; source strike spans 0..360 and seismic strike is circularly wrapped."}
    arrays["metadata_json"]=np.array(json.dumps(meta))
    np.savez_compressed(OUT_FILE,**arrays); print("\nSaved:",OUT_FILE)
if __name__=="__main__": main()
