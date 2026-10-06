#!/usr/bin/env python3
"""Unified real-event evaluator for the frozen 50k XY5/no-depth architecture."""
from pathlib import Path
import argparse, csv, sys
import numpy as np
import torch
import torch.nn as nn

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from physics.okada import okada_forward

NAMES=["x0","y0","depth","slip","length","width","dip","strike","rake"]
BOUNDS=np.array([[-30,30],[-30,30],[5,40],[.1,3.5],[10,45],[5,25],[30,80],[0,360],[45,135]],np.float32)
IDX=np.array([0,1,2,3,4,5,6,8]); LO=BOUNDS[:,0]; HI=BOUNDS[:,1]; R=HI-LO

EVENTS={
"jishishan":{"data":"results/real_china_2023/preprocessing/china_2023_dual_los_64.npz",
"event":(1.805,8.896,17.6,6.1),"H":40.0,
"planes":(("NP1",167.,48.,124.),("NP2",302.,52.,58.))},
"albania":{"data":"results/real_albania_2019/preprocessing/albania_2019_dual_los_64_h60.npz",
"event":(-6.588683,-13.655938,24.1,6.4),"H":60.0,
"planes":(("NP1",145.,68.,79.),("NP2",351.,25.,114.))},
"morocco":{"data":"results/real_morocco_2023/preprocessing/morocco_2023_dual_los_64.npz",
"event":(9.344868,-14.686786,23.8,6.9),"H":60.0,
"planes":(("NP1",118.,26.,128.),("NP2",257.,70.,73.))}
}

class InversionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(4,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
            nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.Conv2d(64,128,3,2,1),nn.ReLU())
        self.event=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.plane=nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.fuse=nn.Sequential(nn.Linear(128*4*4+64,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.scalar=nn.Sequential(nn.Linear(128,8),nn.Sigmoid()); self.strike=nn.Linear(128,2)
    def forward(self,img,ev,mech):
        h=self.fuse(torch.cat([self.cnn(img).flatten(1),self.event(ev),self.plane(mech)],1))
        s=self.scalar(h); v=self.strike(h)
        v=v/torch.clamp(torch.linalg.vector_norm(v,dim=1,keepdim=True),min=1e-8)
        return s,v

def get(z,*names):
    for n in names:
        if n in z.files:return z[n]
    raise KeyError(f"None of {names} found. Available: {z.files}")

def trig(a):
    r=np.deg2rad(np.asarray(a,np.float32))
    return np.stack([np.sin(r),np.cos(r)],axis=-1).astype(np.float32)

def decode(s,v):
    t=np.empty(9,float); t[IDX]=LO[IDX]+s*R[IDX]
    t[7]=np.degrees(np.arctan2(v[0],v[1]))%360.
    return t

def forward_pixel_los(theta,X,Y,le,ln,lu):
    kw=dict(x0=float(theta[0]),y0=float(theta[1]),depth=float(theta[2]),slip=float(theta[3]),
        x_grid=X,y_grid=Y,length=float(theta[4]),width=float(theta[5]),dip=float(theta[6]),
        strike=float(theta[7]),rake=float(theta[8]),los_vector=None)
    east,north,up=okada_forward(**kw)
    return east*le+north*ln+up*lu

def evaluate_event(name,checkpoint,data_root):
    cfg=EVENTS[name]; path=data_root/cfg["data"]
    if not path.is_file(): raise FileNotFoundError(path)
    z=np.load(path,allow_pickle=False)
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    obs=get(z,"X_obs","los","LOS","d_los","data","X").astype(np.float32)
    if obs.shape==(64,64,2):obs=np.moveaxis(obs,-1,0)
    mask=get(z,"mask","valid_mask","M").astype(bool)
    if mask.shape==(64,64,2):mask=np.moveaxis(mask,-1,0)
    mask &= np.isfinite(obs)
    lm=np.asarray(ck["los_mean"],np.float32); ls=np.asarray(ck["los_std"],np.float32)
    xn=(obs-lm[:,None,None])/ls[:,None,None]; xn[~np.isfinite(xn)]=0
    img=np.concatenate([xn,mask.astype(np.float32)],axis=0).astype(np.float32)

    X=torch.as_tensor(get(z,"X_km"),dtype=torch.float64)
    Y=torch.as_tensor(get(z,"Y_km"),dtype=torch.float64)
    le=np.asarray(get(z,"los_E"),float); ln=np.asarray(get(z,"los_N"),float); lu=np.asarray(get(z,"los_U"),float)
    if le.ndim==2:le=np.stack([le,le]);ln=np.stack([ln,ln]);lu=np.stack([lu,lu])

    model=InversionNet(); model.load_state_dict(ck["model_state"],strict=True); model.eval()
    em=np.asarray(ck["event_mean"],np.float32); es=np.asarray(ck["event_std"],np.float32)
    if em.shape!=(4,) or es.shape!=(4,):
        raise ValueError(f"Not an XY5/no-depth checkpoint: event stats shapes {em.shape}, {es.shape}")

    x,y,_depth,mw=cfg["event"]
    raw=np.array([x,y,mw,cfg["H"]],np.float32)  # seismic depth deliberately omitted
    ev=(raw-em)/es
    rows=[]
    for label,strike,dip,rake in cfg["planes"]:
        mech=np.concatenate([trig(strike),trig(dip),trig(rake)]).astype(np.float32)
        with torch.no_grad():
            s,v=model(torch.from_numpy(img[None]),torch.from_numpy(ev[None]),torch.from_numpy(mech[None]))
        theta=decode(s.numpy()[0],v.numpy()[0])
        pred=np.empty_like(obs,dtype=float)
        for c in range(2):
            pred[c]=forward_pixel_los(theta,X,Y,torch.as_tensor(le[c],dtype=torch.float64),
                torch.as_tensor(ln[c],dtype=torch.float64),torch.as_tensor(lu[c],dtype=torch.float64)).numpy()
        rm=[]
        for c in range(2):
            q=mask[c]&np.isfinite(pred[c])
            rm.append(float(np.sqrt(np.mean((pred[c,q]-obs[c,q])**2))*1000.))
        eq=float(np.sqrt((rm[0]**2+rm[1]**2)/2.))
        top=float(theta[2]-.5*theta[5]*np.sin(np.deg2rad(theta[6])))
        row={"event":name,"plane":label,**{n:float(v) for n,v in zip(NAMES,theta)},
             "top_depth":top,"asc_rmse_mm":rm[0],"des_rmse_mm":rm[1],"equal_rmse_mm":eq}
        rows.append(row)
        print(f"{name:10s} {label}: equal={eq:8.4f} mm  ASC={rm[0]:8.4f}  DES={rm[1]:8.4f}")
    z.close(); return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--event",choices=["jishishan","albania","morocco","all"],default="all")
    ap.add_argument("--model",choices=["baseline","piml"],required=True)
    ap.add_argument("--lambda-phys",type=float,default=None,help="PIML provenance label; inference is unchanged.")
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--data-root",type=Path,default=Path.home()/"Documents"/"PIML")
    ap.add_argument("--output",type=Path,default=None)
    a=ap.parse_args()
    if not a.checkpoint.is_file():raise SystemExit(f"Checkpoint not found: {a.checkpoint}")
    if a.model=="piml" and a.lambda_phys is None:raise SystemExit("--lambda-phys required for PIML provenance.")
    torch.set_num_threads(4)
    rows=[]
    for name in (list(EVENTS) if a.event=="all" else [a.event]):
        rows.extend(evaluate_event(name,a.checkpoint,a.data_root))
    if a.output is None:
        suffix="" if a.model=="baseline" else f"_lambda{a.lambda_phys:g}".replace(".","p")
        out=ROOT/"results"/"summary"/f"real_{a.model}{suffix}.csv"
    else:out=a.output
    out.parent.mkdir(parents=True,exist_ok=True)
    fields=["event","plane",*NAMES,"top_depth","asc_rmse_mm","des_rmse_mm","equal_rmse_mm"]
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print("Saved:",out)

if __name__=="__main__":main()
