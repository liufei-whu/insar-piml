#!/usr/bin/env python3
"""Masked differentiable-Okada PIML sweep for frozen 50k XY5/no-depth setup.

Runs lambda_phys = 0.01, 0.1, 1.0 sequentially.
The optimizer batch remains 64 (identical to pure ML); physics is evaluated in
microbatches of 16. Physics residual is evaluated only on valid realistic-mask
pixels, against the corresponding CLEAN synthetic Okada LOS field.

Requires:
  data/realistic_9d/global_adaptive_fov_single_np_360/
      dataset_9d_adaptive_fov_single_np_360_realistic.npz
  data/clean_9d/global_adaptive_fov_single_np_360/
      dataset_9d_adaptive_fov_single_np_360.npz
"""
from pathlib import Path
import json,time,random,sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from physics.okada import okada_forward

REAL=ROOT/"data"/"realistic_9d"/"global_adaptive_fov_single_np_360"/"dataset_9d_adaptive_fov_single_np_360_realistic.npz"
CLEAN=ROOT/"data"/"clean_9d"/"global_adaptive_fov_single_np_360"/"dataset_9d_adaptive_fov_single_np_360.npz"
OUTROOT=ROOT/"results"/"piml"
LAMBDAS=[1.0]
SEED=42; BATCH=64; PHYS_MICROBATCH=16; MAX_EPOCHS=100; PATIENCE=12; LR=1e-3; WD=1e-5
GRAD_CLIP=5.0
TOP_MIN_KM=0.05
NAMES=["x0","y0","depth","slip","length","width","dip","strike","rake"]
BOUNDS=np.array([[-30,30],[-30,30],[5,40],[.1,3.5],[10,45],[5,25],[30,80],[0,360],[45,135]],np.float32)
SCALAR_IDX=np.array([0,1,2,3,4,5,6,8]); LO=BOUNDS[:,0]; HI=BOUNDS[:,1]; RNG=HI-LO
LOS_ASC=np.array([-0.558875709446653,-0.12902662410396473,0.8191520442889918],np.float32)
LOS_DES=np.array([0.5588757094466531,-0.1290266241039646,0.8191520442889918],np.float32)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)

def reset_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def circ_features(deg):
    r=np.deg2rad(deg.astype(np.float32))
    return np.stack([np.sin(r),np.cos(r)],axis=1).astype(np.float32)

def finite_los_stats(X):
    m=[];s=[]
    for c in range(2):
        v=X[:,c]; q=np.isfinite(v); m.append(float(v[q].mean())); s.append(float(v[q].std()))
    return np.array(m,np.float32),np.array(s,np.float32)

def weaken_xy(S,y):
    W=S.astype(np.float32).copy()
    W[:,:2]=y[:,:2].astype(np.float32)+(S[:,:2].astype(np.float32)-y[:,:2].astype(np.float32))*2.5
    return W

def build_seismic(S,event_mean,event_std):
    ev=S[:,[0,1,3,7]].astype(np.float32)
    ev=(ev-event_mean)/event_std
    mech=np.concatenate([circ_features(S[:,4]),circ_features(S[:,5]),circ_features(S[:,6])],axis=1)
    return ev.astype(np.float32),mech.astype(np.float32)

class DS(Dataset):
    def __init__(self,X,M,Xclean,S,y,lm,ls,em,es):
        xn=(X.astype(np.float32)-lm[None,:,None,None])/ls[None,:,None,None]
        xn[~np.isfinite(xn)]=0
        self.img=np.concatenate([xn,M.astype(np.float32)],axis=1)
        self.mask=M.astype(np.float32)
        self.clean=Xclean.astype(np.float32)
        self.ev,self.mech=build_seismic(S,em,es)
        self.scal=((y[:,SCALAR_IDX].astype(np.float32)-LO[SCALAR_IDX])/RNG[SCALAR_IDX]).astype(np.float32)
        self.strike=circ_features(y[:,7])
        self.truth=y.astype(np.float32); self.H=S[:,7].astype(np.float32)
    def __len__(self): return len(self.truth)
    def __getitem__(self,i):
        return (torch.from_numpy(self.img[i]),torch.from_numpy(self.ev[i]),torch.from_numpy(self.mech[i]),
                torch.from_numpy(self.scal[i]),torch.from_numpy(self.strike[i]),
                torch.from_numpy(self.mask[i]),torch.from_numpy(self.clean[i]),
                torch.from_numpy(self.truth[i]),torch.tensor(self.H[i]))

class Model(nn.Module):
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

def theta_loss(ps,pv,ys,yv):
    return (8*nn.functional.mse_loss(ps,ys)+2*nn.functional.mse_loss(pv,yv))/10

def decode_torch(ps,pv):
    # Differentiable physical units. Strike atan2(sin,cos) is periodic.
    cols=[]; j=0
    scalar_set=set(SCALAR_IDX.tolist())
    for k in range(9):
        if k==7:
            strike=torch.remainder(torch.atan2(pv[:,0],pv[:,1])*180.0/np.pi,360.0)
            cols.append(strike)
        elif k in scalar_set:
            cols.append(float(LO[k])+ps[:,j]*float(RNG[k])); j+=1
    return torch.stack(cols,dim=1)

def decode_np(ps,pv):
    out=np.empty((len(ps),9),np.float32); out[:,SCALAR_IDX]=LO[SCALAR_IDX]+ps*RNG[SCALAR_IDX]
    out[:,7]=(np.degrees(np.arctan2(pv[:,0],pv[:,1]))%360).astype(np.float32); return out

def physics_loss(ps,pv,mask,clean,H,phys_scale2):
    """Masked fixed-scale Okada loss, restricted to the clean generator's valid domain."""
    theta=decode_torch(ps,pv); total=theta.new_zeros(()); count=0
    losA=torch.as_tensor(LOS_ASC,dtype=theta.dtype,device=theta.device)
    losD=torch.as_tensor(LOS_DES,dtype=theta.dtype,device=theta.device)
    scale2=torch.as_tensor(phys_scale2,dtype=theta.dtype,device=theta.device)
    for st in range(0,len(theta),PHYS_MICROBATCH):
        en=min(st+PHYS_MICROBATCH,len(theta)); t=theta[st:en]
        m=mask[st:en]; target=clean[st:en]; h=H[st:en]
        diprad=t[:,6]*np.pi/180.0
        min_depth=TOP_MIN_KM+0.5*t[:,5]*torch.sin(diprad)
        violation=torch.relu(min_depth-t[:,2])
        depth_safe=torch.maximum(t[:,2],min_depth)
        u=torch.linspace(-1.,1.,64,dtype=t.dtype,device=t.device)
        yy0,xx0=torch.meshgrid(u,u,indexing="ij")
        X=h[:,None,None]*xx0[None]; Y=h[:,None,None]*yy0[None]
        q=[t[:,i,None,None] for i in range(9)]
        kw=dict(x0=q[0],y0=q[1],depth=depth_safe[:,None,None],slip=q[3],
                x_grid=X,y_grid=Y,length=q[4],width=q[5],dip=q[6],
                strike=q[7],rake=q[8])
        pred=torch.stack([okada_forward(**kw,los_vector=losA),
                          okada_forward(**kw,los_vector=losD)],dim=1)
        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite Okada prediction inside admissible domain")
        valid=m.sum(dim=(-2,-1)).clamp_min(1.)
        mse=((pred-target).square()*m).sum(dim=(-2,-1))/valid
        data_term=(mse/scale2[None,:]).mean(dim=1)
        admiss_term=violation.square()
        total+=(data_term+admiss_term).sum(); count+=len(t)
    return total/count

def circular_abs(a,b): return np.abs((a-b+180)%360-180)

@torch.no_grad()
def predict(model,loader):
    model.eval();P=[];T=[];HH=[]
    for img,ev,mech,_,_,_,_,truth,h in loader:
        ps,pv=model(img.to(device),ev.to(device),mech.to(device))
        P.append(decode_np(ps.cpu().numpy(),pv.cpu().numpy()));T.append(truth.numpy());HH.append(h.numpy())
    return np.concatenate(P),np.concatenate(T),np.concatenate(HH)

def metrics(pred,true,H):
    d=(pred-true)/RNG; d[:,7]=((pred[:,7]-true[:,7]+180)%360-180)/360
    E=np.sqrt(np.sum(d*d,axis=1)); ae=np.abs(pred-true); ae[:,7]=circular_abs(pred[:,7],true[:,7])
    out={"mean_E":float(E.mean()),"median_E":float(np.median(E)),
         "mae":{NAMES[i]:float(ae[:,i].mean()) for i in range(9)}}
    for h in (40.,60.):
        q=H==h; out[f"H{int(h)}"]={"n":int(q.sum()),"mean_E":float(E[q].mean()),"median_E":float(np.median(E[q]))}
    return out,E

def make_loaders():
    zr=np.load(REAL,allow_pickle=False); zc=np.load(CLEAN,allow_pickle=False)
    data={}
    for split in ("train","val","test"):
        X=zr[f"X_{split}"]; M=zr[f"mask_{split}"]; S0=zr[f"seismic_{split}"]; y=zr[f"y_{split}"]
        Xc=zc[f"X_{split}"]
        # Hard pairing checks: realistic degradation copied these arrays unchanged.
        if X.shape!=Xc.shape: raise RuntimeError(f"{split}: clean/realistic shape mismatch")
        if not np.array_equal(y,zc[f"y_{split}"]): raise RuntimeError(f"{split}: y arrays do not pair exactly")
        if not np.array_equal(S0,zc[f"seismic_{split}"]): raise RuntimeError(f"{split}: seismic arrays do not pair exactly")
        data[split]=(X,M,Xc,S0,y)
    Xtr,Mtr,Xctr,S0tr,ytr=data["train"]
    phys_scale2=np.mean(Xctr.astype(np.float64)**2,axis=(0,2,3)).astype(np.float32)
    if (not np.isfinite(phys_scale2).all()) or np.any(phys_scale2<=0):
        raise RuntimeError(f"Invalid physics scales: {phys_scale2}")
    Str=weaken_xy(S0tr,ytr)
    lm,ls=finite_los_stats(Xtr); evtr=Str[:,[0,1,3,7]].astype(np.float32)
    em=evtr.mean(0); es=np.maximum(evtr.std(0),1e-8)
    dsets=[]
    for split in ("train","val","test"):
        X,M,Xc,S0,y=data[split]; S=weaken_xy(S0,y)
        dsets.append(DS(X,M,Xc,S,y,lm,ls,em,es))
    return dsets,lm,ls,em,es,phys_scale2

def run_lambda(lam,dsets,lm,ls,em,es,phys_scale2):
    reset_seed()
    tag=str(lam).replace(".","p")
    OUT=OUTROOT/f"global_adaptive_fov_single_np_360_realistic_xy5_no_depth_masked_lambda{tag}_50k"
    OUT.mkdir(parents=True,exist_ok=True)
    tr,va,te=dsets
    g=torch.Generator().manual_seed(SEED)
    tl=DataLoader(tr,BATCH,shuffle=True,num_workers=0,generator=g)
    vl=DataLoader(va,BATCH,num_workers=0); ql=DataLoader(te,BATCH,num_workers=0)
    model=Model().to(device); opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=4)
    best=np.inf;best_ep=0;stale=0;hist=[];ck=OUT/"best_model.pt";t0=time.time()
    print("\n"+"="*92);print(f"MASKED PIML lambda_phys={lam:g}  optimizer batch={BATCH}  Okada microbatch={PHYS_MICROBATCH}  buried-domain=ON");print("="*92)
    for ep in range(1,MAX_EPOCHS+1):
        model.train(); st=sp=stot=0.; n=0
        for img,ev,mech,ys,yv,mask,clean,_,h in tl:
            img,ev,mech,ys,yv,mask,clean,h=[x.to(device) for x in (img,ev,mech,ys,yv,mask,clean,h)]
            opt.zero_grad(set_to_none=True)
            ps,pv=model(img,ev,mech)
            lt=theta_loss(ps,pv,ys,yv); lp=physics_loss(ps,pv,mask,clean,h,phys_scale2)
            loss=lt+lam*lp
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: lambda={lam}, epoch={ep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step()
            b=len(ys); st+=lt.item()*b; sp+=lp.item()*b; stot+=loss.item()*b; n+=b
        # Validation uses the SAME total objective for scheduler/checkpoint selection.
        model.eval(); vt=vp=vtot=0.; vn=0
        # no torch.no_grad(): Okada forward is differentiable code, but validation graph is not needed.
        with torch.no_grad():
            for img,ev,mech,ys,yv,mask,clean,_,h in vl:
                img,ev,mech,ys,yv,mask,clean,h=[x.to(device) for x in (img,ev,mech,ys,yv,mask,clean,h)]
                ps,pv=model(img,ev,mech); a=theta_loss(ps,pv,ys,yv); p=physics_loss(ps,pv,mask,clean,h,phys_scale2); q=a+lam*p
                b=len(ys); vt+=a.item()*b; vp+=p.item()*b; vtot+=q.item()*b; vn+=b
        A,P,T=st/n,sp/n,stot/n; VA,VP,VT=vt/vn,vp/vn,vtot/vn
        sch.step(VT); lr=opt.param_groups[0]["lr"]; hist.append([ep,A,P,T,VA,VP,VT,lr])
        print(f"epoch {ep:03d} theta={A:.6f} phys={P:.6f} total={T:.6f} | val_theta={VA:.6f} val_phys={VP:.6f} val_total={VT:.6f} lr={lr:.2e}")
        if VT<best:
            best=VT;best_ep=ep;stale=0
            torch.save({"model_state":model.state_dict(),"epoch":ep,"val_loss":VT,
                "val_theta_loss":VA,"val_phys_loss":VP,"lambda_phys":lam,"physics_masked":True,"top_min_km":TOP_MIN_KM,
                "physics_target":"paired clean synthetic LOS","physics_microbatch":PHYS_MICROBATCH,"physics_scale2":phys_scale2,
                "los_mean":lm,"los_std":ls,"event_mean":em,"event_std":es,
                "event_indices":np.array([0,1,3,7],dtype=np.int64),"bounds":BOUNDS,"parameter_names":NAMES,
                "seismic_representation":"event=[x,y,Mw,H], xy sigma=5 km, depth REMOVED; one NP=[sin/cos strike,dip,rake]",
                "xy_sigma_km":5.0,"depth_used":False,"method":"masked_differentiable_okada_piml_xy5_no_depth_50k"},ck)
        else:
            stale+=1
            if stale>=PATIENCE: print(f"Early stopping after epoch {ep}."); break
    c=torch.load(ck,map_location=device,weights_only=False);model.load_state_dict(c["model_state"])
    pred,true,H=predict(model,ql);result,E=metrics(pred,true,H);runtime=(time.time()-t0)/60
    result.update(best_epoch=int(best_ep),best_val_total_loss=float(best),runtime_min=float(runtime),
                  lambda_phys=float(lam),physics_masked=True,physics_microbatch=PHYS_MICROBATCH)
    np.savetxt(OUT/"training_history.csv",np.asarray(hist),delimiter=",",
               header="epoch,train_theta,train_phys,train_total,val_theta,val_phys,val_total,lr",comments="")
    np.savez_compressed(OUT/"test_predictions.npz",pred=pred,truth=true,E=E.astype(np.float32),H=H.astype(np.float32))
    with open(OUT/"test_metrics.json","w") as f: json.dump(result,f,indent=2)
    print("\nTEST",f"lambda={lam:g}",f"best_epoch={best_ep}",f"runtime={runtime:.2f} min",
          f"mean_E={result['mean_E']:.6f}",f"median_E={result['median_E']:.6f}")
    for nme in NAMES: print(f"{nme:>7s}: {result['mae'][nme]:.6f}")
    return result

def main():
    print("="*92);print("50k MASKED DIFFERENTIABLE-OKADA PIML SWEEP: XY5 + NO SEISMIC DEPTH");print("="*92)
    print("Device:",device);print("Realistic:",REAL);print("Clean:",CLEAN)
    print("Lambdas:",LAMBDAS); print(f"Okada admissibility: top >= {TOP_MIN_KM} km (same domain as clean generator)")
    dsets,lm,ls,em,es,phys_scale2=make_loaders()
    print("Pairing checks: PASS")
    print("Fixed clean RMS per track [m]:",np.sqrt(phys_scale2))
    print("Fixed clean RMS^2 per track [m^2]:",phys_scale2)
    print("Event [x,y,Mw,H] mean:",em);print("Event [x,y,Mw,H] std :",es)
    results={}
    for lam in LAMBDAS: results[str(lam)]=run_lambda(lam,dsets,lm,ls,em,es,phys_scale2)
    summary=OUTROOT/"masked_piml_xy5_no_depth_50k_sweep_summary.json"
    with open(summary,"w") as f: json.dump(results,f,indent=2)
    print("\n"+"="*92);print("SWEEP COMPLETE");print("="*92)
    for lam,r in results.items(): print(f"lambda={lam:>4s}: mean_E={r['mean_E']:.6f}, median_E={r['median_E']:.6f}, best_epoch={r['best_epoch']}")
    print("Summary:",summary)

if __name__=="__main__": main()
