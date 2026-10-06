#!/usr/bin/env python3
"""50k ablation: sigma_x=sigma_y=5 km, seismic depth removed.

Reuses the existing realistic 50k dataset. Original seismic x/y errors have
sigma~2 km, so their exact realizations are multiplied by 2.5 relative to truth.
Event conditioning is [x_seis_5km, y_seis_5km, Mw, H]. Depth is never supplied.
"""
from pathlib import Path
import json,time,random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"/"realistic_9d"/"global_adaptive_fov_single_np_360"/"dataset_9d_adaptive_fov_single_np_360_realistic.npz"
OUT=ROOT/"results"/"seismic_ml"/"global_adaptive_fov_single_np_360_realistic_xy5_no_depth_50k"
OUT.mkdir(parents=True,exist_ok=True)
SEED=42; BATCH=64; MAX_EPOCHS=100; PATIENCE=12; LR=1e-3; WD=1e-5
NAMES=["x0","y0","depth","slip","length","width","dip","strike","rake"]
BOUNDS=np.array([[-30,30],[-30,30],[5,40],[.1,3.5],[10,45],[5,25],[30,80],[0,360],[45,135]],np.float32)
SCALAR_IDX=np.array([0,1,2,3,4,5,6,8]); LO=BOUNDS[:,0]; HI=BOUNDS[:,1]; RNG=HI-LO
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ev=S[:,[0,1,3,7]].astype(np.float32)  # x,y,Mw,H; NO DEPTH
    ev=(ev-event_mean)/event_std
    mech=np.concatenate([circ_features(S[:,4]),circ_features(S[:,5]),circ_features(S[:,6])],axis=1)
    return ev.astype(np.float32),mech.astype(np.float32)

class DS(Dataset):
    def __init__(self,X,M,S,y,lm,ls,em,es):
        xn=(X.astype(np.float32)-lm[None,:,None,None])/ls[None,:,None,None]; xn[~np.isfinite(xn)]=0
        self.img=np.concatenate([xn,M.astype(np.float32)],axis=1)
        self.ev,self.mech=build_seismic(S,em,es)
        self.scal=((y[:,SCALAR_IDX].astype(np.float32)-LO[SCALAR_IDX])/RNG[SCALAR_IDX]).astype(np.float32)
        self.strike=circ_features(y[:,7]); self.truth=y.astype(np.float32); self.H=S[:,7].astype(np.float32)
    def __len__(self): return len(self.truth)
    def __getitem__(self,i):
        return (torch.from_numpy(self.img[i]),torch.from_numpy(self.ev[i]),torch.from_numpy(self.mech[i]),
                torch.from_numpy(self.scal[i]),torch.from_numpy(self.strike[i]),torch.from_numpy(self.truth[i]),
                torch.tensor(self.H[i]))

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

def loss_fn(ps,pv,ys,yv):
    return (8*nn.functional.mse_loss(ps,ys)+2*nn.functional.mse_loss(pv,yv))/10

def decode(ps,pv):
    out=np.empty((len(ps),9),np.float32); out[:,SCALAR_IDX]=LO[SCALAR_IDX]+ps*RNG[SCALAR_IDX]
    out[:,7]=(np.degrees(np.arctan2(pv[:,0],pv[:,1]))%360).astype(np.float32); return out

def circular_abs(a,b): return np.abs((a-b+180)%360-180)

@torch.no_grad()
def predict(model,loader):
    model.eval();P=[];T=[];H=[]
    for img,ev,mech,_,_,truth,h in loader:
        ps,pv=model(img.to(device),ev.to(device),mech.to(device))
        P.append(decode(ps.cpu().numpy(),pv.cpu().numpy()));T.append(truth.numpy());H.append(h.numpy())
    return np.concatenate(P),np.concatenate(T),np.concatenate(H)

def metrics(pred,true,H):
    d=(pred-true)/RNG; d[:,7]=((pred[:,7]-true[:,7]+180)%360-180)/360
    E=np.sqrt(np.sum(d*d,axis=1)); ae=np.abs(pred-true); ae[:,7]=circular_abs(pred[:,7],true[:,7])
    out={"mean_E":float(E.mean()),"median_E":float(np.median(E)),
         "mae":{NAMES[i]:float(ae[:,i].mean()) for i in range(9)}}
    for h in (40.,60.):
        q=H==h; out[f"H{int(h)}"]={"n":int(q.sum()),"mean_E":float(E[q].mean()),"median_E":float(np.median(E[q]))}
    return out,E

def main():
    print("="*78);print("50k XY5 + NO-DEPTH ABLATION: event=[x5,y5,Mw,H]");print("="*78)
    print("Device:",device);print("Loading:",DATA)
    z=np.load(DATA,allow_pickle=False)
    Xtr,Mtr,S0tr,ytr=z["X_train"],z["mask_train"],z["seismic_train"],z["y_train"]
    Xva,Mva,S0va,yva=z["X_val"],z["mask_val"],z["seismic_val"],z["y_val"]
    Xte,Mte,S0te,yte=z["X_test"],z["mask_test"],z["seismic_test"],z["y_test"]
    Str=weaken_xy(S0tr,ytr); Sva=weaken_xy(S0va,yva); Ste=weaken_xy(S0te,yte)
    base=(S0tr[:,:2]-ytr[:,:2]).std(0); new=(Str[:,:2]-ytr[:,:2]).std(0)
    print("Baseline XY error std:",base);print("XY5 error std     :",new);print("Seismic depth supplied to model: NO")
    lm,ls=finite_los_stats(Xtr); evtr=Str[:,[0,1,3,7]].astype(np.float32)
    em=evtr.mean(0); es=np.maximum(evtr.std(0),1e-8)
    print("Event [x,y,Mw,H] mean:",em);print("Event [x,y,Mw,H] std :",es)
    tr=DS(Xtr,Mtr,Str,ytr,lm,ls,em,es);va=DS(Xva,Mva,Sva,yva,lm,ls,em,es);te=DS(Xte,Mte,Ste,yte,lm,ls,em,es)
    g=torch.Generator().manual_seed(SEED)
    tl=DataLoader(tr,BATCH,shuffle=True,num_workers=0,generator=g);vl=DataLoader(va,BATCH,num_workers=0);ql=DataLoader(te,BATCH,num_workers=0)
    model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=4)
    best=np.inf;best_ep=0;stale=0;hist=[];ck=OUT/"best_model.pt";t0=time.time()
    for ep in range(1,MAX_EPOCHS+1):
        model.train();sm=0.;n=0
        for img,ev,mech,ys,yv,_,_ in tl:
            img,ev,mech,ys,yv=[x.to(device) for x in (img,ev,mech,ys,yv)]
            opt.zero_grad(set_to_none=True);ps,pv=model(img,ev,mech);loss=loss_fn(ps,pv,ys,yv)
            loss.backward();opt.step();sm+=loss.item()*len(ys);n+=len(ys)
        model.eval();vm=0.;vn=0
        with torch.no_grad():
            for img,ev,mech,ys,yv,_,_ in vl:
                img,ev,mech,ys,yv=[x.to(device) for x in (img,ev,mech,ys,yv)]
                ps,pv=model(img,ev,mech);l=loss_fn(ps,pv,ys,yv);vm+=l.item()*len(ys);vn+=len(ys)
        a=sm/n;b=vm/vn;sch.step(b);lr=opt.param_groups[0]["lr"];hist.append([ep,a,b,lr])
        print(f"epoch {ep:03d} train={a:.6f} val={b:.6f} lr={lr:.2e}")
        if b<best:
            best=b;best_ep=ep;stale=0
            torch.save({"model_state":model.state_dict(),"epoch":ep,"val_loss":b,"los_mean":lm,"los_std":ls,
                "event_mean":em,"event_std":es,"event_indices":np.array([0,1,3,7],dtype=np.int64),
                "bounds":BOUNDS,"parameter_names":NAMES,
                "seismic_representation":"event=[x,y,Mw,H], xy sigma=5 km, depth REMOVED; one NP=[sin/cos strike,dip,rake]",
                "xy_sigma_km":5.0,"depth_used":False,"method":"single_np_360_realistic_xy5_no_depth_50k"},ck)
        else:
            stale+=1
            if stale>=PATIENCE:print(f"Early stopping after epoch {ep}.");break
    c=torch.load(ck,map_location=device,weights_only=False);model.load_state_dict(c["model_state"])
    pred,true,H=predict(model,ql);result,E=metrics(pred,true,H);runtime=(time.time()-t0)/60
    result.update(best_epoch=int(best_ep),best_val_loss=float(best),runtime_min=float(runtime),xy_sigma_km=5.0,seismic_depth_used=False)
    np.savetxt(OUT/"training_history.csv",np.asarray(hist),delimiter=",",header="epoch,train_loss,val_loss,lr",comments="")
    np.savez_compressed(OUT/"test_predictions.npz",pred=pred,truth=true,E=E.astype(np.float32),H=H.astype(np.float32))
    with open(OUT/"test_metrics.json","w") as f:json.dump(result,f,indent=2)
    print("\n"+"="*78);print("TEST RESULTS");print("="*78)
    print(f"Best epoch: {best_ep}  Runtime: {runtime:.2f} min  Mean E: {result['mean_E']:.6f}  Median E: {result['median_E']:.6f}")
    for n in NAMES:print(f"{n:>7s}: {result['mae'][n]:.6f}")
    for h in (40,60):
        r=result[f"H{h}"];print(f"H={h}: n={r['n']}, mean E={r['mean_E']:.6f}, median E={r['median_E']:.6f}")
    print("Saved checkpoint:",ck)
if __name__=="__main__":main()
