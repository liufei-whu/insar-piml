#!/usr/bin/env python3
"""Apply the established realistic degradation to the full-360 single-NP dataset."""
from pathlib import Path
import json, time
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[2]
INPUT=PROJECT_ROOT/"data"/"clean_9d"/"global_adaptive_fov_single_np_360"/"dataset_9d_adaptive_fov_single_np_360.npz"
OUT_DIR=PROJECT_ROOT/"data"/"realistic_9d"/"global_adaptive_fov_single_np_360"
OUTPUT=OUT_DIR/"dataset_9d_adaptive_fov_single_np_360_realistic.npz"
GRID_N=64
CHUNK=256
DEFAULTS=dict(white_sigma_mm=5.0,atm_rms_mm=8.0,atm_corr_km=20.0,ramp_edge_mm=10.0,
              bilinear_edge_mm=3.0,decor_fraction=0.12,decor_sigma_mm=15.0,
              decor_corr_km=8.0,missing_fraction=0.12,missing_corr_km=10.0)
SEEDS={"train":73001,"val":73002,"test":73003}

def spectral_grf(rng,n,corr_km,batch,dx_km):
    z=rng.standard_normal((batch,2,n,n)).astype(np.float32)
    fy=np.fft.fftfreq(n,d=dx_km); fx=np.fft.rfftfreq(n,d=dx_km)
    ky,kx=np.meshgrid(fy,fx,indexing="ij")
    sigma_k=1.0/max(corr_km,dx_km)
    filt=np.exp(-.5*(kx*kx+ky*ky)/(sigma_k*sigma_k)).astype(np.float32)
    Z=np.fft.rfft2(z,axes=(-2,-1))
    field=np.fft.irfft2(Z*filt[None,None],s=(n,n),axes=(-2,-1)).astype(np.float32)
    field-=field.mean(axis=(-2,-1),keepdims=True)
    rms=np.sqrt(np.mean(field*field,axis=(-2,-1),keepdims=True))
    return field/np.maximum(rms,1e-8)

def exact_fraction_mask(score,fraction):
    B,C,H,W=score.shape; k=int(round(fraction*H*W)); flat=score.reshape(B,C,-1)
    out=np.zeros_like(flat,dtype=bool)
    if k<=0:return out.reshape(B,C,H,W)
    if k>=H*W:return np.ones_like(score,dtype=bool)
    idx=np.argpartition(flat,-k,axis=-1)[...,-k:]
    np.put_along_axis(out,idx,True,axis=-1)
    return out.reshape(B,C,H,W)

def degrade_chunk(X,Hvals,rng,cfg):
    B,_,n,_=X.shape
    out=np.empty_like(X); mask=np.empty_like(X,dtype=np.uint8)
    comps={k:np.empty_like(X) for k in ["white","atmosphere","ramp","bilinear","decorrelation"]}
    for h in (40.0,60.0):
        ii=np.flatnonzero(Hvals==h)
        if not len(ii):continue
        x=X[ii]; b=len(ii); dx=2.0*h/(n-1)
        white=rng.normal(0,cfg["white_sigma_mm"]/1000.0,size=x.shape).astype(np.float32)
        atm=spectral_grf(rng,n,cfg["atm_corr_km"],b,dx)*(cfg["atm_rms_mm"]/1000.0)
        yy,xx=np.meshgrid(np.linspace(-1,1,n,dtype=np.float32),np.linspace(-1,1,n,dtype=np.float32),indexing="ij")
        a=rng.uniform(-1,1,(b,2,1,1)).astype(np.float32); c=rng.uniform(-1,1,(b,2,1,1)).astype(np.float32)
        ramp=(cfg["ramp_edge_mm"]/1000.0)*(a*xx+c*yy)
        q=rng.uniform(-1,1,(b,2,3,1,1)).astype(np.float32)
        bil=(cfg["bilinear_edge_mm"]/1000.0)*(q[:,:,0]+q[:,:,1]*xx+q[:,:,2]*yy+q[:,:,0]*0+0.5*q[:,:,1]*q[:,:,2]*xx*yy)
        decor_score=spectral_grf(rng,n,cfg["decor_corr_km"],b,dx)
        decor_mask=exact_fraction_mask(decor_score,cfg["decor_fraction"])
        decor=rng.normal(0,cfg["decor_sigma_mm"]/1000.0,size=x.shape).astype(np.float32)*decor_mask
        miss_score=spectral_grf(rng,n,cfg["missing_corr_km"],b,dx)
        miss=exact_fraction_mask(miss_score,cfg["missing_fraction"])
        xr=x+white+atm+ramp+bil+decor; xr[miss]=np.nan
        out[ii]=xr; mask[ii]=(~miss).astype(np.uint8)
        for name,val in zip(comps,[white,atm,ramp,bil,decor]): comps[name][ii]=val
    return out,mask,comps

def print_stats(split,X,Xr,M,H):
    valid=M.astype(bool); d=Xr-X
    rmse=np.sqrt(np.nanmean(d*d))*1000
    clean=np.sqrt(np.mean(X*X))*1000
    print(f"{split}: valid={valid.mean():.3f}, degradation RMSE={rmse:.2f} mm, clean RMS={clean:.2f} mm")
    for h in (40.,60.):
        i=H==h
        if i.any(): print(f"  H{int(h)} n={i.sum()} RMSE={np.sqrt(np.nanmean(d[i]**2))*1000:.2f} mm")

def main():
    t0=time.time(); OUT_DIR.mkdir(parents=True,exist_ok=True)
    print("="*78); print("FULL-360 SINGLE-NP REALISTIC DEGRADATION"); print("="*78)
    print("Input:",INPUT); print("Output:",OUTPUT)
    z=np.load(INPUT,allow_pickle=False); cfg=DEFAULTS.copy(); out={}
    # Carry source/seismic/FOV arrays unchanged.
    for split in ("train","val","test"):
        X=z[f"X_{split}"].astype(np.float32); H=z[f"H_{split}"].astype(np.float32)
        rng=np.random.default_rng(SEEDS[split])
        Xr=np.empty_like(X); M=np.empty_like(X,dtype=np.uint8)
        comp={k:np.empty_like(X) for k in ["white","atmosphere","ramp","bilinear","decorrelation"]}
        for start in range(0,len(X),CHUNK):
            end=min(start+CHUNK,len(X))
            xr,m,c=degrade_chunk(X[start:end],H[start:end],rng,cfg)
            Xr[start:end]=xr; M[start:end]=m
            for k in comp: comp[k][start:end]=c[k]
        out[f"X_{split}"]=Xr; out[f"mask_{split}"]=M
        out[f"y_{split}"]=z[f"y_{split}"]; out[f"seismic_{split}"]=z[f"seismic_{split}"]
        out[f"H_{split}"]=z[f"H_{split}"]; out[f"Mw_true_{split}"]=z[f"Mw_true_{split}"]
        if split=="test":
            out["X_test_clean"]=X
            for name,a in comp.items(): out[f"test_{name}"]=a
        print_stats(split,X,Xr,M,H)
    clean_meta=json.loads(str(z["metadata_json"].item()))
    meta={"description":"Full-360 single-NP adaptive-FOV dataset with established realistic InSAR degradation",
          "units":"LOS displacement in metres","grid_n":GRID_N,
          "fov_rule":"H=40 km if Mw_s<=6.8 and depth_s<=20 km, otherwise H=60 km",
          "spacing_km":{"H40":80.0/63.0,"H60":120.0/63.0},
          "degradation":cfg,"seeds":SEEDS,"mask_convention":"1=valid, 0=missing; degraded X contains NaN at missing pixels",
          "notes":["Degradation amplitudes/fractions match the established realistic benchmark.",
                   "ASC and DES corruptions are independent.","Correlation lengths are fixed in physical km.",
                   "Single-NP seismic/source arrays are copied unchanged from the validated clean dataset."],
          "clean_dataset_metadata":clean_meta}
    out["metadata_json"]=np.array(json.dumps(meta))
    print("\nSaving:",OUTPUT); np.savez_compressed(OUTPUT,**out)
    print(f"Done in {(time.time()-t0)/60:.1f} min; output size={OUTPUT.stat().st_size/(1024**3):.2f} GB")
    print("Saved:",OUTPUT)
if __name__=="__main__":main()
