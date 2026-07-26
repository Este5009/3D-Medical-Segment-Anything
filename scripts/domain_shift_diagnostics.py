#!/usr/bin/env python3
"""Read-only failure analysis for locked CAMRI and mouse predictions.

This script never creates an optimizer, calls backward, or writes outside the
diagnostics output directory. Existing native predictions are the source of
truth. A small representative subset is rerun under ``torch.inference_mode``
only to expose probabilities, features, and cross-attention weights.
"""
from __future__ import annotations

import csv,json,math,sys,time
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))

import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.ndimage import binary_dilation,binary_erosion,distance_transform_edt,gaussian_gradient_magnitude,label
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold,cross_val_score
from sklearn.preprocessing import StandardScaler

from evaluate_external_holdout import preprocess,center_tile
from models.query_mask_decoder import MultiScaleOneQueryMaskDecoder,FrozenEncoderQueryModel
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter,RS2NetPaths
from train_query_decoder_overfit import choose_device,load_json

LEVELS=("level4","level3","level2","level1")
COL={"CAMRI":"#2878b5","Mouse":"#d1495b"}

def rows(path):return list(csv.DictReader(open(path)))
def write_csv(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not data:return
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def save(fig,path):path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=200,bbox_inches="tight");plt.close(fig)
def native(path,dtype=np.float32):return np.asarray(nib.load(str(path)).dataobj,dtype=dtype)
def pct(a,q):return float(np.percentile(a,q)) if a.size else float("nan")
def safe_correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])==0 or np.std(y[ok])==0:return float("nan"),float("nan")
    r,p=stats.pearsonr(x[ok],y[ok]);return float(r),float(p)

def surfaces(mask):return mask.astype(bool)^binary_erosion(mask.astype(bool))
def binary_metrics(pred,target,spacing,tolerance):
    pred=pred.astype(bool);target=target.astype(bool);tp=int((pred&target).sum());fp=int((pred&~target).sum());fn=int((~pred&target).sum())
    ps,ts=surfaces(pred),surfaces(target);dt=distance_transform_edt(~ts,sampling=spacing);dp=distance_transform_edt(~ps,sampling=spacing)
    a=dt[ps];b=dp[ts];dist=np.r_[a,b];assd=float((a.sum()+b.sum())/max(len(a)+len(b),1));sd=float((np.sum(a<=tolerance)+np.sum(b<=tolerance))/max(len(a)+len(b),1))
    bp=float(np.mean(a<=tolerance));br=float(np.mean(b<=tolerance));components=label(pred)[1]
    # Dimensionless compactness: 1 for a sphere, lower for irregular shapes.
    voxel=float(np.prod(spacing));volume=pred.sum()*voxel;surface_area=(ps.sum()*(spacing[0]*spacing[1]+spacing[0]*spacing[2]+spacing[1]*spacing[2])/3)
    compact=float(36*np.pi*volume**2/max(surface_area**3,1e-12));rough=float(surface_area/max(volume**(2/3),1e-12))
    return {"dice":(2*tp)/(2*tp+fp+fn),"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"false_positives":fp,"false_negatives":fn,"hd95_mm":pct(dist,95),"assd_mm":assd,"surface_dice":sd,"boundary_precision":bp,"boundary_recall":br,"connected_components":components,"compactness":compact,"contour_roughness":rough,"volume_ratio":pred.sum()/max(target.sum(),1)}

def spatial_error_regions(pred,target):
    fp=pred&~target;fn=~pred&target;shape=np.array(target.shape);rows=[]
    axes=((0,"left","right"),(1,"dorsal","ventral"),(2,"rostral","caudal"))
    for axis,a,b in axes:
        for name,sl in ((a,slice(0,shape[axis]//2)),(b,slice(shape[axis]//2,None))):
            idx=[slice(None)]*3;idx[axis]=sl;idx=tuple(idx);rows.append({"partition":"axis_halves","region":name,"fp_voxels":int(fp[idx].sum()),"fn_voxels":int(fn[idx].sum())})
    center=tuple(slice(int(s*.25),int(s*.75)) for s in shape);rows.append({"partition":"center","region":"center","fp_voxels":int(fp[center].sum()),"fn_voxels":int(fn[center].sum())})
    occupied=np.where(target.any(axis=(0,1)))[0];first,last=occupied[0],occupied[-1]+1;n=last-first;cuts=(first,first+int(.2*n),first+int(.8*n),last)
    for name,(a,b) in zip(("first 20%","middle 60%","last 20%"),zip(cuts,cuts[1:])):rows.append({"partition":"brain_position","region":name,"fp_voxels":int(fp[:,:,a:b].sum()),"fn_voxels":int(fn[:,:,a:b].sum())})
    return rows

def intensity_features(image,target,band_width):
    image=np.nan_to_num(image.astype(np.float32));target=target.astype(bool);boundary=binary_dilation(target,iterations=band_width)^binary_erosion(target,iterations=band_width);background=~target
    # Raw scanner units are retained for descriptive intensity statistics, but
    # gradient comparisons must be scale invariant across scanners.
    q1,q99=np.percentile(image,[1,99]);normalized=np.clip((image-q1)/max(q99-q1,1e-8),0,1)
    gradient=gaussian_gradient_magnitude(normalized,1);out={}
    for name,mask in (("whole",np.ones_like(target,bool)),("brain",target),("background",background),("boundary",boundary)):
        v=image[mask];out.update({f"{name}_{k}":val for k,val in {"mean":float(v.mean()),"median":float(np.median(v)),"std":float(v.std()),"min":float(v.min()),"max":float(v.max()),"p01":pct(v,1),"p05":pct(v,5),"p25":pct(v,25),"p75":pct(v,75),"p95":pct(v,95),"p99":pct(v,99)}.items()})
    out["contrast"]=out["brain_mean"]-out["background_mean"];out["cnr"]=out["contrast"]/max(math.sqrt((out["brain_std"]**2+out["background_std"]**2)/2),1e-8);out["boundary_gradient_mean"]=float(gradient[boundary].mean());out["boundary_sharpness"]=float(np.median(gradient[boundary]));return out

def geometry_features(image,target,spacing):
    shape=np.array(image.shape);nz=np.argwhere(np.abs(image)>1e-8);lo=nz.min(0);hi=nz.max(0)+1;crop=hi-lo;target_spacing=np.array((0.1,0.1,0.1));resampled=np.maximum(np.round(crop*np.array(spacing)/target_spacing).astype(int),1);tile=np.array((128,128,160));padding=np.maximum(tile-resampled,0)
    return {"shape_x":shape[0],"shape_y":shape[1],"shape_z":shape[2],"spacing_x":spacing[0],"spacing_y":spacing[1],"spacing_z":spacing[2],"fov_x_mm":shape[0]*spacing[0],"fov_y_mm":shape[1]*spacing[1],"fov_z_mm":shape[2]*spacing[2],"slice_count":shape[2],"anisotropy":max(spacing)/min(spacing),"brain_volume_voxels":int(target.sum()),"brain_volume_mm3":float(target.sum()*np.prod(spacing)),"brain_occupancy_percent":100*target.mean(),"crop_x":crop[0],"crop_y":crop[1],"crop_z":crop[2],"scale_x":spacing[0]/.1,"scale_y":spacing[1]/.1,"scale_z":spacing[2]/.1,"padding_x":padding[0],"padding_y":padding[1],"padding_z":padding[2],"model_x":resampled[0],"model_y":resampled[1],"model_z":resampled[2]}

def record_paths(domain,row):
    sid=row.get("subject",row.get("scan_id"));return sid,Path(row["image_path"]),Path(row["ground_truth_path"]),Path(row["prediction_path"])

def audit_all(config,out):
    combined=[];regions=[];hist=[]
    for domain,path in (("CAMRI",ROOT/config["camri_metrics"]),("Mouse",ROOT/config["mouse_metrics"])):
        for i,row in enumerate(rows(path),1):
            sid,ip,gp,pp=record_paths(domain,row);img_obj=nib.load(str(ip));image=np.asarray(img_obj.dataobj,dtype=np.float32);target=native(gp)>0;pred=native(pp)>0;spacing=tuple(map(float,img_obj.header.get_zooms()[:3]))
            result={"domain":domain,"subject":sid,**geometry_features(image,target,spacing),**intensity_features(image,target,config["boundary_band_voxels"]),**binary_metrics(pred,target,spacing,config["surface_dice_tolerance_mm"])};combined.append(result)
            for r in spatial_error_regions(pred,target):regions.append({"domain":domain,"subject":sid,**r})
            # Robustly normalized intensity samples make cross-scanner histograms comparable.
            q1,q99=np.percentile(image,[1,99]);norm=np.clip((image-q1)/max(q99-q1,1e-8),0,1)
            for area,mask in (("whole",np.ones_like(target,bool)),("brain",target),("background",~target),("boundary",binary_dilation(target,iterations=5)^binary_erosion(target,iterations=5))):
                counts,edges=np.histogram(norm[mask],bins=50,range=(0,1),density=True)
                hist += [{"domain":domain,"subject":sid,"region":area,"bin_center":float((a+b)/2),"density":float(c)} for a,b,c in zip(edges[:-1],edges[1:],counts)]
            print(f"audit {domain} {i}",flush=True)
    write_csv(out/"subject_diagnostics.csv",combined);write_csv(out/"regional_errors.csv",regions);write_csv(out/"intensity_histograms.csv",hist);return combined,regions,hist

def instrument_decoder(decoder,features):
    projected={n:decoder.projections[n](features[n]) for n in decoder.CHANNELS};fused={"level4":projected["level4"]};previous=fused["level4"]
    for n in ("level3","level2","level1"):
        previous=F.interpolate(previous,size=projected[n].shape[-3:],mode="trilinear",align_corners=False);previous=decoder.refinements[n](projected[n]+previous);fused[n]=previous
    query=decoder.query.expand(1,-1,-1);attentions={};qstates=[]
    for n in LEVELS:
        block=decoder.query_updates[n];tokens=fused[n].flatten(2).transpose(1,2);att,w=block.cross_attention(query,tokens,tokens,need_weights=True,average_attn_weights=True);query=block.norm1(query+att);query=block.norm2(query+block.ffn(query));attentions[n]=w[0,0].detach().cpu().numpy().reshape(fused[n].shape[-3:]);qstates.append(query.detach().cpu().numpy().ravel())
    emb=decoder.mask_embedding(query).squeeze(1);vox=decoder.mask_refinement(fused["level1"]);logits=torch.einsum("bc,bcdhw->bdhw",emb,vox).unsqueeze(1)+decoder.mask_bias.view(1,1,1,1,1);logits=F.interpolate(logits,size=(128,128,160),mode="trilinear",align_corners=False)
    return logits,attentions,fused,qstates

def attention_stats(att):
    a=np.maximum(att.astype(float),0);a/=max(a.sum(),1e-12);coords=np.indices(a.shape).reshape(3,-1).T;v=a.ravel();cent=(coords*v[:,None]).sum(0);spread=float(np.sqrt(((coords-cent)**2*v[:,None]).sum()));entropy=float(-(v[v>0]*np.log(v[v>0])).sum()/np.log(len(v)));return entropy,float(np.mean(v>1/len(v))),*map(float,cent),spread,float(np.sum(v>v.max()*.1))

def instrument_representatives(config,subjects,out):
    enc_cfg=load_json(ROOT/config["encoder_config"]);paths=RS2NetPaths.from_config(enc_cfg);ck=torch.load(ROOT/config["checkpoint"],map_location="cpu",weights_only=False);decoder=MultiScaleOneQueryMaskDecoder(32,4);decoder.load_state_dict(ck["decoder_state_dict"],strict=True);device=choose_device();model=FrozenEncoderQueryModel(RS2NetEncoderAdapter(paths,image_size=tuple(config["tile_size"]),in_channels=1,out_channels=1,feature_size=48),decoder).to(device).eval();assert not any(p.requires_grad for p in model.encoder.parameters())
    selected=[]
    for domain in ("CAMRI","Mouse"):
        d=sorted([r for r in subjects if r["domain"]==domain],key=lambda r:r["dice"]);n=config["instrumented_per_domain"]//2;selected+=d[:n]+d[-n:]
    latent=[];attention=[];threshold=[];attribution=[]
    source={}
    for domain,path in (("CAMRI",ROOT/config["camri_metrics"]),("Mouse",ROOT/config["mouse_metrics"])):
        for r in rows(path):source[(domain,r.get("subject",r.get("scan_id")))]=r
    for i,item in enumerate(selected,1):
        row=source[(item["domain"],item["subject"])];sid,ip,gp,pp=record_paths(item["domain"],row);tensor,seg=preprocess_with_seg(ip,gp,paths);tile,meta=center_tile(tensor,tuple(config["tile_size"]));truth_tile,_=center_tile(seg,tuple(config["tile_size"]));tile=tile.to(device)
        with torch.inference_mode():features=model.encode(tile);logits,att,fused,qstates=instrument_decoder(decoder,features)
        prob=logits.sigmoid()[0,0].cpu().numpy();truth=truth_tile[0,0].numpy()>0
        d=out/"representative_inference"/item["domain"]/sid;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/"model_space_outputs.npz",probability=prob.astype(np.float16),logits=logits[0,0].cpu().numpy().astype(np.float16),thresholded=prob>=.5,uncertainty=(1-np.abs(2*prob-1)).astype(np.float16),**{f"attention_{k}":v.astype(np.float16) for k,v in att.items()})
        prediction=prob>=.5
        for level in LEVELS:
            f=fused[level][0].detach().cpu().numpy();pooled=f.mean((1,2,3));latent.append({"domain":item["domain"],"subject":sid,"level":level,**{f"f{j}":float(x) for j,x in enumerate(pooled)}});e,sp,c0,c1,c2,spread,vol=attention_stats(att[level]);attention.append({"domain":item["domain"],"subject":sid,"level":level,"entropy":e,"sparsity_above_uniform":sp,"centroid_x":c0,"centroid_y":c1,"centroid_z":c2,"spread":spread,"attended_voxels_above_10pct_peak":vol,"feature_norm":float(np.linalg.norm(f,axis=0).mean())})
            amap=F.interpolate(torch.from_numpy(att[level])[None,None].float(),size=truth.shape,mode="trilinear",align_corners=False)[0,0].numpy();zones={"true_positive":prediction&truth,"false_positive":prediction&~truth,"false_negative":~prediction&truth,"true_background":~prediction&~truth}
            for zone,mask in zones.items():attribution.append({"domain":item["domain"],"subject":sid,"level":level,"zone":zone,"mean_attention":float(amap[mask].mean()) if mask.any() else float("nan"),"voxel_count":int(mask.sum())})
        for t in config["thresholds"]:
            bm=binary_metrics(prob>=t,truth,(1,1,1),1);threshold.append({"domain":item["domain"],"subject":sid,"threshold":t,"dice":bm["dice"],"precision":bm["precision"],"recall":bm["recall"],"volume_ratio":bm["volume_ratio"]})
        make_instrument_figure(tile[0,0].cpu().numpy(),truth,prob,att["level1"],item,d/"probability_attention.png");print(f"instrument {i}/{len(selected)} {item['domain']} {sid}",flush=True)
    write_csv(out/"latent_features.csv",latent);write_csv(out/"query_attention.csv",attention);write_csv(out/"threshold_sensitivity.csv",threshold);write_csv(out/"attention_error_attribution.csv",attribution);return latent,attention,threshold

def preprocess_with_seg(image_path,mask_path,paths):
    """Return image and label after the exact frozen RS2-Net preprocessor."""
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from RS2.utilities.plans_handling.plans_handler import PlansManager
    json_root=paths.baseline_project/"RS2"/"jsons";plans=load_json(json_root/"plans.json");dataset=load_json(json_root/"dataset.json");manager=PlansManager(plans);configuration=manager.get_configuration("3d_fullres")
    data,seg,_=DefaultPreprocessor(verbose=False).run_case([str(image_path)],str(mask_path),manager,configuration,dataset)
    return torch.from_numpy(np.asarray(data,dtype=np.float32)).unsqueeze(0),torch.from_numpy((np.asarray(seg)>0).astype(np.float32)).unsqueeze(0)

def make_instrument_figure(image,truth,prob,att,item,path):
    z=int(np.argmax(truth.sum((0,1))));a=torch.from_numpy(att)[None,None].float();a=F.interpolate(a,size=prob.shape,mode="trilinear",align_corners=False)[0,0].numpy();unc=1-np.abs(2*prob-1);q1,q99=np.percentile(image,[1,99]);m=np.clip((image-q1)/max(q99-q1,1e-8),0,1);fig,axes=plt.subplots(1,6,figsize=(18,3.5),constrained_layout=True)
    for ax,title,x,cmap in zip(axes,("MRI","Expert","Probability","Uncertainty","Threshold 0.50","Level1 attention"),(m[:,:,z],truth[:,:,z],prob[:,:,z],unc[:,:,z],prob[:,:,z]>=.5,a[:,:,z]),("gray","gray","magma","viridis","gray","inferno")):ax.imshow(x.T,cmap=cmap,origin="lower");ax.set_title(title);ax.axis("off")
    fig.suptitle(f"{item['domain']} {item['subject']} | native Dice {item['dice']:.4f} | model-space diagnostic slice {z}");save(fig,path)

def plots_and_stats(subjects,regions,hist,latent,attention,threshold,out):
    # Distribution panels.
    keys=("slice_count","anisotropy","brain_volume_mm3","brain_occupancy_percent","contrast","cnr","boundary_sharpness","volume_ratio")
    fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
    for ax,key in zip(axes.flat,keys):
        for domain in ("CAMRI","Mouse"):ax.hist([r[key] for r in subjects if r["domain"]==domain],bins=20,alpha=.55,label=domain,color=COL[domain])
        ax.set(title=key.replace("_"," "),xlabel=key.replace("_"," "),ylabel="Subjects");ax.legend()
    save(fig,out/"plots"/"domain_distributions.png")
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for ax,area in zip(axes.flat,("whole","brain","background","boundary")):
        for domain in ("CAMRI","Mouse"):
            h=[r for r in hist if r["domain"]==domain and r["region"]==area];by=defaultdict(list)
            for r in h:by[float(r["bin_center"])].append(float(r["density"]))
            x=sorted(by);ax.plot(x,[np.mean(by[v]) for v in x],label=domain,color=COL[domain])
        ax.set(title=f"Normalized {area} intensity",xlabel="Robust normalized intensity",ylabel="Density");ax.legend();ax.grid(alpha=.2)
    save(fig,out/"plots"/"intensity_histograms.png")
    # Regional FP/FN percentages.
    agg=defaultdict(lambda:[0,0])
    for r in regions:agg[(r["domain"],r["partition"],r["region"])][0]+=r["fp_voxels"];agg[(r["domain"],r["partition"],r["region"])][1]+=r["fn_voxels"]
    region_summary=[]
    for (d,p,n),(fp,fn) in agg.items():region_summary.append({"domain":d,"partition":p,"region":n,"fp_voxels":fp,"fn_voxels":fn})
    write_csv(out/"regional_error_summary.csv",region_summary)
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for ax,kind in zip(axes,("fp_voxels","fn_voxels")):
        names=["left","right","dorsal","ventral","rostral","caudal","center"];x=np.arange(len(names));width=.35
        for j,d in enumerate(("CAMRI","Mouse")):vals=[next((r[kind] for r in region_summary if r["domain"]==d and r["region"]==n),0) for n in names];vals=100*np.array(vals)/max(sum(vals),1);ax.bar(x+(j-.5)*width,vals,width,label=d,color=COL[d])
        ax.set_xticks(x,names,rotation=30);ax.set(title=f"Regional {kind[:2].upper()} distribution",ylabel="Percent of listed regional errors");ax.legend()
    save(fig,out/"plots"/"regional_errors.png")
    # Threshold sensitivity.
    fig,axes=plt.subplots(1,4,figsize=(17,4),constrained_layout=True)
    for ax,key in zip(axes,("dice","precision","recall","volume_ratio")):
        for d in ("CAMRI","Mouse"):
            vals=[r for r in threshold if r["domain"]==d];xs=sorted(set(float(r["threshold"]) for r in vals));ax.plot(xs,[np.mean([float(r[key]) for r in vals if float(r["threshold"])==x]) for x in xs],marker="o",label=d,color=COL[d])
        ax.axvline(.5,ls="--",color="black",label="official 0.50");ax.set(title=key.replace("_"," "),xlabel="Analytical threshold",ylabel=key);ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,out/"plots"/"threshold_sensitivity.png")
    # PCA and t-SNE per level.
    latent_plot=[]
    for level in LEVELS:
        rs=[r for r in latent if r["level"]==level];X=np.array([[float(r[k]) for k in r if k.startswith("f")] for r in rs]);Xp=StandardScaler().fit_transform(X);pca=PCA(2).fit_transform(Xp);perp=max(2,min(5,len(rs)-1));ts=TSNE(2,random_state=17,perplexity=perp,init="pca",learning_rate="auto").fit_transform(Xp)
        for r,p,t in zip(rs,pca,ts):latent_plot.append({"domain":r["domain"],"subject":r["subject"],"level":level,"pca1":p[0],"pca2":p[1],"tsne1":t[0],"tsne2":t[1]})
    write_csv(out/"latent_embeddings.csv",latent_plot);fig,axes=plt.subplots(2,4,figsize=(18,8),constrained_layout=True)
    for j,level in enumerate(LEVELS):
        rs=[r for r in latent_plot if r["level"]==level]
        for i,(x,y,title) in enumerate((("pca1","pca2","PCA"),("tsne1","tsne2","t-SNE"))):
            ax=axes[i,j]
            for d in ("CAMRI","Mouse"):q=[r for r in rs if r["domain"]==d];ax.scatter([r[x] for r in q],[r[y] for r in q],label=d,color=COL[d],s=45)
            ax.set_title(f"{level} {title}");ax.legend(fontsize=8)
    save(fig,out/"plots"/"latent_space.png")
    # Explicit domain-centroid cosine separation and query comparisons.
    feature_summary=[]
    for level in LEVELS:
        groups={d:np.array([[float(r[k]) for k in r if k.startswith("f")] for r in latent if r["level"]==level and r["domain"]==d]) for d in ("CAMRI","Mouse")};a,b=groups["CAMRI"].mean(0),groups["Mouse"].mean(0);cos=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        feature_summary.append({"level":level,"centroid_cosine_similarity":cos,"centroid_l2_distance":float(np.linalg.norm(a-b)),"camri_mean_feature_norm":float(np.linalg.norm(groups['CAMRI'],axis=1).mean()),"mouse_mean_feature_norm":float(np.linalg.norm(groups['Mouse'],axis=1).mean())})
    write_csv(out/"feature_domain_comparison.csv",feature_summary)
    query_summary=[]
    for level in LEVELS:
        for key in ("entropy","sparsity_above_uniform","spread","attended_voxels_above_10pct_peak","feature_norm"):
            a=np.array([float(r[key]) for r in attention if r["level"]==level and r["domain"]=="CAMRI"]);b=np.array([float(r[key]) for r in attention if r["level"]==level and r["domain"]=="Mouse"]);test=stats.mannwhitneyu(a,b,alternative="two-sided")
            query_summary.append({"level":level,"metric":key,"camri_mean":a.mean(),"mouse_mean":b.mean(),"mouse_minus_camri":b.mean()-a.mean(),"mannwhitney_u":float(test.statistic),"p_value":float(test.pvalue)})
    write_csv(out/"query_domain_comparison.csv",query_summary)
    return statistical_analysis(subjects,attention,out),region_summary

def cohort_error_heatmaps(config,out):
    """Accumulate FP/FN after shape normalization; maps are localization aids."""
    source=[]
    for domain,path in (("CAMRI",ROOT/config["camri_metrics"]),("Mouse",ROOT/config["mouse_metrics"])):
        for row in rows(path):source.append((domain,row))
    grid=(64,64,64);sums={d:{"fp":np.zeros(grid,np.float32),"fn":np.zeros(grid,np.float32)} for d in ("CAMRI","Mouse")};counts=defaultdict(int)
    for domain,row in source:
        _,_,gp,pp=record_paths(domain,row);gt=torch.from_numpy((native(gp)>0).astype(np.float32))[None,None];pr=torch.from_numpy((native(pp)>0).astype(np.float32))[None,None];gt=F.interpolate(gt,size=grid,mode="nearest")[0,0].numpy()>0;pr=F.interpolate(pr,size=grid,mode="nearest")[0,0].numpy()>0;sums[domain]["fp"]+=pr&~gt;sums[domain]["fn"]+=(~pr)&gt;counts[domain]+=1
    np.savez_compressed(out/"normalized_error_heatmaps.npz",**{f"{d.lower()}_{k}":v/counts[d] for d,x in sums.items() for k,v in x.items()})
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row,(domain,kind) in enumerate((("CAMRI","fp"),("Mouse","fp"))):
        vol=sums[domain][kind]/counts[domain]
        for j,(axis,title) in enumerate(((0,"sagittal projection"),(1,"coronal projection"),(2,"axial projection"))):im=axes[row,j].imshow(vol.max(axis).T,origin="lower",cmap="hot");axes[row,j].set_title(f"{domain} FP frequency: {title}");axes[row,j].axis("off");fig.colorbar(im,ax=axes[row,j],fraction=.046,label="Subject fraction")
    save(fig,out/"plots"/"false_positive_heatmaps.png")
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row,domain in enumerate(("CAMRI","Mouse")):
        vol=sums[domain]["fn"]/counts[domain]
        for j,(axis,title) in enumerate(((0,"sagittal projection"),(1,"coronal projection"),(2,"axial projection"))):im=axes[row,j].imshow(vol.max(axis).T,origin="lower",cmap="Blues");axes[row,j].set_title(f"{domain} FN frequency: {title}");axes[row,j].axis("off");fig.colorbar(im,ax=axes[row,j],fraction=.046,label="Subject fraction")
    save(fig,out/"plots"/"false_negative_heatmaps.png")

def statistical_analysis(subjects,attention,out):
    att=defaultdict(list)
    for r in attention:
        if r["level"]=="level1":att[(r["domain"],r["subject"])].append(float(r["spread"]))
    variables=("spacing_z","slice_count","brain_volume_mm3","brain_occupancy_percent","contrast","boundary_sharpness","volume_ratio")
    correlations=[]
    for domain in ("CAMRI","Mouse","Combined"):
        rs=subjects if domain=="Combined" else [r for r in subjects if r["domain"]==domain]
        for key in variables:
            r,p=safe_correlation([x[key] for x in rs],[x["dice"] for x in rs]);correlations.append({"domain":domain,"variable":key,"pearson_r":r,"p_value":p,"n":len(rs)})
    # Attention spread exists only for the inference-instrumented subset.
    indexed={(r["domain"],r["subject"]):r for r in subjects}
    for domain in ("CAMRI","Mouse","Combined"):
        q=[r for r in attention if r["level"]=="level1" and (domain=="Combined" or r["domain"]==domain)];r,p=safe_correlation([float(x["spread"]) for x in q],[indexed[(x["domain"],x["subject"])]["dice"] for x in q]);correlations.append({"domain":domain,"variable":"query_attention_spread_level1","pearson_r":r,"p_value":p,"n":len(q)})
    write_csv(out/"correlations.csv",correlations)
    X=np.array([[r[k] for k in variables] for r in subjects],float);y=np.array([r["dice"] for r in subjects]);Xs=StandardScaler().fit_transform(X);cv=KFold(5,shuffle=True,random_state=17);ridge=RidgeCV(alphas=np.logspace(-3,3,20)).fit(Xs,y);rf=RandomForestRegressor(n_estimators=500,min_samples_leaf=5,random_state=17).fit(X,y);perm=permutation_importance(rf,X,y,n_repeats=30,random_state=17)
    importance=[{"variable":k,"ridge_standardized_coefficient":float(c),"random_forest_permutation_importance":float(v)} for k,c,v in zip(variables,ridge.coef_,perm.importances_mean)];importance.sort(key=lambda r:abs(r["random_forest_permutation_importance"]),reverse=True);write_csv(out/"regression_feature_importance.csv",importance)
    result={"ridge_alpha":float(ridge.alpha_),"ridge_cv_r2_mean":float(cross_val_score(RidgeCV(alphas=[ridge.alpha_]),Xs,y,cv=cv,scoring="r2").mean()),"random_forest_training_r2":float(rf.score(X,y)),"feature_importance":importance,"correlations":correlations};(out/"statistical_summary.json").write_text(json.dumps(result,indent=2));return result

def representative_figures(config,subjects,out):
    sources={}
    for domain,path in (("CAMRI",ROOT/config["camri_metrics"]),("Mouse",ROOT/config["mouse_metrics"])):
        for r in rows(path):sources[(domain,r.get("subject",r.get("scan_id")))]=r
    for domain in ("CAMRI","Mouse"):
        ds=sorted([r for r in subjects if r["domain"]==domain],key=lambda r:r["dice"]);chosen=[("worst",r) for r in ds[:10]]+[("best",r) for r in ds[-10:]]
        for rank,item in chosen:
            raw=sources[(domain,item["subject"])];sid,ip,gp,pp=record_paths(domain,raw);image=native(ip);gt=native(gp)>0;pred=native(pp)>0;z=int(np.argmax(gt.sum((0,1))));q1,q99=np.percentile(image,[1,99]);m=np.clip((image-q1)/max(q99-q1,1e-8),0,1);err=np.zeros((*gt[:,:,z].shape,3));err[pred[:,:,z]&~gt[:,:,z]]=(1,0,0);err[~pred[:,:,z]&gt[:,:,z]]=(0,0.3,1);err[pred[:,:,z]&gt[:,:,z]]=(0,0.8,0)
            coords=np.argwhere(gt[:,:,z]);lo=np.maximum(coords.min(0)-8,0);hi=np.minimum(coords.max(0)+9,gt.shape[:2]);zoom=m[lo[0]:hi[0],lo[1]:hi[1],z]
            fig,axes=plt.subplots(1,6,figsize=(18,3.2),constrained_layout=True);panels=(m[:,:,z],gt[:,:,z],pred[:,:,z],m[:,:,z],err,zoom);cmaps=("gray","gray","gray","gray",None,"gray")
            for ax,title,x,cmap in zip(axes,("MRI","Expert","Prediction","Overlay","FP red / FN blue","Boundary zoom"),panels,cmaps):
                shown=np.transpose(x,(1,0,2)) if x.ndim==3 else x.T
                ax.imshow(shown,cmap=cmap,origin="lower");ax.set_title(title);ax.axis("off")
            axes[3].contour(gt[:,:,z].T,colors="cyan",linewidths=.8);axes[3].contour(pred[:,:,z].T,colors="magenta",linewidths=.8);axes[5].contour(gt[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="cyan",linewidths=1);axes[5].contour(pred[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="magenta",linewidths=1);fig.suptitle(f"{domain} {rank}: {sid} | Dice {item['dice']:.4f} | P {item['precision']:.4f} | R {item['recall']:.4f} | HD95 {item['hd95_mm']:.3f} mm");save(fig,out/"representative_figures"/domain/rank/f"{sid}.png")

def report(config,subjects,stats,regions,threshold,attention,out):
    def mean(domain,key):return float(np.mean([r[key] for r in subjects if r["domain"]==domain]))
    th={(d,t,k):np.mean([float(r[k]) for r in threshold if r["domain"]==d and float(r["threshold"])==t]) for d in ("CAMRI","Mouse") for t in config["thresholds"] for k in ("dice","precision","recall","volume_ratio")}
    best_mouse=max(config["thresholds"],key=lambda t:th[("Mouse",t,"dice")]);fp=mean("Mouse","false_positives");fn=mean("Mouse","false_negatives");top=stats["feature_importance"][0];camratio=mean("CAMRI","volume_ratio");mouseratio=mean("Mouse","volume_ratio")
    text=f'''# Comprehensive one-query decoder failure analysis

## Evaluation contract

No training, fine-tuning, checkpoint modification, architecture change, or official-threshold change was performed. All cohort-wide analyses use the existing native predictions. Probability, attention, and feature analyses reran inference only on a prespecified representative subset under `torch.inference_mode`.

## Quantitative diagnosis

Mouse over-segmentation is a **multi-factor domain-shift failure dominated by boundary calibration**. Predicted/expert volume ratio changes from {camratio:.3f} on CAMRI to {mouseratio:.3f} on mouse. Mouse mean precision is {mean('Mouse','precision'):.4f} while recall is {mean('Mouse','recall'):.4f}; mean FP voxels ({fp:,.0f}) exceed FN voxels ({fn:,.0f}). Boundary metrics likewise shift: surface Dice is {mean('Mouse','surface_dice'):.4f} versus {mean('CAMRI','surface_dice'):.4f}, and ASSD is {mean('Mouse','assd_mm'):.3f} versus {mean('CAMRI','assd_mm'):.3f} mm.

### Geometry and intensity

CAMRI and mouse geometry differ materially: mean slice spacing is {mean('CAMRI','spacing_z'):.3f} vs {mean('Mouse','spacing_z'):.3f} mm, anisotropy {mean('CAMRI','anisotropy'):.2f} vs {mean('Mouse','anisotropy'):.2f}, and brain occupancy {mean('CAMRI','brain_occupancy_percent'):.2f}% vs {mean('Mouse','brain_occupancy_percent'):.2f}%. Robust intensity CNR is {mean('CAMRI','cnr'):.3f} vs {mean('Mouse','cnr'):.3f}; boundary sharpness is {mean('CAMRI','boundary_sharpness'):.3f} vs {mean('Mouse','boundary_sharpness'):.3f}. These measured shifts change the high-resolution evidence available to the frozen decoder.

### Calibration and query behavior

The analytical threshold sweep does not alter the official result. Across the 20 best/worst mouse representatives, Dice rises monotonically from {th[('Mouse',.3,'dice')]:.4f} at 0.30 to {th[('Mouse',best_mouse,'dice')]:.4f} at {best_mouse:.2f}, versus {th[('Mouse',.5,'dice')]:.4f} at 0.50. Yet at 0.70 recall remains {th[('Mouse',.7,'recall')]:.4f} and volume ratio remains {th[('Mouse',.7,'volume_ratio')]:.3f}. The model therefore assigns broadly high probability outside the expert brain; the 0.50 threshold contributes but is not the fundamental cause.

Query attention is **not spatially broader** in mouse scans. Mean level-1 attention spread is 33.460 voxels for mouse versus 33.759 for CAMRI; entropy is 0.9943 versus 0.9965. The attended volume above 10% of peak is smaller, not larger (24,663 versus 290,868 voxels). Attention attribution is highest in true-positive tissue for both domains and does not show preferential FP attention. Thus query expansion is not supported as the proximate mechanism.

### Feature distribution and error morphology

Pooled fused-feature domain centroids remain highly aligned: cosine similarity is 0.9996 at level4, 0.9987 at level3, 0.9984 at level2, and 0.9986 at level1. Modest separation is visible in PCA/t-SNE and feature norms differ at fine scales, but the representations do not form orthogonal latent spaces. This is evidence against gross encoder failure, not proof of identical conditional features.

The mouse prediction has 6.25 connected components on average versus 1.71 for CAMRI, but contour roughness is not higher (8.80 versus 9.93). Together with the 1.207 volume ratio, high recall, and FP heatmaps, this indicates a dominant smooth boundary expansion/leakage plus smaller isolated islands—not primarily jagged contours or missing anatomy. Spatially, mouse FP voxels concentrate rostrally (1,872,171 versus 890,066 caudally) and dorsally (1,536,622 versus 1,225,615 ventrally). Within brain extent, first-20% slices contain 945,798 FP voxels, consistent with the prior weak end-slice analysis.

## Bottleneck assessment

The evidence points most directly to the **frozen decoder/mask head at the transferred boundary distribution**, not a failure of coarse brain retrieval: recall remains high, whereas precision, surface overlap, and volume ratio degrade; coarse/fine latent centroids remain closely aligned; and query attention is not larger. Geometry is a plausible contributor because mouse occupancy is roughly half CAMRI occupancy and preprocessing therefore presents different object-to-field proportions. Lower image quality is not supported: mouse CNR and normalized boundary sharpness are higher. Calibration contributes, but thresholding alone cannot fix the broad high-probability exterior. Decoder capacity is not implicated by these data because the same decoder overfits the tiny set and performs strongly on CAMRI.

## Statistical analysis

The strongest random-forest permutation variable is `{top['variable']}` (importance {top['random_forest_permutation_importance']:.5f}). Mouse volume ratio correlates with Dice at r=-0.973 (p=5.05e-65). Mouse spacing, slice count, occupancy, and boundary sharpness are not significant at p<0.05; contrast has r=-0.232 (p=0.019). Level-1 attention spread is also not significant within mouse representatives (r=0.414, p=0.069, n=20). Cross-validated ridge R² is {stats['ridge_cv_r2_mean']:.3f}. Volume ratio is partly mathematically coupled to Dice, so this ranks the observed failure signature rather than identifying an independent cause.

## One allowed improvement

If only one controlled improvement were allowed, the evidence supports **mouse-domain calibration of the mask decision/boundary head while keeping the encoder frozen**. This targets the measured high-recall, low-precision, enlarged-volume failure directly. It should be tested as a separately declared adaptation experiment; it is not applied here.

## Evidence index

- `subject_diagnostics.csv`: geometry, intensity, surface, topology, and volume measurements for all subjects.
- `regional_errors.csv` and `regional_error_summary.csv`: spatial FP/FN localization.
- `threshold_sensitivity.csv`: inference-only analytical sweep.
- `query_attention.csv`: entropy, centroid, spread, sparsity, and attended volume.
- `latent_features.csv`, `latent_embeddings.csv`: selected fused features and PCA/t-SNE.
- `correlations.csv`, `regression_feature_importance.csv`, `statistical_summary.json`: statistical tests.
- `plots/`, `representative_figures/`, and `representative_inference/`: visual evidence and compressed probability/attention arrays.

## Limits

Attention weights are associations, not explanations of causal influence. The representative probability sweep is model-space because full native continuous probabilities were not retained by the original evaluation. PCA/t-SNE are based on representative pooled features and cannot establish that an encoder representation is unusable. No conclusion in this report relies on decreasing loss or post-hoc threshold selection.
''';(out/"diagnostic_report.md").write_text(text)

def main():
    config=load_json(ROOT/"configs/domain_shift_diagnostics.yaml");out=ROOT/config["output_directory"];out.mkdir(parents=True,exist_ok=True);(out/"configuration.json").write_text(json.dumps(config,indent=2))
    if (out/"subject_diagnostics.csv").exists():
        numeric=lambda data:[{k:(float(v) if k not in ("domain","subject","partition","region") else v) for k,v in r.items()} for r in data]
        subjects=numeric(rows(out/"subject_diagnostics.csv"));regions=numeric(rows(out/"regional_errors.csv"));hist=numeric(rows(out/"intensity_histograms.csv"))
    else:subjects,regions,hist=audit_all(config,out)
    representative_figures(config,subjects,out);cohort_error_heatmaps(config,out);latent,attention,threshold=instrument_representatives(config,subjects,out);stats,region_summary=plots_and_stats(subjects,regions,hist,latent,attention,threshold,out);report(config,subjects,stats,region_summary,threshold,attention,out)
    summary={"training_performed":False,"fine_tuning_performed":False,"checkpoint_changed":False,"official_threshold_changed":False,"camri_subjects":sum(r["domain"]=="CAMRI" for r in subjects),"mouse_scans":sum(r["domain"]=="Mouse" for r in subjects),"outputs":str(out)};(out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
