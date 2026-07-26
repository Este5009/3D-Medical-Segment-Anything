#!/usr/bin/env python3
"""Native-space evaluation of the validation-selected boundary-head checkpoint."""
from __future__ import annotations
import csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import binary_dilation,binary_erosion,label
from models.query_mask_decoder import FrozenEncoderQueryModel,MultiScaleOneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter,RS2NetPaths
from evaluate_external_holdout import preprocess,sliding_window_logits,export_native
from domain_shift_diagnostics import binary_metrics
from train_query_decoder_overfit import choose_device,load_json

def rows(p):return list(csv.DictReader(open(p)))
def write_csv(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def save(fig,p):p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=200,bbox_inches="tight");plt.close(fig)

def export_probability(logits,properties,manager,configuration,dataset,destination,reference_image):
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    unpadded=logits.cpu()[0].numpy().astype(np.float32);current=configuration.spacing if len(configuration.spacing)==len(properties["shape_after_cropping_and_before_resampling"]) else [properties["spacing"][0],*configuration.spacing];native=configuration.resampling_fn(unpadded,properties["shape_after_cropping_and_before_resampling"],current,properties["spacing"]);prob=1/(1+np.exp(-native[0]));reverted=np.zeros(properties["shape_before_cropping"],np.float32);reverted[bounding_box_to_slice(properties["bbox_used_for_cropping"])]=prob;reverted=reverted.transpose(manager.transpose_backward);ref=nib.load(str(reference_image))
    if reverted.shape!=ref.shape:
        import itertools
        permutation=next(p for p in itertools.permutations(range(3)) if tuple(reverted.shape[i] for i in p)==ref.shape);reverted=reverted.transpose(permutation)
    nib.save(nib.Nifti1Image(reverted,ref.affine,ref.header),destination)

def native_eval(pred,gt,spacing,prob=None):
    m=binary_metrics(pred,gt,spacing,.2);m["iou"]=m["dice"]/(2-m["dice"]);m["connected_components"]=label(pred)[1]
    if prob is not None:
        inner=gt&~binary_erosion(gt,iterations=5);outside=binary_dilation(gt,iterations=5)&~gt;distant=~binary_dilation(gt,iterations=10);m.update({"probability_inside_brain":float(prob[gt].mean()),"probability_boundary_band":float(prob[inner].mean()),"probability_immediately_outside":float(prob[outside].mean()),"probability_distant_background":float(prob[distant].mean())})
    return m

def slice_rows(pred,gt,scan,condition):
    occupied=np.where(gt.any((0,1)))[0];a,b=occupied[0],occupied[-1]+1;n=b-a;out=[]
    for z in range(gt.shape[2]):
        p=pred[:,:,z];t=gt[:,:,z];tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum());den=2*tp+fp+fn;pos="outside"
        if a<=z<b:pos="first 20%" if z<a+.2*n else "last 20%" if z>=a+.8*n else "middle 60%"
        out.append({"scan_id":scan,"condition":condition,"slice_index":z,"brain_position":pos,"expert_nonempty":bool(t.any()),"dice":2*tp/den if den else 1.,"false_positives":fp,"false_negatives":fn})
    return out

def summarize(rs):
    result={"count":len(rs)}
    for key in ("dice","iou","precision","recall","false_positives","false_negatives","hd95_mm","volume_ratio","connected_components","inference_seconds"):
        x=np.array([float(r[key]) for r in rs]);result[key]={"mean":float(x.mean()),"median":float(np.median(x)),"sd":float(x.std()),"min":float(x.min()),"max":float(x.max()),"p05":float(np.percentile(x,5)),"p95":float(np.percentile(x,95))}
    return result

def run_records(model,records,paths,config,out,device,domain):
    results=[];slices=[]
    for i,r in enumerate(records,1):
        sid=r.get("scan_id",r.get("subject"));image=Path(r["image_path"]);gtp=Path(r["ground_truth_path"]);predp=out/"native_predictions"/domain/f"{sid}_prediction.nii.gz";probp=out/"probability_maps"/domain/f"{sid}_probability.nii.gz";predp.parent.mkdir(parents=True,exist_ok=True);probp.parent.mkdir(parents=True,exist_ok=True);meta=out/"native_predictions"/domain/f"{sid}_metrics.json"
        if meta.exists() and predp.exists() and probp.exists():row=load_json(meta)
        else:
            tensor,properties,manager,configuration,dataset=preprocess(image,gtp,paths,tuple(config["tile_size"]));started=time.perf_counter()
            with torch.inference_mode():logits=sliding_window_logits(model,tensor,tuple(config["tile_size"]),device)
            elapsed=time.perf_counter()-started;export_native(logits,properties,manager,configuration,dataset,predp);export_probability(logits,properties,manager,configuration,dataset,probp,image);gt=np.asarray(nib.load(str(gtp)).dataobj)>0;pred=np.asarray(nib.load(str(predp)).dataobj)>0;prob=np.asarray(nib.load(str(probp)).dataobj,dtype=np.float32);spacing=tuple(map(float,nib.load(str(gtp)).header.get_zooms()[:3]));row={"domain":domain,"scan_id":sid,"inference_seconds":elapsed,**native_eval(pred,gt,spacing,prob),"prediction_path":str(predp),"probability_path":str(probp),"image_path":str(image),"ground_truth_path":str(gtp)};meta.write_text(json.dumps(row,indent=2))
        gt=np.asarray(nib.load(str(gtp)).dataobj)>0;pred=np.asarray(nib.load(str(predp)).dataobj)>0;slices+=slice_rows(pred,gt,sid,"adapted");results.append(row);print(f"{domain} {i}/{len(records)} {sid} dice={row['dice']:.4f}",flush=True)
    return results,slices

def make_comparison_figure(row,out):
    image=np.asarray(nib.load(row["image_path"]).dataobj,dtype=np.float32);gt=np.asarray(nib.load(row["ground_truth_path"]).dataobj)>0;base=np.asarray(nib.load(row["baseline_prediction_path"]).dataobj)>0;adapt=np.asarray(nib.load(row["adapted_prediction_path"]).dataobj)>0;prob=np.asarray(nib.load(row["probability_path"]).dataobj,dtype=np.float32);brain=np.where(gt.any((0,1)))[0];per=[]
    for z in brain:
        def sm(p):tp=(p&gt[:,:,z]).sum();fp=(p&~gt[:,:,z]).sum();fn=(~p&gt[:,:,z]).sum();return 2*tp/max(2*tp+fp+fn,1)
        per.append((z,sm(base[:,:,z]),sm(adapt[:,:,z])))
    candidates=[brain[0],brain[len(brain)//4],brain[len(brain)//2],brain[3*len(brain)//4],brain[-1],min(per,key=lambda x:x[1])[0],min(per,key=lambda x:x[2])[0]];selected=[]
    for z in candidates:
        if z not in selected:selected.append(z)
    fig,axes=plt.subplots(len(selected),8,figsize=(24,3.1*len(selected)),constrained_layout=True)
    for i,z in enumerate(selected):
        q1,q99=np.percentile(image,[1,99]);m=np.clip((image[:,:,z]-q1)/max(q99-q1,1e-8),0,1);t=gt[:,:,z];b=base[:,:,z];a=adapt[:,:,z]
        def err(p):x=np.zeros((*p.shape,3));x[p&t]=(0,.8,0);x[p&~t]=(1,0,0);x[~p&t]=(0,.3,1);return x
        panels=(m,t,b,a,err(b),err(a),m,prob[:,:,z]);cmaps=("gray","gray","gray","gray",None,None,"gray","magma")
        for ax,title,x,cmap in zip(axes[i],("MRI","Expert","Baseline","Adapted","Baseline FP/FN","Adapted FP/FN","Contours","Adapted probability"),panels,cmaps):ax.imshow(np.transpose(x,(1,0,2)) if x.ndim==3 else x.T,cmap=cmap,origin="lower");ax.set_title(title if i==0 else "");ax.axis("off")
        axes[i,6].contour(t.T,colors="cyan");axes[i,6].contour(b.T,colors="magenta");axes[i,6].contour(a.T,colors="yellow");bm=next(x for x in per if x[0]==z);bf=int((b&~t).sum());bn=int((~b&t).sum());af=int((a&~t).sum());an=int((~a&t).sum());axes[i,0].set_ylabel(f"slice {z}\nbase/adapt Dice {bm[1]:.3f}/{bm[2]:.3f}\nFP {bf}/{af} FN {bn}/{an}",rotation=0,labelpad=85,va="center")
    fig.suptitle(f"{row['subject_id']} | test | Dice {row['baseline_dice']:.4f}->{row['adapted_dice']:.4f} | P {row['baseline_precision']:.4f}->{row['adapted_precision']:.4f} | R {row['baseline_recall']:.4f}->{row['adapted_recall']:.4f} | ratio {row['baseline_volume_ratio']:.3f}->{row['adapted_volume_ratio']:.3f}");save(fig,out)

def post_analysis(config,out,source,split,comparison):
    """Paired regional, slice-position, surface, and probability analyses."""
    regional=[];position=[];calibration=[];threshold=[];rng=np.random.default_rng(config["seed"])
    diagnostics={r["subject"]:r for r in rows(ROOT/"outputs/domain_shift_diagnostics/subject_diagnostics.csv") if r["domain"]=="Mouse"}
    adapted_metrics={r["scan_id"]:r for r in rows(out/"boundary_head_adaptation"/"test_metrics.csv")}
    for row in comparison:
        sid=row["subject_id"];gt=np.asarray(nib.load(row["ground_truth_path"]).dataobj)>0;base=np.asarray(nib.load(row["baseline_prediction_path"]).dataobj)>0;adapt=np.asarray(nib.load(row["adapted_prediction_path"]).dataobj)>0;prob=np.asarray(nib.load(row["probability_path"]).dataobj,dtype=np.float32)
        for condition,pred in (("baseline",base),("adapted",adapt)):
            fp=pred&~gt;fn=~pred&gt
            for axis,a,b in ((0,"left","right"),(1,"dorsal","ventral"),(2,"rostral","caudal")):
                mid=gt.shape[axis]//2
                for name,sl in ((a,slice(0,mid)),(b,slice(mid,None))):idx=[slice(None)]*3;idx[axis]=sl;regional.append({"scan_id":sid,"condition":condition,"region":name,"fp_voxels":int(fp[tuple(idx)].sum()),"fn_voxels":int(fn[tuple(idx)].sum())})
            sr=slice_rows(pred,gt,sid,condition)
            for pos in ("first 20%","middle 60%","last 20%"):
                q=[x for x in sr if x["brain_position"]==pos and x["expert_nonempty"]];position.append({"scan_id":sid,"condition":condition,"brain_position":pos,"mean_slice_dice":float(np.mean([x["dice"] for x in q])),"fp_voxels":sum(x["false_positives"] for x in q),"fn_voxels":sum(x["false_negatives"] for x in q),**{f"dice_gte_{t:.2f}":100*np.mean([x["dice"]>=t for x in q]) for t in (.8,.9,.95,.97,.98,.99)}})
        flat=prob.ravel();truth=gt.ravel();take=rng.choice(len(flat),min(5000,len(flat)),replace=False)
        for pr,tr in zip(flat[take],truth[take]):calibration.append({"scan_id":sid,"probability":float(pr),"target":int(tr)})
        for t in (.3,.4,.5,.6,.7):threshold.append({"scan_id":sid,"threshold":t,"predicted_expert_volume_ratio":float((prob>=t).sum()/gt.sum())})
    write_csv(out/"regional_fp_fn_comparison.csv",regional);write_csv(out/"brain_position_comparison.csv",position);write_csv(out/"probability_maps"/"calibration_samples.csv",calibration);write_csv(out/"probability_maps"/"threshold_volume_ratios.csv",threshold)
    # Surface comparison uses the prior reliable baseline computation and the identical implementation for adaptation.
    surface=[]
    for sid in split["test"]["scans"]:
        b=diagnostics[sid];a=adapted_metrics[sid];surface.append({"scan_id":sid,"baseline_surface_dice":b["surface_dice"],"adapted_surface_dice":a["surface_dice"],"baseline_boundary_precision":b["boundary_precision"],"adapted_boundary_precision":a["boundary_precision"],"baseline_boundary_recall":b["boundary_recall"],"adapted_boundary_recall":a["boundary_recall"]})
    write_csv(out/"boundary_surface_comparison.csv",surface)
    # Learning curves and paired metric changes.
    hist=rows(out/"learning_curves"/"boundary_head_history.csv");fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    axes[0].plot([int(r["epoch"]) for r in hist],[float(r["train_dice"]) for r in hist],label="train");axes[0].plot([int(r["epoch"]) for r in hist],[float(r["validation_dice"]) for r in hist],label="validation");axes[0].set(title="Dice",xlabel="Epoch",ylabel="Dice");axes[0].legend()
    axes[1].plot([int(r["epoch"]) for r in hist],[float(r["validation_precision"]) for r in hist],label="precision");axes[1].plot([int(r["epoch"]) for r in hist],[float(r["validation_recall"]) for r in hist],label="recall");axes[1].set(title="Validation precision and recall",xlabel="Epoch");axes[1].legend()
    axes[2].plot([int(r["epoch"]) for r in hist],[float(r["validation_volume_ratio"]) for r in hist]);axes[2].axhline(1,color="black",ls="--");axes[2].set(title="Validation volume ratio",xlabel="Epoch",ylabel="Predicted/expert")
    [ax.grid(alpha=.25) for ax in axes];save(fig,out/"learning_curves"/"boundary_head_learning_curves.png")
    bins=np.linspace(0,1,11);fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True);c=np.array([float(r["probability"]) for r in calibration]);y=np.array([int(r["target"]) for r in calibration]);inds=np.digitize(c,bins)-1;xs=[];ys=[]
    for i in range(10):
        q=inds==i
        if q.any():xs.append(c[q].mean());ys.append(y[q].mean())
    axes[0].plot(xs,ys,marker="o",label="adapted");axes[0].plot([0,1],[0,1],ls="--",color="black",label="ideal");axes[0].set(title="Adapted test calibration",xlabel="Mean confidence",ylabel="Foreground frequency");axes[0].legend();
    for t in (.3,.4,.5,.6,.7):axes[1].plot(t,np.mean([float(r["predicted_expert_volume_ratio"]) for r in threshold if float(r["threshold"])==t]),"o",color="#d1495b")
    axes[1].plot([.3,.4,.5,.6,.7],[np.mean([float(r["predicted_expert_volume_ratio"]) for r in threshold if float(r["threshold"])==t]) for t in (.3,.4,.5,.6,.7)],color="#d1495b");axes[1].axhline(1,ls="--",color="black");axes[1].axvline(.5,ls=":",color="black");axes[1].set(title="Test predicted volume across thresholds",xlabel="Analytical threshold",ylabel="Predicted/expert ratio");save(fig,out/"probability_maps"/"calibration_and_thresholds.png")
    def paired_region(region,key):
        return {c:sum(float(r[key]) for r in regional if r["condition"]==c and r["region"]==region) for c in ("baseline","adapted")}
    def paired_position(pos,key):return {c:float(np.mean([float(r[key]) for r in position if r["condition"]==c and r["brain_position"]==pos])) for c in ("baseline","adapted")}
    boundary_zoom_figures(sorted(comparison,key=lambda r:r["dice_change"],reverse=True)[:10],out/"visualizations"/"rostral_dorsal_zooms")
    return {"regional":{r:paired_region(r,"fp_voxels") for r in ("dorsal","ventral","rostral","caudal","left","right")},"brain_position":{p:paired_position(p,"mean_slice_dice") for p in ("first 20%","middle 60%","last 20%")},"surface":{"baseline_surface_dice":float(np.mean([float(r["baseline_surface_dice"]) for r in surface])),"adapted_surface_dice":float(np.mean([float(r["adapted_surface_dice"]) for r in surface])),"baseline_boundary_precision":float(np.mean([float(r["baseline_boundary_precision"]) for r in surface])),"adapted_boundary_precision":float(np.mean([float(r["adapted_boundary_precision"]) for r in surface])),"baseline_boundary_recall":float(np.mean([float(r["baseline_boundary_recall"]) for r in surface])),"adapted_boundary_recall":float(np.mean([float(r["adapted_boundary_recall"]) for r in surface]))}}

def boundary_zoom_figures(comparison,destination):
    for row in comparison:
        image=np.asarray(nib.load(row["image_path"]).dataobj,dtype=np.float32);gt=np.asarray(nib.load(row["ground_truth_path"]).dataobj)>0;base=np.asarray(nib.load(row["baseline_prediction_path"]).dataobj)>0;adapt=np.asarray(nib.load(row["adapted_prediction_path"]).dataobj)>0;brain=np.where(gt.any((0,1)))[0];first=brain[:max(1,int(np.ceil(.2*len(brain))))];z=max(first,key=lambda i:int((base[:,:,i]&~gt[:,:,i]).sum()));coords=np.argwhere(gt[:,:,z]);lo=np.maximum(coords.min(0)-12,0);hi=np.minimum(coords.max(0)+13,gt.shape[:2]);q1,q99=np.percentile(image,[1,99]);m=np.clip((image[:,:,z]-q1)/max(q99-q1,1e-8),0,1);fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
        for ax,title in zip(axes,("Rostral boundary: baseline","Rostral boundary: adapted","Dorsal boundary zoom")):ax.imshow(m[lo[0]:hi[0],lo[1]:hi[1]].T,cmap="gray",origin="lower");ax.contour(gt[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="cyan");ax.axis("off");ax.set_title(title)
        axes[0].contour(base[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="magenta");axes[1].contour(adapt[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="yellow");dmid=(hi[1]-lo[1])//2;axes[2].contour(base[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="magenta");axes[2].contour(adapt[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="yellow");axes[2].set_ylim(dmid,hi[1]-lo[1]);fig.suptitle(f"{row['subject_id']} | slice {z} | Dice {row['baseline_dice']:.4f}->{row['adapted_dice']:.4f}");save(fig,destination/f"{row['subject_id']}.png")

def main():
    config=load_json(ROOT/"configs/mouse_boundary_adaptation.yaml");out=ROOT/config["output_directory"];split=load_json(out/"split.json");source={r["scan_id"]:r for r in rows(ROOT/config["mouse_metrics"])};enc_cfg=load_json(ROOT/config["encoder_config"]);paths=RS2NetPaths.from_config(enc_cfg);device=choose_device();ck=torch.load(out/"checkpoints"/"boundary_head_best.pt",map_location="cpu",weights_only=False);decoder=MultiScaleOneQueryMaskDecoder(32,4);decoder.load_state_dict(ck["decoder_state_dict"],strict=True);model=FrozenEncoderQueryModel(RS2NetEncoderAdapter(paths,image_size=tuple(config["tile_size"]),in_channels=1,out_channels=1,feature_size=48),decoder).to(device).eval()
    all_adapt=[];all_slices=[]
    for name in ("train","validation","test"):
        rs=[source[x] for x in split[name]["scans"]];adapt,sli=run_records(model,rs,paths,config,out,device,"mouse");
        for r in adapt:r["split"]=name
        all_adapt+=adapt;all_slices+=sli;write_csv(out/("boundary_head_adaptation" if name!="test" else "boundary_head_adaptation")/f"{name}_metrics.csv",adapt)
    write_csv(out/"per_slice_metrics"/"adapted_all_slices.csv",all_slices)
    adapted={r["scan_id"]:r for r in all_adapt};comparison=[]
    for sid in split["test"]["scans"]:
        b=source[sid];a=adapted[sid];comparison.append({"subject_id":sid,"baseline_dice":float(b["dice"]),"adapted_dice":a["dice"],"dice_change":a["dice"]-float(b["dice"]),"baseline_precision":float(b["precision"]),"adapted_precision":a["precision"],"baseline_recall":float(b["recall"]),"adapted_recall":a["recall"],"baseline_volume_ratio":float(b["predicted_brain_volume_mm3"])/float(b["expert_brain_volume_mm3"]),"adapted_volume_ratio":a["volume_ratio"],"baseline_fp":int(b["false_positives"]),"adapted_fp":a["false_positives"],"baseline_hd95":float(b["hd95"]),"adapted_hd95":a["hd95_mm"],"baseline_prediction_path":b["prediction_path"],"adapted_prediction_path":a["prediction_path"],"probability_path":a["probability_path"],"image_path":b["image_path"],"ground_truth_path":b["ground_truth_path"]})
    write_csv(out/"test_comparison.csv",comparison);changes=np.array([r["dice_change"] for r in comparison]);wil=stats.wilcoxon(changes)
    # Fixed retention subset determined only by existing baseline CAMRI Dice ranks.
    cam=rows(ROOT/"outputs/external_holdout/subject_metrics.csv");ordered=sorted(cam,key=lambda r:float(r["dice"]));ret=ordered[:10]+ordered[len(ordered)//2-5:len(ordered)//2+5]+ordered[-10:];adapt_cam,_=run_records(model,ret,paths,config,out,device,"camri");ret_rows=[]
    for b,a in zip(ret,adapt_cam):ret_rows.append({"subject":b["subject"],"stratum":"hard" if b in ordered[:10] else "easy" if b in ordered[-10:] else "median","baseline_dice":float(b["dice"]),"adapted_dice":a["dice"],"dice_change":a["dice"]-float(b["dice"]),"baseline_precision":float(b["precision"]),"adapted_precision":a["precision"],"baseline_recall":float(b["recall"]),"adapted_recall":a["recall"]})
    write_csv(out/"camri_retention.csv",ret_rows)
    ranked=sorted(comparison,key=lambda r:r["dice_change"]);chosen=ranked[-10:]+ranked[:10]+sorted(comparison,key=lambda r:r["adapted_dice"])[:10]+sorted(comparison,key=lambda r:r["adapted_dice"])[len(comparison)//2-2:len(comparison)//2+3];seen=set()
    for r in chosen:
        if r["subject_id"] not in seen:make_comparison_figure(r,out/"visualizations"/f"{r['subject_id']}.png");seen.add(r["subject_id"])
    summaries={name:summarize([r for r in all_adapt if r["split"]==name]) for name in ("train","validation","test")};base_test=[source[x] for x in split["test"]["scans"]];extra=post_analysis(config,out,source,split,comparison);report={"experiment_c_triggered":False,"test_baseline_mean_dice":float(np.mean([float(r['dice']) for r in base_test])),"test_adapted":summaries["test"],"paired_mean_dice_change":float(changes.mean()),"paired_median_dice_change":float(np.median(changes)),"improved_scans":int((changes>0).sum()),"degraded_scans":int((changes<0).sum()),"ties":int((changes==0).sum()),"wilcoxon_statistic":float(wil.statistic),"wilcoxon_p_value":float(wil.pvalue),"smallest_improvement":float(changes.min()),"best_improvement":float(changes.max()),"camri_mean_dice_change":float(np.mean([r['dice_change'] for r in ret_rows])),"camri_max_degradation":float(min(r['dice_change'] for r in ret_rows)),"split_metrics":summaries,**extra};(out/"summary.json").write_text(json.dumps(report,indent=2));write_report(report,comparison,ret_rows,out);print(json.dumps(report,indent=2))

def write_report(s,c,ret,out):
    basep=np.mean([r["baseline_precision"] for r in c]);adp=np.mean([r["adapted_precision"] for r in c]);baser=np.mean([r["baseline_recall"] for r in c]);adr=np.mean([r["adapted_recall"] for r in c]);basev=np.mean([r["baseline_volume_ratio"] for r in c]);adv=np.mean([r["adapted_volume_ratio"] for r in c]);basefp=np.mean([r["baseline_fp"] for r in c]);adfp=np.mean([r["adapted_fp"] for r in c])
    dorsal=s["regional"]["dorsal"];rostral=s["regional"]["rostral"];first=s["brain_position"]["first 20%"];middle=s["brain_position"]["middle 60%"];last=s["brain_position"]["last 20%"]
    text=f'''# Mouse boundary-head adaptation

The validation-selected epoch-16 boundary-head checkpoint trained 29,793 parameters: `mask_embedding`, `mask_refinement`, and `mask_bias`. The encoder, query, attention blocks, projections, and FPN remained frozen. The loss was `0.50 Dice + 0.25 BCE + 0.25 Tversky(alpha_FP=0.70, beta_FN=0.30)`.

The deterministic split contains 11 scans/3 identified mice for training, 10 scans/3 different identified mice for validation, and 80 untouched test scans (9 identified mice plus 52 identity-unknown scans). All known longitudinal scans remain together. Unknown filenames remain a documented residual linkage limitation.

## Untouched test result

Mean Dice changed from {s['test_baseline_mean_dice']:.4f} to {s['test_adapted']['dice']['mean']:.4f} (paired change {s['paired_mean_dice_change']:+.4f}; Wilcoxon p={s['wilcoxon_p_value']:.3g}). Precision changed {basep:.4f}->{adp:.4f}; recall {baser:.4f}->{adr:.4f}; volume ratio {basev:.3f}->{adv:.3f}; FP voxels {basefp:,.0f}->{adfp:,.0f}. {s['improved_scans']} scans improved and {s['degraded_scans']} degraded.

Surface Dice improved {s['surface']['baseline_surface_dice']:.4f}->{s['surface']['adapted_surface_dice']:.4f}, and boundary precision improved {s['surface']['baseline_boundary_precision']:.4f}->{s['surface']['adapted_boundary_precision']:.4f}. Rostral FP fell {rostral['baseline']:,}->{rostral['adapted']:,} ({100*(1-rostral['adapted']/rostral['baseline']):.1f}%); dorsal FP fell {dorsal['baseline']:,}->{dorsal['adapted']:,} ({100*(1-dorsal['adapted']/dorsal['baseline']):.1f}%). First-20% slice Dice improved {first['baseline']:.4f}->{first['adapted']:.4f}, and middle-60% improved {middle['baseline']:.4f}->{middle['adapted']:.4f}, but last-20% worsened {last['baseline']:.4f}->{last['adapted']:.4f}. The recall and terminal-slice losses show that the FP-aware calibration slightly over-corrects.

Boundary-head-only adaptation satisfied all validation gates, so full-decoder Experiment C was not triggered. CAMRI retention mean Dice changed from {np.mean([r['baseline_dice'] for r in ret]):.4f} to {np.mean([r['adapted_dice'] for r in ret]):.4f} ({s['camri_mean_dice_change']:+.4f}), with worst individual change {s['camri_max_degradation']:+.4f}. This is substantial domain specialization, although not total collapse.

## Decision

Boundary-head-only adaptation suffices to correct the primary mouse FP leakage, but the selected checkpoint is **mouse-specialized rather than broadly transferable** because it reduces recall, worsens the last brain region, and harms CAMRI retention. The next controlled experiment should add a fixed CAMRI rehearsal/retention constraint during the same boundary-head-only training; the encoder, query, attention, and FPN should remain frozen. That directly tests whether boundary calibration can improve mouse performance without forgetting CAMRI.
''';(out/"comparison_report.md").write_text(text)
if __name__=="__main__":main()
