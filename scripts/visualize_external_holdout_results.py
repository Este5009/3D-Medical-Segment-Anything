#!/usr/bin/env python3
"""Publication-style figures from saved external-holdout artifacts only.

No model module or checkpoint is imported. All metrics are recomputed from the
saved native prediction and expert NIfTI masks before any figure is created.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/external_holdout"
OUT = SOURCE / "visualizations"
FONT_PATH = "/System/Library/Fonts/Supplemental/Verdana.ttf"
BOLD_PATH = "/System/Library/Fonts/Supplemental/Verdana Bold.ttf"

BLUE = (35, 105, 230)       # expert only / false negative
RED = (225, 45, 45)         # prediction only / false positive
GREEN = (35, 190, 80)       # overlap
CYAN = (0, 235, 255)        # expert contour
MAGENTA = (255, 40, 220)    # prediction contour
WHITE = (245, 245, 245)
BLACK = (12, 12, 12)
GRAY = (105, 105, 105)


def font(size, bold=False):
    return ImageFont.truetype(BOLD_PATH if bold else FONT_PATH, size)


def recompute_binary_metrics(prediction, target):
    prediction = prediction.astype(bool); target = target.astype(bool)
    tp = int((prediction & target).sum()); fp = int((prediction & ~target).sum()); fn = int((~prediction & target).sum())
    dice = (2*tp+1e-5)/(2*tp+fp+fn+1e-5); iou=(tp+1e-5)/(tp+fp+fn+1e-5)
    precision=(tp+1e-5)/(tp+fp+1e-5); recall=(tp+1e-5)/(tp+fn+1e-5)
    return {"dice":dice,"iou":iou,"precision":precision,"recall":recall,"false_positives":fp,"false_negatives":fn}


def recompute_slices(prediction, target):
    rows=[]
    for index in range(target.shape[2]):
        pred=prediction[:,:,index].astype(bool);truth=target[:,:,index].astype(bool)
        tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());denom=2*tp+fp+fn;union=tp+fp+fn
        # Match the saved evaluation's exact unsmoothed per-slice convention.
        metric={"dice":2*tp/denom if denom else 1.0,"iou":tp/union if union else 1.0,"false_positives":fp,"false_negatives":fn}
        rows.append({"slice_index":index,**metric,"error":fp+fn,"brain_voxels":int(truth.sum())})
    return rows


def sitk_geometry(path):
    image=sitk.ReadImage(str(path));return {"size":list(image.GetSize()),"spacing":list(image.GetSpacing()),"origin":list(image.GetOrigin()),"direction":list(image.GetDirection())}


def geometry_equal(first,second,tolerance=1e-5):
    return first["size"]==second["size"] and all(np.allclose(first[key],second[key],atol=tolerance) for key in ("spacing","origin","direction"))


def audit_and_load():
    rows=list(csv.DictReader((SOURCE/"subject_metrics.csv").open()))
    records=[];audit={"subjects":{},"failures":[],"metric_tolerance":1e-6,"slice_axis":2}
    for row in rows:
        subject=row["subject"];image_path=Path(row["image_path"]);target_path=Path(row["ground_truth_path"]);prediction_path=Path(row["prediction_path"])
        image_obj=nib.load(str(image_path));target_obj=nib.load(str(target_path));prediction_obj=nib.load(str(prediction_path))
        image=np.asarray(image_obj.dataobj,dtype=np.float32);target=np.asarray(target_obj.dataobj)>0;prediction=np.asarray(prediction_obj.dataobj)>0
        shapes={"image":list(image.shape),"target":list(target.shape),"prediction":list(prediction.shape)}
        geometries={"image":sitk_geometry(image_path),"target":sitk_geometry(target_path),"prediction":sitk_geometry(prediction_path)}
        axcodes={"image":nib.aff2axcodes(image_obj.affine),"target":nib.aff2axcodes(target_obj.affine),"prediction":nib.aff2axcodes(prediction_obj.affine)}
        if not (image.shape==target.shape==prediction.shape):audit["failures"].append(f"sub-{subject}: shape mismatch {shapes}");continue
        if not geometry_equal(geometries["image"],geometries["target"]):audit["failures"].append(f"sub-{subject}: image/target physical geometry mismatch");continue
        if not geometry_equal(geometries["image"],geometries["prediction"]):audit["failures"].append(f"sub-{subject}: image/prediction physical geometry mismatch");continue
        calculated=recompute_binary_metrics(prediction,target)
        for key in ("dice","iou","precision","recall"):
            if not np.isclose(calculated[key],float(row[key]),atol=1e-6):audit["failures"].append(f"sub-{subject}: saved {key} mismatch");break
        saved_slices=list(csv.DictReader((SOURCE/"per_slice"/f"sub-{subject}_slices.csv").open()));calculated_slices=recompute_slices(prediction,target)
        if len(saved_slices)!=len(calculated_slices):audit["failures"].append(f"sub-{subject}: slice row count mismatch");continue
        for saved,current in zip(saved_slices,calculated_slices):
            if not (np.isclose(float(saved["dice"]),current["dice"],atol=1e-6) and np.isclose(float(saved["iou"]),current["iou"],atol=1e-6) and int(saved["false_positives"])==current["false_positives"] and int(saved["false_negatives"])==current["false_negatives"]):
                audit["failures"].append(f"sub-{subject}: per-slice metric mismatch at {current['slice_index']}");break
        audit["subjects"][subject]={"shape":shapes["image"],"spacing":geometries["image"]["spacing"],"orientation_codes":[str(x) for x in axcodes["image"]],"geometry_matches":True,"metrics_match":True,"nonempty_image":bool(np.ptp(image)>0),"nonempty_target":bool(target.any()),"nonempty_prediction":bool(prediction.any())}
        records.append({**row,"dice":calculated["dice"],"iou":calculated["iou"],"precision":calculated["precision"],"recall":calculated["recall"],"false_positives":calculated["false_positives"],"false_negatives":calculated["false_negatives"],"brain_volume_mm3":float(row["brain_volume_mm3"]),"slice_count":int(row["slice_count"]),"spacing":tuple(geometries["image"]["spacing"]),"image":image,"target":target,"prediction":prediction,"slices":calculated_slices})
    audit["subjects_validated"]=len(records);audit["orientation_codes_seen"]=sorted({tuple(v["orientation_codes"]) for v in audit["subjects"].values()});audit["slice_axis_explanation"]="NIfTI array axis 2; consistent native superior/inferior acquisition slice index within each subject"
    if audit["failures"]:raise RuntimeError("Visualization audit failed:\n"+"\n".join(audit["failures"][:20]))
    return records,audit


def brain_crop(target,margin=8):
    locations=np.argwhere(target)
    if len(locations)==0:return (slice(0,target.shape[0]),slice(0,target.shape[1]))
    low=locations[:,:2].min(0);high=locations[:,:2].max(0)+1
    low=np.maximum(low-margin,0);high=np.minimum(high+margin,target.shape[:2])
    return slice(int(low[0]),int(high[0])),slice(int(low[1]),int(high[1]))


def robust_mri(slice_array,whole_volume):
    foreground=whole_volume[np.isfinite(whole_volume)&(whole_volume!=0)]
    low,high=np.percentile(foreground,(1,99)) if len(foreground) else (float(slice_array.min()),float(slice_array.max()))
    return np.clip((slice_array-low)/max(float(high-low),1e-8),0,1)


def contour(mask):
    return mask.astype(bool)^binary_erosion(mask.astype(bool))


def five_panels(record,index,crop,panel_size=300):
    xs,ys=crop;mri=robust_mri(record["image"][:,:,index],record["image"])[xs,ys];gt=record["target"][:,:,index][xs,ys];pred=record["prediction"][:,:,index][xs,ys]
    gray=(mri*255).astype(np.uint8);base=np.stack([gray]*3,-1)
    expert=np.zeros_like(base);expert[gt]=BLUE;prediction=np.zeros_like(base);prediction[pred]=RED
    compared=base.copy();compared[contour(gt)]=CYAN;compared[contour(pred)]=MAGENTA
    error=np.zeros_like(base);error[gt&pred]=GREEN;error[pred&~gt]=RED;error[gt&~pred]=BLUE
    return [fit_panel(x,panel_size) for x in (base,expert,prediction,compared,error)]


def fit_panel(array,size):
    image=Image.fromarray(array);image.thumbnail((size,size),Image.Resampling.NEAREST)
    canvas=Image.new("RGB",(size,size),BLACK);canvas.paste(image,((size-image.width)//2,(size-image.height)//2));return canvas


def informative_slices(record):
    brain=[row["slice_index"] for row in record["slices"] if row["brain_voxels"]>0];first,last=brain[0],brain[-1];span=last-first
    targets=[first,round(first+.25*span),round(first+.5*span),round(first+.75*span),last,min((r for r in record["slices"] if r["brain_voxels"]>0),key=lambda r:r["dice"])["slice_index"]]
    selected=[]
    for target in targets:
        candidates=sorted(brain,key=lambda value:(abs(value-target),-record["slices"][value]["error"]))
        selected.append(next(value for value in candidates if value not in selected))
    return selected


def legend_strip(width):
    image=Image.new("RGB",(width,55),WHITE);draw=ImageDraw.Draw(image);items=[("Expert-only / FN",BLUE),("Prediction-only / FP",RED),("Correct overlap",GREEN),("Expert contour",CYAN),("Prediction contour",MAGENTA)];x=18
    for label,color in items:draw.rectangle((x,15,x+22,37),fill=color);draw.text((x+30,14),label,font=font(15),fill=BLACK);x+=width//5
    return image


def subject_overview(record,destination):
    selected=informative_slices(record);crop=brain_crop(record["target"].any(axis=2));panel=270;left=245;header=125;row_height=panel+8;width=left+5*panel;height=header+len(selected)*row_height+60
    canvas=Image.new("RGB",(width,height),WHITE);draw=ImageDraw.Draw(canvas);draw.text((25,18),f"Subject sub-{record['subject']}  |  Volumetric Dice {record['dice']:.4f}",font=font(28,True),fill=BLACK)
    labels=["MRI","Expert mask","Decoder prediction","Contours on MRI","Error map"]
    for column,label in enumerate(labels):draw.text((left+column*panel+12,78),label,font=font(18,True),fill=BLACK)
    for row_index,index in enumerate(selected):
        metric=record["slices"][index];y=header+row_index*row_height
        draw.multiline_text((12,y+70),f"Slice {index}\nDice {metric['dice']:.4f}\nFP {metric['false_positives']}\nFN {metric['false_negatives']}",font=font(17,True),fill=BLACK,spacing=8)
        for column,image in enumerate(five_panels(record,index,crop,panel)):canvas.paste(image,(left+column*panel,y))
    canvas.paste(legend_strip(width),(0,height-55));canvas.save(destination,optimize=True)


def acquisition_group(record):return "64-slice ~0.2 mm isotropic" if record["slice_count"]==64 else "12-slice ~0.1×0.1×1.0 mm anisotropic"


def hard_slice_figure(record,index,destination):
    metric=record["slices"][index];crop=brain_crop(record["target"][:,:,index]);panel=330;header=115;width=5*panel;height=header+panel+60;canvas=Image.new("RGB",(width,height),WHITE);draw=ImageDraw.Draw(canvas)
    brain=[r["slice_index"] for r in record["slices"] if r["brain_voxels"]>0];edge=min(index-brain[0],brain[-1]-index)
    title=f"sub-{record['subject']} | slice {index} | slice Dice {metric['dice']:.4f} | volume Dice {record['dice']:.4f}"
    subtitle=f"{acquisition_group(record)} | spacing {record['spacing']} mm | FP {metric['false_positives']} | FN {metric['false_negatives']} | nearest brain edge {edge} slices"
    draw.text((20,12),title,font=font(25,True),fill=BLACK);draw.text((20,52),subtitle,font=font(17),fill=BLACK)
    for column,(label,image) in enumerate(zip(["MRI","Expert","Prediction","Contours","Error map"],five_panels(record,index,crop,panel))):draw.text((column*panel+10,84),label,font=font(16,True),fill=BLACK);canvas.paste(image,(column*panel,header))
    canvas.paste(legend_strip(width),(0,height-55));canvas.save(destination,optimize=True)


def select_hard_slices(records):
    nonempty=[(record,row) for record in records for row in record["slices"] if row["brain_voxels"]>0]
    selected={}
    def add(items):
        for record,row in items:selected[(record["subject"],row["slice_index"])]=(record,row)
    add(sorted(nonempty,key=lambda item:item[1]["dice"])[:20]);add(sorted(nonempty,key=lambda item:item[1]["false_negatives"],reverse=True)[:10]);add(sorted(nonempty,key=lambda item:item[1]["false_positives"],reverse=True)[:10])
    edges=[]
    for record in records:
        brain=[r for r in record["slices"] if r["brain_voxels"]>0];edges.extend([(record,brain[0]),(record,brain[-1])])
    add(sorted(edges,key=lambda item:item[1]["dice"])[:10])
    for group in (12,64):
        for record in sorted((r for r in records if r["slice_count"]==group),key=lambda r:r["dice"])[:6]:add([(record,min((x for x in record["slices"] if x["brain_voxels"]>0),key=lambda x:x["dice"]))])
    return list(selected.values())


def chart_axes(draw,box,title,ymin,ymax,values,color,empty,worst,first,last):
    x0,y0,x1,y1=box;plot_top=y0+30;plot_bottom=y1-24;draw.rectangle(box,outline=BLACK,width=2);draw.text((x0+8,y0+4),title,font=font(16,True),fill=BLACK)
    count=len(values)
    for index,is_empty in enumerate(empty):
        if is_empty:
            xa=x0+int(index/max(count-1,1)*(x1-x0));xb=x0+int((index+1)/max(count-1,1)*(x1-x0));draw.rectangle((xa,plot_top,xb,plot_bottom),fill=(235,235,235))
    # Native index ticks and numeric y ticks make each panel independently
    # interpretable when extracted from the full figure for a manuscript.
    for tick in sorted({0,count//4,count//2,3*count//4,count-1}):
        px=x0+int(tick/max(count-1,1)*(x1-x0));draw.line((px,plot_top,px,plot_bottom),fill=(220,220,220),width=1);draw.text((px-8,plot_bottom+3),str(tick),font=font(11),fill=BLACK)
    for value in np.linspace(ymin,ymax,3):
        py=plot_bottom-int((value-ymin)/max(ymax-ymin,1e-8)*(plot_bottom-plot_top));draw.line((x0,py,x1,py),fill=(225,225,225),width=1);label=f"{value:.2f}" if ymax<=1.01 else f"{value:.0f}";draw.text((x0+4,py-14),label,font=font(10),fill=GRAY)
    points=[]
    for index,value in enumerate(values):
        px=x0+int(index/max(count-1,1)*(x1-x0));py=plot_bottom-int((value-ymin)/max(ymax-ymin,1e-8)*(plot_bottom-plot_top));points.append((px,py))
    draw.line(points,fill=color,width=3)
    for index,fill in ((first,CYAN),(last,MAGENTA),(worst,RED)):
        px,py=points[index];draw.ellipse((px-6,py-6,px+6,py+6),fill=fill,outline=BLACK)


def per_slice_chart(record,destination,focused=False):
    width=1300;height=1100 if focused else 980;canvas=Image.new("RGB",(width,height),WHITE);draw=ImageDraw.Draw(canvas);draw.text((30,15),f"sub-{record['subject']} | Per-slice metrics | Volumetric Dice {record['dice']:.4f}",font=font(26,True),fill=BLACK)
    brain=[r["slice_index"] for r in record["slices"] if r["brain_voxels"]>0];first,last=brain[0],brain[-1];worst=min(brain,key=lambda i:record["slices"][i]["dice"]);empty=[r["brain_voxels"]==0 for r in record["slices"]]
    panels=[("Slice Dice",[r["dice"] for r in record["slices"]],0,1,(0,95,180)),("Slice IoU",[r["iou"] for r in record["slices"]],0,1,(0,145,100)),("False-positive voxels",[r["false_positives"] for r in record["slices"]],0,max(r["false_positives"] for r in record["slices"])+1,RED),("False-negative voxels",[r["false_negatives"] for r in record["slices"]],0,max(r["false_negatives"] for r in record["slices"])+1,BLUE)]
    top=65;panel_h=190 if not focused else 145
    for i,(title,values,ymin,ymax,color) in enumerate(panels):chart_axes(draw,(75,top+i*(panel_h+12),1250,top+i*(panel_h+12)+panel_h),title,ymin,ymax,values,color,empty,worst,first,last)
    draw.text((80,top+4*(panel_h+12)),f"First brain slice {first}   Middle {brain[len(brain)//2]}   Last {last}   Worst non-empty {worst}",font=font(17,True),fill=BLACK)
    if focused:
        worst_three=sorted(brain,key=lambda i:record["slices"][i]["dice"])[:3];crop=brain_crop(record["target"].any(axis=2));thumb_y=top+4*(panel_h+12)+38
        for j,index in enumerate(worst_three):
            image=five_panels(record,index,crop,220)[0];canvas.paste(image,(160+j*360,thumb_y));draw.text((160+j*360,thumb_y+225),f"slice {index} | Dice {record['slices'][index]['dice']:.4f}",font=font(15,True),fill=BLACK)
    draw.text((75,height-35),"Gray shading = empty expert slice | cyan = first | magenta = last | red = worst non-empty",font=font(15),fill=BLACK);canvas.save(destination,optimize=True)


def dataset_chart(records,kind,destination,data_destination):
    rows=[]
    for record in records:
        brain=[r for r in record["slices"] if r["brain_voxels"]>0];worst=min(brain,key=lambda r:r["dice"]);normalized=(worst["slice_index"]-brain[0]["slice_index"])/max(brain[-1]["slice_index"]-brain[0]["slice_index"],1)
        rows.append({"subject":record["subject"],"dice":record["dice"],"precision":record["precision"],"recall":record["recall"],"slice_count":record["slice_count"],"brain_volume_mm3":record["brain_volume_mm3"],"mean_spacing_mm":float(np.mean(record["spacing"])),"acquisition_group":acquisition_group(record),"worst_slice_normalized":normalized})
    with data_destination.open("w",newline="") as stream:w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Dataset plots use an intentionally narrow Dice range.  A 0--1 axis made
    # all 91 high-performing cases visually indistinguishable at the ceiling.
    dice_values=np.array([r["dice"] for r in rows]);dice_low=max(0.0,float(dice_values.min())-.0015);dice_high=min(1.0,float(dice_values.max())+.0015)
    if kind=="sorted":
        ordered=sorted(rows,key=lambda r:r["dice"]);width,height=1500,2500;canvas=Image.new("RGB",(width,height),WHITE);draw=ImageDraw.Draw(canvas);left=205;right=1430;top=115;bottom=2425;step=(bottom-top)/len(ordered)
        draw.text((30,20),"External holdout Dice by subject",font=font(30,True),fill=BLACK);draw.text((30,62),f"n={len(rows)} | sorted worst to best | truncated x-axis {dice_low:.3f}--{dice_high:.3f}",font=font(17),fill=GRAY)
        for tick in np.linspace(dice_low,dice_high,6):
            x=left+int((tick-dice_low)/(dice_high-dice_low)*(right-left));draw.line((x,top-10,x,bottom),fill=(215,215,215),width=2);draw.text((x-30,bottom+10),f"{tick:.3f}",font=font(14),fill=BLACK)
        for i,row in enumerate(ordered):
            y=int(top+(i+.5)*step);draw.text((28,y-8),f"sub-{row['subject']}",font=font(13),fill=BLACK);x=left+int((row["dice"]-dice_low)/(dice_high-dice_low)*(right-left));color=(0,110,175) if row["slice_count"]==12 else (210,80,45);draw.line((left,y,x,y),fill=color,width=max(8,int(step*.45)))
        draw.text((650,2470),"Volumetric Dice",font=font(18,True),fill=BLACK);canvas.save(destination,optimize=True);return
    width,height=1200,820;canvas=Image.new("RGB",(width,height),WHITE);draw=ImageDraw.Draw(canvas);plot=(135,110,1145,710)
    titles={"histogram":"Distribution of external-holdout Dice","acquisition_group":"Dice by acquisition group","dice_vs_slice_count":"Dice vs native slice count","dice_vs_volume":"Dice vs expert brain volume","recall_vs_dice":"Dice vs recall","precision_vs_dice":"Dice vs precision","worst_slice_location":"Dice vs normalized worst-slice location"}
    draw.text((30,18),titles[kind],font=font(28,True),fill=BLACK);draw.text((30,58),f"n={len(rows)} | volumetric subject-level metrics",font=font(16),fill=GRAY)
    def axes(xmin,xmax,ymin,ymax,xlabel,ylabel,xticks=6,yticks=6):
        draw.rectangle(plot,outline=BLACK,width=2)
        for value in np.linspace(ymin,ymax,yticks):
            py=plot[3]-int((value-ymin)/max(ymax-ymin,1e-9)*(plot[3]-plot[1]));draw.line((plot[0],py,plot[2],py),fill=(220,220,220),width=1);draw.text((55,py-9),f"{value:.3f}",font=font(13),fill=BLACK)
        for value in np.linspace(xmin,xmax,xticks):
            px=plot[0]+int((value-xmin)/max(xmax-xmin,1e-9)*(plot[2]-plot[0]));draw.line((px,plot[1],px,plot[3]),fill=(232,232,232),width=1);label=f"{value:.3g}";draw.text((px-22,plot[3]+10),label,font=font(13),fill=BLACK)
        draw.text((plot[0]+(plot[2]-plot[0])//2-80,770),xlabel,font=font(17,True),fill=BLACK);draw.text((10,390),ylabel,font=font(17,True),fill=BLACK)
    mean=float(dice_values.mean());median=float(np.median(dice_values))
    if kind=="histogram":
        bins=np.linspace(dice_low,dice_high,18);counts,_=np.histogram(dice_values,bins);maximum=max(counts)+1;axes(dice_low,dice_high,0,maximum,"Volumetric Dice","Subjects",6,6);bar=(plot[2]-plot[0])/len(counts)
        for i,count in enumerate(counts):draw.rectangle((plot[0]+i*bar+2,plot[3]-count/maximum*(plot[3]-plot[1]),plot[0]+(i+1)*bar-2,plot[3]),fill=(0,110,175))
        for value,color,label in ((mean,RED,"mean"),(median,MAGENTA,"median")):
            x=plot[0]+int((value-dice_low)/(dice_high-dice_low)*(plot[2]-plot[0]));draw.line((x,plot[1],x,plot[3]),fill=color,width=3);draw.text((x+5,plot[1]+8),f"{label} {value:.4f}",font=font(13,True),fill=color)
    elif kind=="acquisition_group":
        groups=[("12-slice",[r["dice"] for r in rows if r["slice_count"]==12],(0,120,190)),("64-slice",[r["dice"] for r in rows if r["slice_count"]==64],(210,80,45))]
        axes(0.5,2.5,dice_low,dice_high,"Acquisition group","Volumetric Dice",3,6)
        for g,(label,values,color) in enumerate(groups):
            center=plot[0]+int(((g+1)-.5)/2*(plot[2]-plot[0]));
            for j,value in enumerate(values):x=center+((j*17)%121)-60;y=plot[3]-int((value-dice_low)/(dice_high-dice_low)*(plot[3]-plot[1]));draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
            draw.text((center-115,725),f"{label}: n={len(values)}, mean={np.mean(values):.4f}",font=font(15,True),fill=color)
    else:
        key={"dice_vs_slice_count":"slice_count","dice_vs_volume":"brain_volume_mm3","recall_vs_dice":"recall","precision_vs_dice":"precision","worst_slice_location":"worst_slice_normalized"}[kind];xs=np.array([r[key] for r in rows]);ys=np.array([r["dice"] for r in rows]);span=max(float(np.ptp(xs)),1e-8)
        labels={"slice_count":"Native slices","brain_volume_mm3":"Expert brain volume (mm3)","recall":"Recall","precision":"Precision","worst_slice_normalized":"Worst-slice position (0=first, 1=last)"};axes(float(xs.min()),float(xs.max()),dice_low,dice_high,labels[key],"Volumetric Dice",6,6)
        mean_y=plot[3]-int((mean-dice_low)/(dice_high-dice_low)*(plot[3]-plot[1]));median_y=plot[3]-int((median-dice_low)/(dice_high-dice_low)*(plot[3]-plot[1]));draw.line((plot[0],mean_y,plot[2],mean_y),fill=RED,width=2);draw.line((plot[0],median_y,plot[2],median_y),fill=MAGENTA,width=2)
        for row,x,y in zip(rows,xs,ys):
            px=plot[0]+(x-xs.min())/span*(plot[2]-plot[0]);py=plot[3]-(y-dice_low)/(dice_high-dice_low)*(plot[3]-plot[1]);color=(0,110,175) if row["slice_count"]==12 else (210,80,45);draw.ellipse((px-6,py-6,px+6,py+6),fill=color)
        for row,prefix in ((min(rows,key=lambda r:r["dice"]),"worst"),(max(rows,key=lambda r:r["dice"]),"best")):
            x=row[key];px=plot[0]+(x-xs.min())/span*(plot[2]-plot[0]);py=plot[3]-(row["dice"]-dice_low)/(dice_high-dice_low)*(plot[3]-plot[1]);draw.text((min(px+8,plot[2]-155),max(plot[1]+5,py-23)),f"{prefix}: sub-{row['subject']}",font=font(13,True),fill=BLACK)
    draw.text((145,740),f"blue: 12-slice   orange: 64-slice   red: mean {mean:.4f}   magenta: median {median:.4f}",font=font(15,True),fill=BLACK);canvas.save(destination,optimize=True)


def validate_pngs(paths):
    failures=[]
    for path in paths:
        image=Image.open(path).convert("RGB");array=np.asarray(image)
        if image.width<700 or image.height<400:failures.append(f"{path}: unreadable dimensions {image.size}")
        if float(array.std())<5:failures.append(f"{path}: near-blank image")
    if failures:raise RuntimeError("Generated PNG validation failed:\n"+"\n".join(failures[:20]))
    return {"figures_validated":len(paths),"minimum_width":min(Image.open(p).width for p in paths),"minimum_height":min(Image.open(p).height for p in paths)}


def main():
    for directory in ("subject_overviews","hard_slices","per_slice_charts","dataset_charts","plotted_data"):(OUT/directory).mkdir(parents=True,exist_ok=True)
    records,audit=audit_and_load();ranked=sorted(records,key=lambda r:r["dice"]);best=list(reversed(ranked[-5:]));median_start=len(ranked)//2-2;median=ranked[median_start:median_start+5];worst=ranked[:10]
    generated=[]
    for record in best+median+worst:
        path=OUT/"subject_overviews"/f"sub-{record['subject']}_overview.png";subject_overview(record,path);generated.append(path)
    hard=select_hard_slices(records)
    for record,row in hard:
        path=OUT/"hard_slices"/f"sub-{record['subject']}_slice-{row['slice_index']:03d}.png";hard_slice_figure(record,row["slice_index"],path);generated.append(path)
    informative=sorted(hard,key=lambda item:(item[1]["dice"],-(item[1]["false_positives"]+item[1]["false_negatives"])))[:12]
    thumbs=[]
    for record,row in informative:
        figure=Image.open(OUT/"hard_slices"/f"sub-{record['subject']}_slice-{row['slice_index']:03d}.png");figure.thumbnail((1200,370),Image.Resampling.LANCZOS);thumbs.append(figure.copy())
    sheet=Image.new("RGB",(2400,6*370),WHITE)
    for index,image in enumerate(thumbs):sheet.paste(image,((index%2)*1200,(index//2)*370))
    contact=OUT/"hard_slices"/"hard_slice_contact_sheet_top12.png";sheet.save(contact,optimize=True);generated.append(contact)
    worst_ids={r["subject"] for r in worst}
    for record in records:
        path=OUT/"per_slice_charts"/f"sub-{record['subject']}_per_slice_metrics.png";per_slice_chart(record,path);generated.append(path)
        if record["subject"] in worst_ids:
            focused=OUT/"per_slice_charts"/f"sub-{record['subject']}_focused_worst_slices.png";per_slice_chart(record,focused,True);generated.append(focused)
    chart_kinds=["sorted","histogram","acquisition_group","dice_vs_slice_count","dice_vs_volume","recall_vs_dice","precision_vs_dice","worst_slice_location"]
    for kind in chart_kinds:
        path=OUT/"dataset_charts"/f"{kind}.png";dataset_chart(records,kind,path,OUT/"plotted_data"/f"{kind}.csv");generated.append(path)
    validation=validate_pngs(generated);audit["generated_validation"]=validation;audit["previous_figure_diagnosis"]=["full-field display made the small brain occupy too few pixels","linear 0-1 y scaling compressed Dice traces against the top border","plots lacked titles, axes, units, legends, and readable tick labels","edge/empty slices were selected without informative replacement","simple raster figures contained excessive whitespace"]
    (OUT/"visualization_audit.json").write_text(json.dumps(audit,indent=2))
    summary={"figures_created":len(generated),"subject_overviews":len(best+median+worst),"hard_slice_figures":len(hard),"per_slice_charts":len(records),"focused_worst_charts":len(worst),"dataset_charts":len(chart_kinds),"contact_sheet":str(contact),"best_subjects":[r["subject"] for r in best],"median_subjects":[r["subject"] for r in median],"worst_subjects":[r["subject"] for r in worst],"hard_slice_count":len(hard),**validation}
    (OUT/"visualization_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
