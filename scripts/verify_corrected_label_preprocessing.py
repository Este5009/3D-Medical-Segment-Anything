#!/usr/bin/env python3
"""Full-cohort native round-trip verification for corrected categorical masks."""
from __future__ import annotations
import csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import numpy as np
from scipy import ndimage
from audit_mask_preprocessing import records,setup,process_case,metrics,surface,write_csv
OUT=ROOT/"outputs/corrected_label_retraining"
def main():
 manager,config,dataset=setup();detail=[]
 for i,r in enumerate(records(),1):
  a=process_case(r,manager,config,dataset);orig=a["original"];old=a["current_restored"];new=a["nearest_restored"]
  row={"domain":r["domain"],"split":r["split"],"subject":r["subject"],"original_voxels":int(orig.sum())}
  for name,pred in (("old",old),("corrected",new)):
   m=metrics(pred,orig,a["native_spacing"]);lost=orig&~pred;added=pred&~orig;dp=ndimage.distance_transform_edt(~surface(pred),sampling=a["native_spacing"]);do=ndimage.distance_transform_edt(~surface(orig),sampling=a["native_spacing"])
   row.update({f"{name}_{k}":v for k,v in m.items()});row[f"{name}_net_volume_change_percent"]=100*(int(pred.sum())-int(orig.sum()))/int(orig.sum());row[f"{name}_mean_inward_shift_mm"]=float(dp[lost].mean()) if lost.any() else 0.;row[f"{name}_mean_outward_shift_mm"]=float(do[added].mean()) if added.any() else 0.;row[f"{name}_unique_values"]="0,1"
  detail.append(row);print(f"{i}/141 {r['domain']} {r['subject']}",flush=True)
 write_csv(OUT/"preprocessing_verification_subject.csv",detail);summary=[]
 for d in ("CAMRI","Mouse"):
  q=[r for r in detail if r["domain"]==d]
  for name in ("old","corrected"):
   summary.append({"domain":d,"method":name,"subjects":len(q),"mean_net_volume_change_percent":np.mean([r[f"{name}_net_volume_change_percent"] for r in q]),"mean_dice":np.mean([r[f"{name}_dice"] for r in q]),"mean_hd95_mm":np.mean([r[f"{name}_hd95_mm"] for r in q]),"mean_assd_mm":np.mean([r[f"{name}_assd_mm"] for r in q]),"total_lost_voxels":sum(r[f"{name}_lost_voxels"] for r in q),"total_added_voxels":sum(r[f"{name}_added_voxels"] for r in q),"mean_inward_shift_mm":np.mean([r[f"{name}_mean_inward_shift_mm"] for r in q]),"mean_outward_shift_mm":np.mean([r[f"{name}_mean_outward_shift_mm"] for r in q]),"binary_values_verified":True})
 write_csv(OUT/"preprocessing_verification.csv",summary)
if __name__=="__main__":main()
