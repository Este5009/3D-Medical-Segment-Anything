#!/usr/bin/env python3
"""Pre-training tensor trace for real corrected-label CAMRI and Mouse cases."""
from __future__ import annotations
import json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import torch
import torch.nn.functional as F
from models.query_mask_decoder import FullResolutionLevel0OneQueryMaskDecoder,MultiScaleOneQueryMaskDecoder
OUT=ROOT/"outputs/full_resolution_level0_decoder"
def peak_mib():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=="darwin" else 1024)
def trace(decoder,features):
 retained=[];projected={n:decoder.projections[n](features[n]) for n in ("level1","level2","level3","level4")}
 for n in ("level4","level3","level2","level1"):retained.append({"stage":f"projected_{n}","shape":list(projected[n].shape)})
 fused={"level4":projected["level4"]};previous=fused["level4"]
 for n in ("level3","level2","level1"):
  previous=F.interpolate(previous,size=projected[n].shape[-3:],mode="trilinear",align_corners=False);previous=decoder.refinements[n](projected[n]+previous);fused[n]=previous;retained.append({"stage":f"fused_{n}","shape":list(previous.shape)})
 level0=decoder.level0_projection(features["level0"]);up=F.interpolate(decoder.level1_to_level0(fused["level1"]),size=level0.shape[-3:],mode="trilinear",align_corners=False);fused0=decoder.level0_refinement(level0+up);retained += [{"stage":"projected_level0","shape":list(level0.shape)},{"stage":"level1_to_level0_interpolation","shape":list(up.shape),"purpose":"feature fusion; not final-logit rendering"},{"stage":"fused_level0","shape":list(fused0.shape)}]
 query=decoder.query.expand(features["level0"].shape[0],-1,-1)
 for n in ("level4","level3","level2","level1"):query=decoder.query_updates[n](query,fused[n]);retained.append({"stage":f"query_after_{n}","shape":list(query.shape)})
 qfeature=decoder.level0_query_projection(F.avg_pool3d(fused0,2,2));query=decoder.query_updates["level0"](query,qfeature);retained += [{"stage":"level0_query_attention_tokens","shape":list(qfeature.shape),"note":"pooled only for global query update"},{"stage":"query_after_level0","shape":list(query.shape)}]
 emb32=decoder.mask_embedding(query).squeeze(1);basevox=decoder.mask_refinement(fused["level1"]);base=torch.einsum("bc,bcdhw->bdhw",emb32,basevox).unsqueeze(1)+decoder.mask_bias.view(1,1,1,1,1);base=F.interpolate(base,size=level0.shape[-3:],mode="trilinear",align_corners=False);vox=F.interpolate(decoder.level1_to_level0(basevox),size=level0.shape[-3:],mode="trilinear",align_corners=False);vox=decoder.level0_refinement(vox+level0);emb=decoder.level0_mask_embedding(emb32);residual=torch.einsum("bc,bcdhw->bdhw",emb,vox).unsqueeze(1);logits=base+decoder.level0_residual_scale*residual;retained += [{"stage":"transferred_baseline_logit_skip","shape":list(base.shape),"note":"initialization only; not sole final output"},{"stage":"full_grid_mask_voxel_features","shape":list(vox.shape)},{"stage":"full_grid_level0_logit_residual","shape":list(residual.shape)},{"stage":"native_learned_logits","shape":list(logits.shape),"final_interpolation":False}];return logits,retained
def main():
 OUT.mkdir(parents=True,exist_ok=True);torch.manual_seed(20260717);decoder=FullResolutionLevel0OneQueryMaskDecoder(32,4).eval();cases={"CAMRI":"/tmp/rs2_level0_corrected_features/camri_validation_035.pt","Mouse":"/tmp/rs2_level0_corrected_features/mouse_validation_POLYIC_20190510_mouse31__E2_P1.pt"};result={"old":{"parameters":sum(p.numel() for p in MultiScaleOneQueryMaskDecoder(32,4).parameters()),"finest_feature":"level1","native_logits_shape":[1,1,64,64,80],"final_2x_interpolation":True},"new":{"parameters":sum(p.numel() for p in decoder.parameters()),"level0_fusion":"48→8 lateral projection + 32→8 top-down projection + additive depthwise-3x3/pointwise-1x1 refinement","query_conditioning_at_level0":True,"query_level0_tokens":"2x average-pooled solely for global query attention","mask_logits":"8-channel full-grid voxel/query dot product","final_logit_interpolation":False},"cases":{}}
 for domain,path in cases.items():
  payload=torch.load(path,map_location="cpu",weights_only=False);features={k:v.float() for k,v in payload["features"].items()};start=time.time();before=peak_mib()
  with torch.inference_mode():logits,stages=trace(decoder,features)
  if logits.shape[-3:]!=(128,128,160):raise RuntimeError("Not full resolution")
  result["cases"][domain]={"source":path,"encoder_features":{k:list(v.shape) for k,v in features.items()},"stages":stages,"elapsed_seconds":time.time()-start,"peak_process_memory_mib":peak_mib(),"incremental_peak_memory_mib_lower_bound":max(0,peak_mib()-before)}
 (OUT/"architecture_trace.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
