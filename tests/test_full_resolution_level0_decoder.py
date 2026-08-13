"""Focused invariants for the controlled level0 decoder ablation."""
from pathlib import Path
import torch
from models.query_mask_decoder import MultiScaleOneQueryMaskDecoder,FullResolutionLevel0OneQueryMaskDecoder
ROOT=Path(__file__).resolve().parents[1]

def features(batch=1):
 return {'level0':torch.randn(batch,48,16,16,20),'level1':torch.randn(batch,48,8,8,10),'level2':torch.randn(batch,96,4,4,5),'level3':torch.randn(batch,192,2,2,3),'level4':torch.randn(batch,384,1,1,2)}

def test_one_query_parameters_and_native_level0_shape():
 model=FullResolutionLevel0OneQueryMaskDecoder(32,4)
 assert tuple(model.query.shape)==(1,1,32)
 assert sum(p.numel() for p in model.parameters())==180466
 assert model(features()).shape==(1,1,16,16,20)

def test_zero_residual_initialization_exactly_preserves_baseline():
 old=MultiScaleOneQueryMaskDecoder(32,4);new=FullResolutionLevel0OneQueryMaskDecoder(32,4)
 shared={k:v for k,v in old.state_dict().items() if k in new.state_dict()}
 new.load_state_dict(shared,strict=False);f=features()
 with torch.no_grad():a=old({k:v for k,v in f.items() if k!='level0'});b=new(f)
 expected=torch.nn.functional.interpolate(a,size=f['level0'].shape[-3:],mode='trilinear',align_corners=False)
 assert torch.equal(b,expected)

def test_checkpoint_and_corrected_label_configuration():
 cfg=__import__('json').loads((ROOT/'configs/full_resolution_level0_decoder.yaml').read_text())
 assert cfg['label_resampling']=={'is_seg':True,'order':0,'order_z':0}
 ck=torch.load(ROOT/'outputs/full_resolution_level0_decoder/checkpoints/best_level0_decoder.pt',map_location='cpu',weights_only=False)
 model=FullResolutionLevel0OneQueryMaskDecoder(32,4);model.load_state_dict(ck['decoder_state_dict'],strict=True)
 assert ck['epoch']==20
