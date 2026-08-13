"""Focused invariants for the controlled corrected-label experiment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import torch
from models.query_mask_decoder import MultiScaleOneQueryMaskDecoder

def test_architecture_remains_locked():
    decoder=MultiScaleOneQueryMaskDecoder(32,4)
    assert tuple(decoder.query.shape)==(1,1,32)
    assert sum(p.numel() for p in decoder.parameters())==170401
    assert tuple(decoder.CHANNELS)==("level1","level2","level3","level4")

def test_checkpoint_matches_locked_architecture():
    path=ROOT/"outputs/corrected_label_retraining/checkpoints/best_corrected_labels.pt"
    payload=torch.load(path,map_location="cpu",weights_only=False)
    decoder=MultiScaleOneQueryMaskDecoder(32,4)
    decoder.load_state_dict(payload["decoder_state_dict"],strict=True)
    assert payload["epoch"]==17

def test_corrected_cache_targets_are_binary_and_old_cache_is_not_reused():
    config=__import__("json").loads((ROOT/"configs/corrected_label_retraining.yaml").read_text())
    assert config["corrected_cache"]=="/tmp/rs2_corrected_label_features"
    assert config["label_resampling"]=={"is_seg":True,"order":0,"order_z":0}
    files=list(Path(config["corrected_cache"]).glob("*.pt"))
    assert len(files)==55
    for path in files:
        payload=torch.load(path,map_location="cpu",weights_only=False)
        assert set(payload["target"].unique().tolist()) <= {0,1}
        assert payload["label_interpolation"]=="nearest/order0/is_seg=True"
