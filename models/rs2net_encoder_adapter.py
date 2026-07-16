"""Safe, decoder-free access to the verified sibling RS2-Net encoder.

The baseline source and checkpoint remain in ``../RS2-Net-Reproduction``.
This module only adds that source tree to Python's import search path; it never
copies, edits, or writes into the reproduction repository.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RS2NetPaths:
    """Resolved locations required to use the sibling baseline."""

    baseline_root: Path
    baseline_project: Path
    checkpoint: Path
    dataset_root: Path

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "RS2NetPaths":
        """Resolve config paths relative to this repository, independent of cwd."""
        baseline_root = (REPOSITORY_ROOT / str(config["baseline_root"])).resolve()
        dataset_root = (REPOSITORY_ROOT / str(config["dataset_root"])).resolve()
        baseline_project = baseline_root / str(config.get("baseline_project", "Rodent-Skull-Stripping"))
        checkpoint = baseline_project / str(config["checkpoint"])
        return cls(baseline_root, baseline_project, checkpoint, dataset_root)

    def validate(self) -> None:
        missing = [
            path
            for path in (self.baseline_root, self.baseline_project, self.checkpoint, self.dataset_root)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Required RS2-Net resource(s) missing: " + ", ".join(map(str, missing)))


def _import_baseline_model(baseline_project: Path):
    """Import RSSNet without requiring the baseline to be installed globally."""
    project = str(baseline_project)
    if project not in sys.path:
        sys.path.insert(0, project)
    try:
        from RS2.network.RSSNet import RSSNet
    except ImportError as error:
        raise ImportError(
            "RS2-Net could not be imported. Install the baseline dependencies "
            "from ../RS2-Net-Reproduction/Rodent-Skull-Stripping/requirements.txt."
        ) from error
    return RSSNet


class RS2NetEncoderAdapter(nn.Module):
    """Wrap RSSNet and return its decoder-ready 3D feature pyramid only.

    The wrapped network is intentionally kept intact so its released state dict
    loads strictly. ``forward`` stops before ``decoder4``, the first original
    U-Net-style upsampling block. Tensor layout is always ``[B, C, D, H, W]``.
    """

    FEATURE_NAMES = ("level0", "level1", "level2", "level3", "level4")

    def __init__(
        self,
        paths: RS2NetPaths,
        image_size: Sequence[int] = (128, 128, 160),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 48,
    ) -> None:
        import torch

        super().__init__()
        paths.validate()
        RSSNet = _import_baseline_model(paths.baseline_project)
        self.paths = paths
        self.network = RSSNet(
            img_size=tuple(image_size),
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
        )
        checkpoint = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # The original release was saved after torch.compile and prefixes every
        # key with _orig_mod.; the supplied clean checkpoint removes that prefix.
        if any(key.startswith("_orig_mod.") for key in state_dict):
            state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
        self.network.load_state_dict(state_dict, strict=True)
        self.network.eval()

    def forward(self, volume) -> Dict[str, object]:
        """Return features consumed by the original decoder, without running it."""
        hidden = self.network.swinViT(volume, self.network.normalize)

        # level0 preserves the input grid; levels1-4 halve D/H/W successively.
        # RSSNet applies convolutional refinement to the levels used by its skip
        # connections. level3 is passed directly to decoder4 as its first skip.
        return {
            "level0": self.network.encoder1(volume),
            "level1": self.network.encoder2(hidden[0]),
            "level2": self.network.encoder3(hidden[1]),
            "level3": hidden[2],
            "level4": self.network.feature(hidden[3]),
        }
