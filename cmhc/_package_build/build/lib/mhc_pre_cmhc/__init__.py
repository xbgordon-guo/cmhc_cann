"""mhc_pre_cmhc — AscendC kernel operator for MHC (Multi-Head Composition)."""

import os
from pathlib import Path
import torch

# 加载 torch custom op (.so)
_pkg_dir = Path(__file__).resolve().parent
_so_files = list(_pkg_dir.glob("mhc_pre_cmhc_torch_ops*.so"))
if _so_files:
    torch.ops.load_library(str(_so_files[0]))

from .model_new_ascendc import ModelNew
from .model import Model

__all__ = ["ModelNew", "Model"]
