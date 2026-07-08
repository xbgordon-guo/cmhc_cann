#!/usr/bin/env python3
"""mhc_pre_cmhc — PyTorch nn.Module wrapper for the cmhc AscendC kernel.

Calls torch.ops.npu.mhc_pre_cmhc() via the custom binding.

Architecture: Cube+AIC+AIV with Softmax-over-24-permutations H_res.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# --- Load the custom .so ---
_BINDING_DIR = Path(__file__).resolve().parent / "binding"
_LIB_PATTERN = str(_BINDING_DIR / "mhc_pre_cmhc_torch_ops.*.so")

try:
    import mhc_pre_cmhc_torch_ops  # noqa: F401 — pip-installed
except ImportError:
    import glob as _glob
    _libs = _glob.glob(_LIB_PATTERN)
    if _libs:
        torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    """MHC with mhc_pre_cmhc kernel (Cube MatMul + Softmax-permutation H_res)."""

    def __init__(self, N: int = 4, C: int = 3584):
        super().__init__()
        self.N = N
        self.C = C
        nC = N * C                     # 14336
        n_perm = math.factorial(N)     # 24
        out_dim = n_perm + 2 * N       # 32

        # phi: (hcMix, N*C) = (32, 14336)
        self.phi   = nn.Parameter(torch.empty(out_dim, nC))
        # alpha: (3,) — pre, post, res scaling
        self.alpha = nn.Parameter(torch.ones(3) * 0.01)
        # bias: (hcMix,) = (32,)
        self.bias  = nn.Parameter(torch.zeros(out_dim) * 0.01)

        self._init_weights()

    def _init_weights(self):
        state = torch.get_rng_state()
        torch.manual_seed(12345)
        nn.init.xavier_uniform_(self.phi)
        torch.set_rng_state(state)

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: (B, S, N, C) input — can be float32, float16, or bfloat16.

        Returns:
            h_in:           (B, S, C)       aggregated output (fp32)
            h_post:         (B, S, N)       post weights (fp32)
            h_res:          (B, S, N, N)    permutation matrix (fp32)
            h_pre:          (B, S, N)       pre weights (fp32)
            hc_before_norm: (B, S, outDim)  raw projection (fp32)
            inv_rms:        (B, S)          RMS norm factors (fp32)
        """
        device = x.device
        phi   = self.phi.data.to(device, copy=False)
        alpha = self.alpha.data.to(device, copy=False)
        bias  = self.bias.data.to(device, copy=False)

        result = torch.ops.npu.mhc_pre_cmhc(
            x, phi, alpha, bias,
            self.N,           # hc_mult
            0,                 # num_iters (unused, was Sinkhorn)
            1e-6,              # hc_eps
            1e-6,              # norm_eps
            False,             # need_backward
        )
        # result = [h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms_out]
        # Return only the 3 outputs that match model.py (h_in, h_post, h_res)
        return result[0], result[1], result[2]
