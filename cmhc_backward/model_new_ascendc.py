#!/usr/bin/env python3
"""mhc_pre_cmhc_backward — PyTorch nn.Module wrapper for the mhc_pre_cmhc_backward AscendC kernel.

Calls torch.ops.npu.mhc_pre_cmhc_backward() via the custom binding.
For CPU testing (verification_ascendc.py), falls back to load_inline compilation.

Architecture: Cube+AIC+AIV implementing the full cmhc backward pass.
Permutation matrices are pre-computed on the Python side and passed as input,
matching the reference cmhc_cann/cmhc pattern.

Device dispatch:
  NPU: keeps tensors on NPU, calls torch.ops.npu.mhc_pre_cmhc_backward (real kernel profiling).
  CPU: uses load_inline for deterministic precision verification.
"""

from pathlib import Path

import torch
import torch.nn as nn

# --- Load the custom .so ---
_BINDING_DIR = Path(__file__).resolve().parent / "binding"
_LIB_PATTERN = str(_BINDING_DIR / "mhc_pre_cmhc_backward_torch_ops.*.so")

_BACKEND_NPU_FN = None  # torch.ops.npu.mhc_pre_cmhc_backward — PrivateUse1 (NPU kernel)
_BACKEND_CPU_FN = None  # load_inline compiled — CPU reference


def _get_backend_fn(device: torch.device):
    """Load the C++ binding, dispatching to NPU or CPU backend by device.

    NPU path: torch.ops.npu.mhc_pre_cmhc_backward — runs at:: ops on NPU via PrivateUse1.
    CPU path: load_inline — CPU-only for deterministic verification precision.
    """
    if device.type == 'cpu':
        global _BACKEND_CPU_FN
        if _BACKEND_CPU_FN is not None:
            return _BACKEND_CPU_FN
        cpp_path = _BINDING_DIR / "mhc_pre_cmhc_backward_torch_host.cpp"
        with open(cpp_path) as f:
            cpp_source = f.read()
        from torch.utils.cpp_extension import load_inline
        module = load_inline(
            name="mhc_pre_cmhc_backward_cpu",
            cpp_sources=[cpp_source],
            functions=["mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward"],
            extra_cflags=["-std=c++17", "-O2"],
            extra_include_paths=[str(_BINDING_DIR)],
            verbose=False,
        )
        _BACKEND_CPU_FN = getattr(module, "mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward")
        return _BACKEND_CPU_FN

    # NPU (PrivateUse1) path — keep tensors on NPU for real kernel profiling
    global _BACKEND_NPU_FN
    if _BACKEND_NPU_FN is not None:
        return _BACKEND_NPU_FN
    try:
        import mhc_pre_cmhc_backward_torch_ops  # noqa: F401 — pip-installed
    except ImportError:
        import glob as _glob
        _libs = _glob.glob(_LIB_PATTERN)
        if _libs:
            torch.ops.load_library(_libs[0])
    _BACKEND_NPU_FN = torch.ops.npu.mhc_pre_cmhc_backward
    return _BACKEND_NPU_FN


class ModelNew(nn.Module):
    """cmhc backward via C++ binding (matches AscendC kernel).

    Forward intermediates (h_pre, hc_before_norm, inv_rms) are recomputed
    from x, phi, alpha, bias on CPU before the C++ binding call. The values
    from get_input_groups() are random placeholders that the reference model
    also ignores.
    """

    def __init__(self, N: int = 4, C: int = 64):
        super().__init__()
        self.N = N
        self.C = C

    def forward(self, grad_h_in, grad_h_post, grad_h_res,
                x, phi, alpha, bias,
                h_pre_unused, hc_before_norm_unused, inv_rms_unused,
                perm_mats):
        """
        Args:
            grad_h_in:        (B, S, C)         gradient w.r.t. h_in
            grad_h_post:      (B, S, N)         gradient w.r.t. h_post
            grad_h_res:       (B, S, N, N)      gradient w.r.t. h_res
            x:                (B, S, N, C)      forward input
            phi:              (outDim, N*C)     forward weight
            alpha:            (3,)             forward scale
            bias:             (outDim,)        forward bias
            h_pre_unused:     (B, S, N)         (ignored — recomputed below)
            hc_before_norm_unused: (B, S, outDim) (ignored — recomputed below)
            inv_rms_unused:   (B, S)            (ignored — recomputed below)
            perm_mats:        (n!, N, N)        permutation matrices

        Returns:
            grad_x:     (B, S, N, C)  gradient w.r.t. x
            grad_phi:   (outDim, N*C) gradient w.r.t. phi
            grad_alpha: (3,)         gradient w.r.t. alpha
            grad_bias:  (outDim,)    gradient w.r.t. bias
        """
        orig_device = x.device
        B, S, N, C = x.shape
        nC = N * C
        M = B * S
        outDim = phi.shape[0]
        nPerm = outDim - 2 * N
        eps = 1e-6

        # Convert to fp32 in-place (on whatever device the tensors are on).
        # CPU path: remains on CPU — load_inline backend.
        # NPU path: stays on NPU — torch.ops.npu backend dispatches to NPU kernels.
        x_f32     = x.float()
        phi_f32   = phi.float()
        alpha_f32 = alpha.float()
        bias_f32  = bias.float()
        ghi_f32   = grad_h_in.float()
        ghp_f32   = grad_h_post.float()
        ghr_f32   = grad_h_res.float()
        perms_f32 = perm_mats.float()

        # Recompute forward intermediates
        x_flat = x_f32.reshape(M, nC)
        mean_sq = (x_flat * x_flat).mean(dim=-1, keepdim=True)
        r = mean_sq.sqrt()
        r_inv = 1.0 / (r + eps)
        H = x_flat @ phi_f32.t()

        alpha_exp = torch.cat([
            alpha_f32[0].expand(N),
            alpha_f32[1].expand(N),
            alpha_f32[2].expand(nPerm),
        ]).view(1, -1)

        act = r_inv * H * alpha_exp + bias_f32.view(1, -1)
        h_pre_computed = act[:, :N].sigmoid().reshape(B, S, N)
        hc_before_norm_computed = H.reshape(B, S, outDim)
        inv_rms_computed = r_inv.reshape(B, S)

        fn = _get_backend_fn(orig_device)
        result = fn(ghi_f32, ghp_f32, ghr_f32,
                    x_f32, phi_f32, alpha_f32, bias_f32,
                    h_pre_computed, hc_before_norm_computed, inv_rms_computed,
                    perms_f32, eps)
        return tuple(t.to(orig_device) for t in result)


def get_init_inputs():
    """Constructor args for ModelNew()."""
    return []
