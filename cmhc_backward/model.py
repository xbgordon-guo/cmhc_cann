#!/usr/bin/env python3
"""mhc_pre_cmhc_backward — PyTorch reference backward for cmhc (Softmax-Permutation H_res).

Computes gradients of the cmhc operator via torch.autograd.grad against the
forward computation, matching the cmhc kernel's backward analytically.

Forward (from cmhc_cann/cmhc model.py):
  1. x_flat = x.reshape(B*S, N*C).float()
  2. r = ||x_flat||_2 / sqrt(N*C),  r_inv = 1 / (r + eps)    (RMS norm)
  3. H = x_flat @ phi^T                                          (MatMul)
  4. act = r_inv * H * alpha_exp + bias
     - h_pre_raw  = act[:, :N]                                   → sigmoid
     - h_post_raw = act[:, N:2N]                                 → 2*sigmoid
     - h_res_logits = act[:, 2N:2N+nPerm]                        → softmax
  5. h_pre  = sigmoid(h_pre_raw)
     h_post = 2 * sigmoid(h_post_raw)
     res_coeff = softmax(h_res_logits)
  6. h_res = einsum('mr,rij->mij', res_coeff, perms)             (Birkhoff-von Neumann)
  7. h_in = sum_n(x_streams[n] * h_pre[n])

Backward:
  Given: grad_h_in, grad_h_post, grad_h_res, x, phi, alpha, bias,
         h_pre, hc_before_norm, inv_rms, perm_mats
  Compute: grad_x, grad_phi, grad_alpha, grad_bias

  The reference uses torch.autograd.grad() on a loss constructed from
  the recomputed forward outputs, yielding exact analytical gradients.

Permutation matrices are pre-computed on the Python side and passed as input,
matching the reference cmhc_cann/cmhc pattern.
"""

import itertools
import json
import math
import os

import torch
import torch.nn as nn


# =========================================================================
# Permutation matrices — pre-computed on Python side, passed to kernel
# =========================================================================

def get_permutation_matrices(n: int, device, dtype):
    """Generate all n! permutation matrices of shape (n!, n, n).

    Args:
        n:      Matrix dimension.
        device: Target torch device.
        dtype:  Target torch dtype.

    Returns:
        Tensor of shape (n!, n, n) where each slice is a permutation matrix.
    """
    perms = []
    for p in itertools.permutations(range(n)):
        mat = torch.zeros(n, n, device=device, dtype=dtype)
        for i, j in enumerate(p):
            mat[i, j] = 1.0
        perms.append(mat)
    return torch.stack(perms, dim=0)


# =========================================================================
# Reference model — autograd backward
# =========================================================================

class Model(nn.Module):
    """PyTorch reference backward for cmhc — uses autograd for exact gradients.

    Permutation matrices are passed as input (pre-computed on Python side),
    matching the reference cmhc_cann/cmhc pattern.

    The h_pre, hc_before_norm, inv_rms inputs from get_input_groups() are
    random placeholders — this model ignores them and recomputes the full
    forward pass inside torch.enable_grad() for correct autograd tracing.
    Computation stays on the input device (NPU) so profiling tools can
    measure real kernel execution time.
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
            h_pre_unused:     (B, S, N)         (ignored — recomputed internally)
            hc_before_norm_unused: (B, S, outDim) (ignored — recomputed internally)
            inv_rms_unused:   (B, S)            (ignored — recomputed internally)
            perm_mats:        (n!, N, N)        permutation matrices

        Returns:
            grad_x:     (B, S, N, C)  gradient w.r.t. x
            grad_phi:   (outDim, N*C) gradient w.r.t. phi
            grad_alpha: (3,)         gradient w.r.t. alpha
            grad_bias:  (outDim,)    gradient w.r.t. bias
        """
        orig_device = x.device
        B, S, N, C = x.shape

        # Keep tensors on the original device (NPU) so that profiling tools
        # can measure real NPU execution time for both Ref and Asc columns.
        # verification_ascendc.py calls models inside torch.no_grad(), so we
        # must re-enable grad for autograd.grad to work.
        with torch.enable_grad():
            x_t     = x.float().requires_grad_(True)
            phi_t   = phi.float().requires_grad_(True)
            alpha_t = alpha.float().requires_grad_(True)
            bias_t  = bias.float().requires_grad_(True)
            ghin_t  = grad_h_in.float()
            ghp_t   = grad_h_post.float()
            ghr_t   = grad_h_res.float()
            perms_t = perm_mats.float()

            eps = 1e-6
            B_, S_, N_, C_ = x_t.shape
            M = B_ * S_
            nC = N_ * C_
            outDim = phi_t.shape[0]
            nPerm = outDim - 2 * N_

            # Forward pass
            x_flat = x_t.reshape(M, nC)
            mean_sq = (x_flat * x_flat).mean(dim=-1, keepdim=True)
            r = mean_sq.sqrt()
            r_inv = 1.0 / (r + eps)
            H = x_flat @ phi_t.t()
            alpha_exp = torch.cat([
                alpha_t[0].expand(N_), alpha_t[1].expand(N_),
                alpha_t[2].expand(nPerm),
            ]).view(1, -1)
            act = r_inv * H * alpha_exp + bias_t.view(1, -1)

            h_pre_out = act[:, :N_].sigmoid()
            h_post_out = 2.0 * act[:, N_:2*N_].sigmoid()
            res_coeff = act[:, 2*N_:2*N_+nPerm].softmax(dim=-1)
            h_res_out = torch.einsum('mr,rij->mij', res_coeff, perms_t)
            x_streams = x_flat.reshape(M, N_, C_)
            h_in_out = (x_streams * h_pre_out.unsqueeze(-1)).sum(dim=1)

            # Reshape
            h_in_out   = h_in_out.reshape(B_, S_, C_)
            h_post_out = h_post_out.reshape(B_, S_, N_)
            h_res_out  = h_res_out.reshape(B_, S_, N_, N_)

            # Loss = weighted sum of outputs with the given gradients
            loss = ((h_in_out * ghin_t).sum() +
                    (h_post_out * ghp_t).sum() +
                    (h_res_out * ghr_t).sum())

            grads = torch.autograd.grad(
                loss, [x_t, phi_t, alpha_t, bias_t],
                retain_graph=False, allow_unused=False,
            )

        return (grads[0].to(orig_device), grads[1].to(orig_device),
                grads[2].to(orig_device), grads[3].to(orig_device))


# =========================================================================
# Verification entry-points
# =========================================================================

def get_input_groups():
    """Read JSON test cases and generate input tensors.

    Returns a list of input groups, each being a list of 11 tensors:
      [grad_h_in, grad_h_post, grad_h_res, x, phi, alpha, bias,
       h_pre, hc_before_norm, inv_rms, perm_mats]

    h_pre, hc_before_norm, inv_rms are placeholder values — both models
    ignore them and recompute from the forward pass.
    Permutation matrices are pre-computed via get_permutation_matrices().
    """
    json_path = os.path.join(os.path.dirname(__file__), "mhc_pre_cmhc_backward.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    torch.manual_seed(42)
    input_groups = []
    for case in cases:
        inputs_info = case["inputs"]
        # Extract shapes and dtypes from JSON
        get_info = lambda name: next(i for i in inputs_info if i["name"] == name)

        x_info   = get_info("x")
        ghi_info = get_info("grad_h_in")
        dtype_x  = dtype_map[x_info["dtype"]]
        device   = 'cpu'

        # Generate tensors from JSON shapes
        grad_h_in  = torch.randn(ghi_info["shape"], dtype=dtype_map[ghi_info["dtype"]], device=device)
        grad_h_post = torch.randn(get_info("grad_h_post")["shape"], dtype=torch.float32, device=device)
        grad_h_res  = torch.randn(get_info("grad_h_res")["shape"], dtype=torch.float32, device=device)
        x           = torch.randn(x_info["shape"], dtype=dtype_x, device=device)
        phi         = torch.randn(get_info("phi")["shape"], dtype=torch.float32, device=device)
        alpha       = torch.ones(get_info("alpha")["shape"], dtype=torch.float32, device=device) * 0.01
        bias        = torch.zeros(get_info("bias")["shape"], dtype=torch.float32, device=device)
        h_pre       = torch.rand(get_info("h_pre")["shape"], dtype=torch.float32, device=device)
        hc_before_norm = torch.randn(get_info("hc_before_norm")["shape"], dtype=torch.float32, device=device)
        inv_rms     = torch.rand(get_info("inv_rms")["shape"], dtype=torch.float32, device=device) * 0.1 + 0.01

        # N is the last dim of h_pre; perm_mats shape = (N!, N, N)
        N = get_info("perm_mats")["shape"][1]
        perm_mats = get_permutation_matrices(N, device, torch.float32)

        input_groups.append([grad_h_in, grad_h_post, grad_h_res,
                             x, phi, alpha, bias,
                             h_pre, hc_before_norm, inv_rms,
                             perm_mats])

    return input_groups


def get_init_inputs():
    """Constructor args for Model()."""
    return []


# =========================================================================
# Smoke test
# =========================================================================

if __name__ == "__main__":
    print("=== mhc_pre_cmhc_backward Reference — smoke test ===\n")
    torch.manual_seed(42)

    model = Model(N=4, C=64)
    model.eval()

    for idx, inputs in enumerate(get_input_groups()):
        with torch.no_grad():
            grad_x, grad_phi, grad_alpha, grad_bias = model(*inputs)

        print(f"Case {idx}: x={list(inputs[3].shape)}  dtype={inputs[3].dtype}")
        print(f"  grad_x     shape={list(grad_x.shape)}     sum={grad_x.sum().item():.4f}")
        print(f"  grad_phi   shape={list(grad_phi.shape)}   sum={grad_phi.sum().item():.4f}")
        print(f"  grad_alpha shape={list(grad_alpha.shape)} sum={grad_alpha.sum().item():.4f}")
        print(f"  grad_bias  shape={list(grad_bias.shape)}  sum={grad_bias.sum().item():.4f}")
        print()
