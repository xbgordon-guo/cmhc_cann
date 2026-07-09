#!/usr/bin/env python3
"""PyTorch reference backward for cmhc (Softmax-Permutation gradient).

Validates the backward math before implementing the AscendC kernel.
The cmhc kernel does NOT use gamma — this reference matches the kernel, not model.py.

Forward (from mhc_pre_cmhc_torch_host.cpp):
  1. x_flat = x.reshape(M, nC).float()           # M = B*S, nC = N*C
  2. r_inv = 1 / (sqrt(mean(x_flat²)) + eps)      # RMS norm factor
  3. H = x_flat @ phi.T                            # (M, outDim)  outDim = nPerm + 2N
  4. act = (r_inv * H) * alpha_part + bias         # alpha_part = [α0]*N + [α1]*N + [α2]*nPerm
  5. h_pre = sigmoid(act[:, :N])
  6. h_post = 2 * sigmoid(act[:, N:2N])
  7. res_coeff = softmax(act[:, 2N:2N+nPerm])
  8. h_res = einsum('mr,rij->mij', res_coeff, perms)
  9. h_in = sum_n(x_streams[n] * h_pre[n])

Backward:
  Given: grad_h_in, grad_h_post, grad_h_res, x, phi, alpha, bias,
         h_pre, hc_before_norm(=H), inv_rms(=r_inv)
  Compute: grad_x, grad_phi, grad_alpha, grad_bias
"""

import itertools
import math
import torch


# =========================================================================
# Constants
# =========================================================================
def get_permutation_matrices(n: int, device, dtype):
    """Generate all n! permutation matrices of shape (n!, n, n).

    Permutation matrices are computed on the Python side and passed as input,
    matching the reference cmhc_cann/cmhc pattern.
    """
    perms = []
    for p in itertools.permutations(range(n)):
        mat = torch.zeros(n, n, device=device, dtype=dtype)
        for i, j in enumerate(p):
            mat[i, j] = 1.0
        perms.append(mat)
    return torch.stack(perms, dim=0)  # (n!, n, n)


# =========================================================================
# Forward (kernel-compatible: no gamma)
# =========================================================================
def mhc_pre_cmhc_forward(x, phi, alpha, bias, perms, eps=1e-6):
    """Forward matching mhc_pre_cmhc_torch_host.cpp exactly.

    Args:
        x:     (B, S, N, C)
        phi:   (outDim, N*C)  where outDim = nPerm + 2N
        alpha: (3,)
        bias:  (outDim,)
        perms: (nPerm, N, N)  permutation matrices

    Returns:
        h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, res_coeff,
        h_pre_raw, h_post_raw, h_res_logits, x_flat, x_streams
    """
    B, S, N, C = x.shape
    nC = N * C
    M = B * S
    outDim = phi.shape[0]
    nPerm = outDim - 2 * N

    x_flat = x.reshape(M, nC).float()

    # RMS norm
    mean_sq = (x_flat * x_flat).mean(dim=-1, keepdim=True)  # (M, 1)
    r = mean_sq.sqrt()
    r_inv = 1.0 / (r + eps)  # (M, 1)

    # MatMul
    H = x_flat @ phi.t()  # (M, outDim)

    # Alpha expansion
    alpha_exp = torch.cat([
        alpha[0].expand(N),
        alpha[1].expand(N),
        alpha[2].expand(nPerm),
    ]).view(1, -1)  # (1, outDim)

    act = r_inv * H * alpha_exp + bias.view(1, -1)  # (M, outDim)

    # Split
    h_pre_raw = act[:, :N]
    h_post_raw = act[:, N:2*N]
    h_res_logits = act[:, 2*N:2*N+nPerm]

    # Activations
    h_pre = h_pre_raw.sigmoid()
    h_post = 2.0 * h_post_raw.sigmoid()
    res_coeff = h_res_logits.softmax(dim=-1)

    # Permutation einsum
    h_res = torch.einsum('mr,rij->mij', res_coeff, perms)

    # Weighted sum
    x_streams = x_flat.reshape(M, N, C)
    h_in = (x_streams * h_pre.unsqueeze(-1)).sum(dim=1)  # (M, C)

    # Reshape
    h_in = h_in.reshape(B, S, C)
    h_post = h_post.reshape(B, S, N)
    h_res = h_res.reshape(B, S, N, N)
    h_pre_out = h_pre.reshape(B, S, N)
    hc_before_norm = H.reshape(B, S, -1)
    inv_rms_out = r_inv.reshape(B, S)

    return (h_in, h_post, h_res, h_pre_out, hc_before_norm, inv_rms_out,
            res_coeff, h_pre_raw, h_post_raw, h_res_logits, x_flat, x_streams)


# =========================================================================
# Manual backward (for kernel implementation reference)
# =========================================================================
def mhc_pre_cmhc_backward_manual(x, phi, alpha, bias, perms,
                         grad_h_in, grad_h_post, grad_h_res,
                         h_pre, hc_before_norm, inv_rms,
                         res_coeff, h_pre_raw, h_post_raw, h_res_logits,
                         x_flat, x_streams, eps=1e-6):
    """Manual backward pass for cmhc.

    This is the reference the AscendC kernel must reproduce.
    All tensors are in (M, ...) flattened form where applicable.

    Args:
        All forward tensors + output gradients.
        grad_h_in:   (B, S, C)
        grad_h_post: (B, S, N)
        grad_h_res:  (B, S, N, N)
        h_pre:       (B, S, N)   — forward sigmoid output
        hc_before_norm: (B, S, outDim) — forward H = x @ phi.T
        inv_rms:     (B, S)      — forward r_inv
        res_coeff:   (M, nPerm)  — forward softmax output
        h_pre_raw:   (M, N)      — forward pre-sigmoid
        h_post_raw:  (M, N)      — forward pre-sigmoid (for h_post)
        h_res_logits:(M, nPerm)  — forward pre-softmax
        x_flat:      (M, nC)
        x_streams:   (M, N, C)

    Returns:
        grad_x:     (B, S, N, C)
        grad_phi:   (outDim, nC)
        grad_alpha: (3,)
        grad_bias:  (outDim,)
    """
    B, S, N = h_pre.shape
    C = grad_h_in.shape[-1]
    nC = N * C
    M = B * S
    outDim = hc_before_norm.shape[-1]
    nPerm = outDim - 2 * N

    # Flatten everything to (M, ...)
    grad_h_in_2d = grad_h_in.reshape(M, C)       # (M, C)
    grad_h_post_2d = grad_h_post.reshape(M, N)    # (M, N)
    grad_h_res_2d = grad_h_res.reshape(M, N, N)   # (M, N, N)
    h_pre_2d = h_pre.reshape(M, N)                # (M, N)
    H = hc_before_norm.reshape(M, outDim)          # (M, outDim)
    r_inv = inv_rms.reshape(M, 1)                  # (M, 1)

    # =====================================================================
    # 1. Backward through weighted sum: h_in = sum_n(x_streams[n] * h_pre[n])
    #    grad_x_streams[n,:] = grad_h_in * h_pre[n]      (broadcast h_pre[n] over C)
    #    grad_h_pre[n] = sum_c(grad_h_in * x_streams[n])
    # =====================================================================
    grad_x_streams = grad_h_in_2d.unsqueeze(1) * h_pre_2d.unsqueeze(-1)  # (M, N, C)
    grad_h_pre = (grad_h_in_2d.unsqueeze(1) * x_streams).sum(dim=-1)      # (M, N)

    # Add grad from the output grad_h_post (h_post itself is an output)
    grad_h_post_total = grad_h_post_2d  # (M, N)

    # =====================================================================
    # 2. Backward through sigmoid: h_pre = sigmoid(h_pre_raw)
    #    dsigmoid/dx = sigmoid(x) * (1 - sigmoid(x))
    #    grad_h_pre_raw = grad_h_pre * h_pre * (1 - h_pre)
    # =====================================================================
    sigmoid_grad_pre = h_pre_2d * (1.0 - h_pre_2d)  # (M, N)
    grad_h_pre_raw = grad_h_pre * sigmoid_grad_pre   # (M, N)

    # =====================================================================
    # 3. Backward through h_post = 2 * sigmoid(h_post_raw)
    #    Let s = sigmoid(x), h_post = 2s
    #    d(h_post)/dx = 2 * s * (1-s) = h_post * (1 - h_post/2)
    # =====================================================================
    h_post_sigmoid = h_post_raw.sigmoid()  # (M, N) — recompute forward internals
    # Verify: h_post = 2*sigmoid(h_post_raw) → sigmoid = h_post/2
    sigmoid_grad_post = h_post_sigmoid * (1.0 - h_post_sigmoid)  # s*(1-s)
    grad_h_post_raw = grad_h_post_total * 2.0 * sigmoid_grad_post  # (M, N)

    # =====================================================================
    # 4. Backward through softmax-permutation:
    #    res_coeff = softmax(h_res_logits)
    #    h_res = einsum('mr,rij->mij', res_coeff, perms)
    #
    #    4a. grad_res_coeff[r] = sum_ij grad_h_res[i,j] * perms[r,i,j]
    #        = einsum('mij,rij->mr', grad_h_res, perms)
    #    4b. grad_h_res_logits = standard softmax backward
    # =====================================================================
    # 4a: einsum backward through permutation sum
    grad_res_coeff = torch.einsum('mij,rij->mr', grad_h_res_2d, perms)  # (M, nPerm)

    # 4b: Softmax backward
    # grad_logits = softmax_output * (grad_output - sum(grad_output * softmax_output))
    weighted = (grad_res_coeff * res_coeff).sum(dim=-1, keepdim=True)  # (M, 1)
    grad_h_res_logits = res_coeff * (grad_res_coeff - weighted)         # (M, nPerm)

    # =====================================================================
    # 5. Backward through act = r_inv * H * alpha_exp + bias
    #    Combine grad from pre, post, res:
    #    grad_act[:, :N] = grad_h_pre_raw
    #    grad_act[:, N:2N] = grad_h_post_raw
    #    grad_act[:, 2N:2N+nPerm] = grad_h_res_logits
    # =====================================================================
    grad_act = torch.cat([grad_h_pre_raw, grad_h_post_raw, grad_h_res_logits], dim=1)  # (M, outDim)

    # alpha_exp for chain rule
    alpha_exp = torch.cat([
        alpha[0].expand(N),
        alpha[1].expand(N),
        alpha[2].expand(nPerm),
    ]).view(1, -1)  # (1, outDim)

    # act = r_inv * H * alpha_exp + bias
    # grad_H[:,i] = grad_act[:,i] * alpha_exp[i] * r_inv
    grad_H = grad_act * alpha_exp * r_inv  # (M, outDim)

    # grad_r_inv = sum_i(grad_act[:,i] * H[:,i] * alpha_exp[i])
    grad_r_inv = (grad_act * H * alpha_exp).sum(dim=-1, keepdim=True)  # (M, 1)

    # grad_bias[i] = sum_m grad_act[m,i]
    grad_bias = grad_act.sum(dim=0)  # (outDim,)

    # grad_alpha: need to accumulate per segment
    grad_alpha_0 = (grad_act[:, :N] * r_inv * H[:, :N]).sum()
    grad_alpha_1 = (grad_act[:, N:2*N] * r_inv * H[:, N:2*N]).sum()
    grad_alpha_2 = (grad_act[:, 2*N:2*N+nPerm] * r_inv * H[:, 2*N:2*N+nPerm]).sum()
    grad_alpha = torch.stack([grad_alpha_0, grad_alpha_1, grad_alpha_2])

    # =====================================================================
    # 6. Backward through H = x_flat @ phi.T:
    #    grad_x_flat_matmul = grad_H @ phi
    #    grad_phi = grad_H.T @ x_flat
    # =====================================================================
    grad_x_flat_matmul = grad_H @ phi  # (M, nC)
    grad_phi = grad_H.t() @ x_flat     # (outDim, nC)

    # =====================================================================
    # 7. Backward through RMS norm: r_inv = 1/(sqrt(mean(x²)) + eps)
    #    r = sqrt(mean(x_flat²)), r_inv = 1/(r + eps)
    #    d(r_inv)/dr = -1/(r+eps)² = -r_inv²
    #    dr/d(mean_sq) = 1/(2r)
    #    d(mean_sq)/d(x_sq) = 1/nC  (broadcast)
    #    d(x_sq)/d(x) = 2x
    #
    #    grad_x_rms = -x_flat * grad_r_inv * r_inv² / (r * nC)
    #    where r = sqrt(mean(x_flat²))
    # =====================================================================
    mean_sq = (x_flat * x_flat).mean(dim=-1, keepdim=True)  # (M, 1)
    r = mean_sq.sqrt()                                       # (M, 1)
    grad_x_flat_rms = -x_flat * grad_r_inv * (r_inv * r_inv) / (r * nC)

    # =====================================================================
    # 8. Total grad_x
    # =====================================================================
    grad_x_flat = grad_x_flat_matmul + grad_x_flat_rms  # (M, nC)
    # Add grad_x_streams contribution (reshaped back)
    grad_x_flat = grad_x_flat + grad_x_streams.reshape(M, nC)  # (M, nC)

    grad_x = grad_x_flat.reshape(B, S, N, C)

    return grad_x, grad_phi, grad_alpha, grad_bias


# =========================================================================
# Numerical verification using torch.autograd
# =========================================================================
class MhcPreCmhcFunction(torch.autograd.Function):
    """Custom autograd Function matching cmhc kernel for gradcheck."""

    @staticmethod
    def forward(ctx, x, phi, alpha, bias, perms, eps=1e-6):
        B, S, N, C = x.shape
        nC = N * C
        M = B * S
        outDim = phi.shape[0]
        nPerm = outDim - 2 * N

        x_flat = x.reshape(M, nC)

        # RMS norm
        mean_sq = (x_flat * x_flat).mean(dim=-1, keepdim=True)
        r = mean_sq.sqrt()
        r_inv = 1.0 / (r + eps)

        # MatMul
        H = x_flat @ phi.t()

        # Alpha expansion
        alpha_exp = torch.cat([
            alpha[0].expand(N),
            alpha[1].expand(N),
            alpha[2].expand(nPerm),
        ]).view(1, -1)

        act = r_inv * H * alpha_exp + bias.view(1, -1)

        h_pre_raw = act[:, :N]
        h_post_raw = act[:, N:2*N]
        h_res_logits = act[:, 2*N:2*N+nPerm]

        h_pre = h_pre_raw.sigmoid()
        h_post = 2.0 * h_post_raw.sigmoid()
        res_coeff = h_res_logits.softmax(dim=-1)

        h_res = torch.einsum('mr,rij->mij', res_coeff, perms)

        x_streams = x_flat.reshape(M, N, C)
        h_in = (x_streams * h_pre.unsqueeze(-1)).sum(dim=1)

        # Reshape
        h_in = h_in.reshape(B, S, C)
        h_post = h_post.reshape(B, S, N)
        h_res = h_res.reshape(B, S, N, N)
        h_pre_out = h_pre.reshape(B, S, N)
        hc_before_norm = H.reshape(B, S, -1)
        inv_rms_out = r_inv.reshape(B, S)

        # Save for backward
        ctx.save_for_backward(x, phi, alpha, bias, perms)
        ctx.eps = eps
        ctx.intermediates = {
            'h_pre': h_pre_out,
            'hc_before_norm': hc_before_norm,
            'inv_rms': inv_rms_out,
            'res_coeff': res_coeff,
            'h_pre_raw': h_pre_raw,
            'h_post_raw': h_post_raw,
            'h_res_logits': h_res_logits,
            'x_flat': x_flat,
            'x_streams': x_streams,
        }

        return h_in, h_post, h_res

    @staticmethod
    def backward(ctx, grad_h_in, grad_h_post, grad_h_res):
        x, phi, alpha, bias, perms = ctx.saved_tensors
        ic = ctx.intermediates
        eps = ctx.eps

        grad_x, grad_phi, grad_alpha, grad_bias = mhc_pre_cmhc_backward_manual(
            x, phi, alpha, bias, perms,
            grad_h_in, grad_h_post, grad_h_res,
            ic['h_pre'], ic['hc_before_norm'], ic['inv_rms'],
            ic['res_coeff'], ic['h_pre_raw'], ic['h_post_raw'],
            ic['h_res_logits'], ic['x_flat'], ic['x_streams'],
            eps=eps,
        )

        # perms and eps don't need gradients
        return grad_x, grad_phi, grad_alpha, grad_bias, None, None


# =========================================================================
# Verification
# =========================================================================
def run_verification():
    """Verify the manual backward against torch.autograd."""
    torch.manual_seed(42)

    B, S, N, C = 2, 4, 4, 64
    # Use small C so it's fast for testing
    nC = N * C
    nPerm = math.factorial(N)   # 24
    outDim = nPerm + 2 * N       # 32

    device = 'cpu'
    dtype = torch.float32

    # Create inputs (matching kernel: no gamma)
    x = torch.randn(B, S, N, C, device=device, dtype=dtype, requires_grad=True)
    phi = torch.randn(outDim, nC, device=device, dtype=dtype, requires_grad=True)
    alpha = torch.ones(3, device=device, dtype=dtype, requires_grad=True) * 0.01
    bias = torch.zeros(outDim, device=device, dtype=dtype, requires_grad=True) * 0.01
    perms = get_permutation_matrices(N, device, dtype)

    # --- Method 1: torch.autograd backward (using torch.autograd.grad) ---
    h_in, h_post, h_res = MhcPreCmhcFunction.apply(x, phi, alpha, bias, perms, 1e-6)

    # Use a scalar loss for backward
    loss_auto = h_in.sum() + h_post.sum() + h_res.sum()
    grad_x_auto, grad_phi_auto, grad_alpha_auto, grad_bias_auto = torch.autograd.grad(
        loss_auto, [x, phi, alpha, bias], retain_graph=True,
    )

    # --- Method 2: Manual backward ---
    # Recompute forward to get intermediates
    (h_in2, h_post2, h_res2, h_pre_out, hc_before_norm, inv_rms_out,
     res_coeff, h_pre_raw, h_post_raw, h_res_logits,
     x_flat, x_streams) = mhc_pre_cmhc_forward(x.detach(), phi.detach(),
                                        alpha.detach(), bias.detach(), perms)

    grad_h_in = torch.ones_like(h_in2)
    grad_h_post = torch.ones_like(h_post2)
    grad_h_res = torch.ones_like(h_res2)

    grad_x_man, grad_phi_man, grad_alpha_man, grad_bias_man = mhc_pre_cmhc_backward_manual(
        x.detach(), phi.detach(), alpha.detach(), bias.detach(), perms,
        grad_h_in, grad_h_post, grad_h_res,
        h_pre_out, hc_before_norm, inv_rms_out,
        res_coeff, h_pre_raw, h_post_raw, h_res_logits,
        x_flat, x_streams,
    )

    # --- Compare ---
    print("=" * 70)
    print("cmhc backward — Manual vs Autograd Comparison")
    print("=" * 70)

    def compare(name, a, b, atol=1e-5):
        diff = (a - b).abs()
        ok = torch.allclose(a, b, atol=atol)
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name:16s}  max_diff={diff.max().item():.2e}  "
              f"mean_diff={diff.mean().item():.2e}  [{status}]")
        if not ok:
            # Show worst elements
            worst_idx = diff.argmax()
            print(f"     Worst: auto={a.flatten()[worst_idx].item():.6f}  "
                  f"manual={b.flatten()[worst_idx].item():.6f}")
        return ok

    all_ok = True
    all_ok &= compare("grad_x",     grad_x_auto,     grad_x_man)
    all_ok &= compare("grad_phi",   grad_phi_auto,   grad_phi_man)
    all_ok &= compare("grad_alpha", grad_alpha_auto, grad_alpha_man)
    all_ok &= compare("grad_bias",  grad_bias_auto,  grad_bias_man)

    # --- Also verify with torch.autograd.gradcheck ---
    print(f"\n--- torch.autograd.gradcheck ---")
    x_gc = torch.randn(B, S, N, C, dtype=torch.float64, requires_grad=True)
    phi_gc = torch.randn(outDim, nC, dtype=torch.float64, requires_grad=True)
    alpha_gc = torch.ones(3, dtype=torch.float64, requires_grad=True) * 0.01
    bias_gc = torch.zeros(outDim, dtype=torch.float64, requires_grad=True) * 0.01
    perms_gc = get_permutation_matrices(N, 'cpu', torch.float64)

    gc_ok = torch.autograd.gradcheck(
        lambda x, phi, alpha, bias: MhcPreCmhcFunction.apply(x, phi, alpha, bias, perms_gc, 1e-6),
        (x_gc, phi_gc, alpha_gc, bias_gc),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )
    print(f"  gradcheck: {'✓ PASS' if gc_ok else '✗ FAIL'}")

    all_ok &= gc_ok

    print(f"\n{'='*70}")
    print(f"  Overall: {'✓ ALL PASS' if all_ok else '✗ SOME FAILED'}")
    print(f"{'='*70}")

    return all_ok


# =========================================================================
# BF16/FP16 cross-check
# =========================================================================
def run_dtype_check():
    """Verify backward works across dtypes (FP32/FP16/BF16)."""
    print("\n" + "=" * 70)
    print("cmhc backward — Multi-dtype verification")
    print("=" * 70)

    B, S, N, C = 2, 4, 4, 64
    nC = N * C
    nPerm = math.factorial(N)
    outDim = nPerm + 2 * N

    for test_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        torch.manual_seed(42)
        device = 'cpu'

        x = torch.randn(B, S, N, C, device=device, dtype=test_dtype)
        phi = torch.randn(outDim, nC, device=device, dtype=torch.float32)
        alpha = torch.ones(3, device=device, dtype=torch.float32) * 0.01
        bias = torch.zeros(outDim, device=device, dtype=torch.float32) * 0.01
        perms = get_permutation_matrices(N, device, torch.float32)

        # Forward in fp32
        (h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms,
         res_coeff, h_pre_raw, h_post_raw, h_res_logits,
         x_flat, x_streams) = mhc_pre_cmhc_forward(x, phi, alpha, bias, perms)

        grad_h_in = torch.ones_like(h_in)
        grad_h_post = torch.ones_like(h_post)
        grad_h_res = torch.ones_like(h_res)

        grad_x, grad_phi, grad_alpha, grad_bias = mhc_pre_cmhc_backward_manual(
            x, phi, alpha, bias, perms,
            grad_h_in, grad_h_post, grad_h_res,
            h_pre, hc_before_norm, inv_rms,
            res_coeff, h_pre_raw, h_post_raw, h_res_logits,
            x_flat, x_streams,
        )

        has_nan = (torch.isnan(grad_x).any() or torch.isnan(grad_phi).any() or
                   torch.isnan(grad_alpha).any() or torch.isnan(grad_bias).any())
        has_inf = (torch.isinf(grad_x).any() or torch.isinf(grad_phi).any() or
                   torch.isinf(grad_alpha).any() or torch.isinf(grad_bias).any())

        status = "✓ OK" if (not has_nan and not has_inf) else "✗ FAIL"
        print(f"  {str(test_dtype):16s}  NaN={has_nan}  Inf={has_inf}  "
              f"grad_x range=[{grad_x.min().item():.4f},{grad_x.max().item():.4f}]  "
              f"[{status}]")


if __name__ == "__main__":
    ok = run_verification()
    run_dtype_check()

    if ok:
        print("\n✓ Backward math verified. Ready for kernel implementation.")
    else:
        print("\n✗ Backward math has errors. Fix before kernel implementation.")
        exit(1)
