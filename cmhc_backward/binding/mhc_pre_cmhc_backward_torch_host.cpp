/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * mhc_pre_cmhc_backward torch.ops host — PyTorch-native backward using at:: ops on NPU.
 *
 * Matches the AscendC kernel algorithm exactly:
 *   1. Backward through weighted sum (h_in = sum_n x[n]*h_pre[n])
 *   2. Sigmoid backward for pre / post
 *   3. Softmax-permutation backward (replaces Sinkhorn backward)
 *   4. RMS norm backward
 *   5. MatMul backward (grad_x_matmul = grad_H @ phi, grad_phi = grad_H^T @ x)
 *   6. Scale / bias gradient accumulation
 *
 * Permutation matrices are passed from Python (perm_mats), enabling
 * gamma-blending with the uniform distribution without recompilation.
 * This matches the reference cmhc_cann/cmhc pattern.
 *
 * Verified against torch.autograd.gradcheck (see backward_reference.py).
 */

#include <cmath>
#include <vector>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include "ops.h"

namespace mhc_pre_cmhc_backward_binding {

std::vector<at::Tensor> mhc_pre_cmhc_backward(
    const at::Tensor &grad_h_in,
    const at::Tensor &grad_h_post,
    const at::Tensor &grad_h_res,
    const at::Tensor &x,
    const at::Tensor &phi,
    const at::Tensor &alpha,
    const at::Tensor &bias,
    const at::Tensor &h_pre,
    const at::Tensor &hc_before_norm,
    const at::Tensor &inv_rms,
    const at::Tensor &perm_mats,
    double hc_eps)
{
    // Validate
    TORCH_CHECK(x.dim() == 4, "x must be [B, S, N, C]");
    TORCH_CHECK(grad_h_in.dim() == 3, "grad_h_in must be [B, S, C]");
    TORCH_CHECK(grad_h_post.dim() == 3, "grad_h_post must be [B, S, N]");
    TORCH_CHECK(grad_h_res.dim() == 4, "grad_h_res must be [B, S, N, N]");

    auto x_sizes = x.sizes();
    int64_t B = x_sizes[0], S = x_sizes[1], N = x_sizes[2], C = x_sizes[3];
    int64_t M = B * S, nC = N * C;
    int64_t outDim = phi.sizes()[0];  // = nPerm + 2*N
    int64_t nPerm = outDim - 2 * N;   // = 24 for N=4
    float eps = static_cast<float>(hc_eps);
    auto opts_f32 = x.options().dtype(at::kFloat);

    // Validate perm_mats shape: (n!, N, N)
    TORCH_CHECK(perm_mats.dim() == 3, "perm_mats must be [n!, N, N]");
    TORCH_CHECK(perm_mats.size(0) == nPerm && perm_mats.size(1) == N && perm_mats.size(2) == N,
                "perm_mats shape must be (", nPerm, ", ", N, ", ", N, ")");

    // =========================================================================
    // 0. Flatten all input tensors to (M, ...)
    // =========================================================================
    at::Tensor grad_h_in_2d  = grad_h_in.reshape({M, C});          // (M, C)
    at::Tensor grad_h_post_2d = grad_h_post.reshape({M, N});       // (M, N)
    at::Tensor grad_h_res_2d = grad_h_res.reshape({M, N, N});      // (M, N, N)
    at::Tensor x_flat   = x.reshape({M, nC}).to(at::kFloat);       // (M, nC)
    at::Tensor h_pre_2d = h_pre.reshape({M, N});                   // (M, N)
    at::Tensor H = hc_before_norm.reshape({M, outDim});            // (M, outDim)
    at::Tensor r_inv = inv_rms.reshape({M, 1});                    // (M, 1)
    at::Tensor x_streams = x_flat.reshape({M, N, C});              // (M, N, C)

    // =========================================================================
    // 1. Backward through weighted sum: h_in = sum_n(x_streams[n] * h_pre[n])
    //    grad_x_streams[n,:] = grad_h_in * h_pre[n]
    //    grad_h_pre = sum_c(grad_h_in * x_streams[n])
    // =========================================================================
    at::Tensor grad_x_streams = grad_h_in_2d.unsqueeze(1) * h_pre_2d.unsqueeze(-1);  // (M, N, C)
    at::Tensor grad_h_pre = (grad_h_in_2d.unsqueeze(1) * x_streams).sum(-1);          // (M, N)

    // =========================================================================
    // 2. Sigmoid backward: h_pre = sigmoid(h_pre_raw)
    //    dsigmoid/dx = sigmoid(x)*(1-sigmoid(x)) = h_pre*(1-h_pre)
    // =========================================================================
    at::Tensor sigmoid_grad_pre = h_pre_2d * (1.0f - h_pre_2d);     // (M, N)
    at::Tensor grad_h_pre_raw = grad_h_pre * sigmoid_grad_pre;       // (M, N)

    // =========================================================================
    // 3. Backward through h_post = 2 * sigmoid(h_post_raw)
    //    Recompute h_post_raw from hc_before_norm, r_inv, alpha, bias
    //    h_post_raw = r_inv * H[:, N:2N] * alpha[1] + bias[N:2N]
    //    Then sigmoid backward with factor 2
    // =========================================================================
    at::Tensor post_raw = r_inv * H.slice(1, N, 2*N) * alpha[1] + bias.slice(0, N, 2*N);  // (M, N)
    at::Tensor h_post_sigmoid = post_raw.sigmoid();                  // (M, N)
    // d/dx 2*sigmoid(x) = 2 * sigmoid(x) * (1-sigmoid(x))
    at::Tensor sigmoid_grad_post = h_post_sigmoid * (1.0f - h_post_sigmoid);
    at::Tensor grad_h_post_raw = grad_h_post_2d * 2.0f * sigmoid_grad_post;  // (M, N)

    // =========================================================================
    // 4. Softmax-permutation backward (replaces Sinkhorn backward)
    //
    //    Forward: res_coeff = softmax(h_res_logits)
    //             h_res = einsum('mr,rij->mij', res_coeff, perms)
    //
    //    4a. Recompute res_coeff from forward intermediates
    //        h_res_logits = r_inv * H[:, 2N:2N+nPerm] * alpha[2] + bias[2N:2N+nPerm]
    //    4b. grad_res_coeff = einsum('mij,rij->mr', grad_h_res, perm_mats)
    //        Uses perm_mats passed from Python (pre-computed, possibly gamma-blended)
    //    4c. grad_h_res_logits = softmax_backward(res_coeff, grad_res_coeff)
    // =========================================================================
    // 4a. Recompute softmax forward
    at::Tensor h_res_logits = r_inv * H.slice(1, 2*N, 2*N + nPerm) * alpha[2]
                              + bias.slice(0, 2*N, 2*N + nPerm);     // (M, nPerm)
    at::Tensor res_coeff = h_res_logits.softmax(-1);                  // (M, nPerm)

    // 4b. grad_res_coeff = einsum('mij,rij->mr', grad_h_res, perm_mats)
    //     Uses at::einsum with perm_mats passed from Python side.
    //     This replaces the old hardcoded kPermIdx gather-based approach.
    at::Tensor perms = perm_mats.to(at::kFloat);  // ensure fp32
    at::Tensor grad_res_coeff = at::einsum("mij,rij->mr", {grad_h_res_2d, perms});  // (M, nPerm)

    // 4c. Softmax backward: grad_logits = res_coeff * (grad_res_coeff - sum(grad_res_coeff * res_coeff))
    at::Tensor weighted = (grad_res_coeff * res_coeff).sum(-1, true);  // (M, 1)
    at::Tensor grad_h_res_logits = res_coeff * (grad_res_coeff - weighted);  // (M, nPerm)

    // =========================================================================
    // 5. Combine grad_act = [grad_h_pre_raw, grad_h_post_raw, grad_h_res_logits]
    // =========================================================================
    at::Tensor grad_act = at::cat({grad_h_pre_raw, grad_h_post_raw, grad_h_res_logits}, -1);  // (M, outDim)

    // Alpha expansion
    at::Tensor alpha_exp = alpha.repeat_interleave(
        at::tensor({N, N, nPerm}, at::TensorOptions().dtype(at::kLong).device(alpha.device())));
    alpha_exp = alpha_exp.unsqueeze(0);  // (1, outDim)

    // act = r_inv * H * alpha_exp + bias
    // grad_H = grad_act * alpha_exp * r_inv
    at::Tensor grad_H = grad_act * alpha_exp * r_inv;  // (M, outDim)

    // grad_r_inv = sum(grad_act * H * alpha_exp, dim=-1)
    at::Tensor grad_r_inv = (grad_act * H * alpha_exp).sum(-1, true);  // (M, 1)

    // grad_bias = sum over M
    at::Tensor grad_bias = grad_act.sum(0);  // (outDim,)

    // grad_alpha: reduce over M for each segment
    at::Tensor ga0 = (grad_act.slice(1, 0, N) * r_inv * H.slice(1, 0, N)).sum();
    at::Tensor ga1 = (grad_act.slice(1, N, 2*N) * r_inv * H.slice(1, N, 2*N)).sum();
    at::Tensor ga2 = (grad_act.slice(1, 2*N, 2*N + nPerm) * r_inv * H.slice(1, 2*N, 2*N + nPerm)).sum();
    at::Tensor grad_alpha = at::stack({ga0, ga1, ga2});  // (3,)

    // =========================================================================
    // 6. MatMul backward: H = x_flat @ phi.T
    //    grad_x_matmul = grad_H @ phi
    //    grad_phi = grad_H.T @ x_flat
    // =========================================================================
    at::Tensor grad_x_matmul = at::matmul(grad_H, phi);  // (M, nC)
    at::Tensor grad_phi = at::matmul(grad_H.t(), x_flat); // (outDim, nC)

    // =========================================================================
    // 7. RMS norm backward: r_inv = 1/(sqrt(mean(x²)) + eps)
    //    grad_x_rms = -x_flat * grad_r_inv * r_inv² / (r * nC)
    //    where r = sqrt(mean(x_flat²))
    // =========================================================================
    at::Tensor mean_sq = (x_flat * x_flat).mean(-1, true);  // (M, 1)
    at::Tensor r = mean_sq.sqrt();                           // (M, 1)
    float inv_nC = 1.0f / static_cast<float>(nC);
    at::Tensor grad_x_rms = -x_flat * grad_r_inv * (r_inv * r_inv) / r * inv_nC;

    // =========================================================================
    // 8. Total grad_x = matmul contribution + RMS contribution + stream contribution
    // =========================================================================
    at::Tensor grad_x_flat = grad_x_matmul + grad_x_rms + grad_x_streams.reshape({M, nC});
    at::Tensor grad_x = grad_x_flat.reshape({B, S, N, C}).to(x.dtype());

    return {grad_x, grad_phi, grad_alpha, grad_bias};
}

}  // namespace mhc_pre_cmhc_backward_binding
