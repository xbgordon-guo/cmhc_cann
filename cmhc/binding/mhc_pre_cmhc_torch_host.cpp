/*
 * mhc_pre_cmhc torch.ops host — PyTorch-native implementation using at:: ops on NPU.
 *
 * Computes exactly the same algorithm as model.py:
 *   1. RMS norm: r_inv = 1 / (sqrt(mean(x²)) + eps)
 *   2. MatMul: H = (gamma * x) @ phi^T    (phi is outDim x nC)
 *   3. Pre/post: sigmoid(r_inv * H * alpha + bias)
 *   4. Res: softmax over n! permutation logits → Birkhoff-von Neumann
 *   5. Weighted sum: h_in = sum_n(x_streams[n] * h_pre[n])
 *
 * Permutation matrices are passed from Python (perm_mats), enabling
 * gamma-blending with the uniform distribution without recompilation.
 *
 * Birkhoff-von Neumann:
 *     h_res[r] = sum_p softmax_coeff[r][p] * perm_mats[p]
 * where perm_mats is (n!, N, N) pre-computed on the Python side.
 */

#include <cmath>
#include <vector>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include "ops.h"

namespace mhc_pre_cmhc_binding {

std::vector<at::Tensor> mhc_pre_cmhc(
    const at::Tensor &x,
    const at::Tensor &phi,
    const at::Tensor &alpha,
    const at::Tensor &bias,
    const at::Tensor &perm_mats,
    int64_t hc_mult,
    int64_t num_iters,   // unused, kept for API compatibility
    double hc_eps,
    double norm_eps,
    bool need_backward)
{
    TORCH_CHECK(x.dim() == 4, "x must be [B, S, N, C]");
    TORCH_CHECK(x.is_contiguous() && phi.is_contiguous(), "tensors must be contiguous");

    int64_t N = hc_mult;                    // = 4
    auto x_sizes = x.sizes();
    int64_t B = x_sizes[0], S = x_sizes[1], C = x_sizes[3];
    int64_t M = B * S, nC = N * C;
    int64_t nPerm = phi.sizes()[0] - 2 * N;
    float eps = static_cast<float>(norm_eps);
    auto opts_f32 = x.options().dtype(at::kFloat);

    // Validate perm_mats shape: (n!, N, N)
    TORCH_CHECK(perm_mats.dim() == 3, "perm_mats must be [n!, N, N]");
    TORCH_CHECK(perm_mats.size(0) == nPerm && perm_mats.size(1) == N && perm_mats.size(2) == N,
                "perm_mats shape must be (", nPerm, ", ", N, ", ", N, ")");

    // 1. Flatten and cast to fp32
    at::Tensor x_flat = x.reshape({M, nC}).to(at::kFloat);

    // 2. RMS norm: r_inv = 1 / (sqrt(mean(x²)) + eps)
    at::Tensor x_sq = x_flat * x_flat;
    at::Tensor mean_sq = x_sq.mean(/*dim=*/-1, /*keepdim=*/true);
    at::Tensor r = mean_sq.sqrt();
    at::Tensor r_inv = 1.0f / (r + eps);    // (M, 1)

    // 3. MatMul: H = x @ phi^T    phi is (outDim, nC)
    at::Tensor H = at::matmul(x_flat, phi.t());  // (M, outDim)

    // 4. Apply r_inv * alpha + bias
    at::Tensor act = r_inv * H;                      // (M, outDim)
    // alpha is (3,), expand to (1, outDim) with [alpha0]*4, [alpha1]*4, [alpha2]*nPerm
    at::Tensor alpha_exp = alpha.repeat_interleave(
        at::tensor({N, N, nPerm}, at::TensorOptions().dtype(at::kLong).device(alpha.device())));
    act = act * alpha_exp.unsqueeze(0) + bias.unsqueeze(0);  // (M, outDim)

    // 5. Sigmoid for pre/post
    at::Tensor h_pre_raw  = act.slice(/*dim=*/1, 0, N);
    at::Tensor h_post_raw = act.slice(/*dim=*/1, N, 2*N);
    at::Tensor h_pre  = h_pre_raw.sigmoid();              // (M, N)
    at::Tensor h_post = 2.0f * h_post_raw.sigmoid();      // (M, N)

    // 6. Softmax-permutation for h_res
    at::Tensor h_res_logits = act.slice(/*dim=*/1, 2*N, 2*N + nPerm);
    at::Tensor res_coeff = h_res_logits.softmax(/*dim=*/-1);  // (M, nPerm)

    // Permutation matrices passed from Python (pre-computed, possibly gamma-blended)
    // perm_mats: (n!, N, N) → use einsum directly
    at::Tensor perms = perm_mats.to(at::kFloat);  // ensure fp32

    // einsum('mr,rij->mij')  →  (M, N, N)
    // Birkhoff-von Neumann: doubly stochastic matrix as convex combination
    at::Tensor h_res = at::einsum("mr,rij->mij", {res_coeff, perms});

    // 7. Weighted sum: h_in = sum_n(x_streams[n] * h_pre[n])
    at::Tensor x_streams = x_flat.reshape({M, N, C});           // (M, N, C)
    at::Tensor h_in = (x_streams * h_pre.unsqueeze(/*dim=*/-1)).sum(/*dim=*/1);  // (M, C)

    // 8. Reshape outputs to (B, S, ...)
    h_in   = h_in.reshape({B, S, C});
    h_post = h_post.reshape({B, S, N});
    h_res  = h_res.reshape({B, S, N, N});
    h_pre  = h_pre.reshape({B, S, N});
    at::Tensor hc_before_norm = H.reshape({B, S, -1});
    at::Tensor inv_rms_out = r_inv.reshape({B, S});

    return {h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms_out};
}

}  // namespace mhc_pre_cmhc_binding
