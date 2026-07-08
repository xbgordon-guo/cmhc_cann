/*
 * cmhc torch.ops host — PyTorch-native implementation using at:: ops on NPU.
 *
 * Computes exactly the same algorithm as model.py:
 *   1. RMS norm: r_inv = 1 / (sqrt(mean(x²)) + eps)
 *   2. MatMul: H = (gamma * x) @ phi^T    (phi is outDim x nC)
 *   3. Pre/post: sigmoid(r_inv * H * alpha + bias)
 *   4. Res: softmax over n! permutation logits → Birkhoff-von Neumann
 *   5. Weighted sum: h_in = sum_n(x_streams[n] * h_pre[n])
 *
 * This is the "model.py on NPU" — uses at:: ops which leverage Cube where available.
 */

#include <cmath>
#include <vector>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include "ops.h"

namespace cmhc_binding {

// Pre-computed permutation index table for N=4 (n! = 24)
static const int32_t kPermIdx[96] = {
    0,1,2,3, 0,1,3,2, 0,2,1,3, 0,2,3,1, 0,3,1,2, 0,3,2,1,
    1,0,2,3, 1,0,3,2, 1,2,0,3, 1,2,3,0, 1,3,0,2, 1,3,2,0,
    2,0,1,3, 2,0,3,1, 2,1,0,3, 2,1,3,0, 2,3,0,1, 2,3,1,0,
    3,0,1,2, 3,0,2,1, 3,1,0,2, 3,1,2,0, 3,2,0,1, 3,2,1,0,
};

std::vector<at::Tensor> cmhc(
    const at::Tensor &x,
    const at::Tensor &phi,
    const at::Tensor &alpha,
    const at::Tensor &bias,
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

    // Build permutation matrix tensor
    auto perm_opts = at::TensorOptions().dtype(at::kLong).device(x.device());
    at::Tensor perm_idx = at::from_blob(const_cast<int32_t*>(kPermIdx),
                                         {nPerm, N}, at::TensorOptions().dtype(at::kInt)
                                         .device(at::kCPU)).to(at::kLong).to(x.device());

    // One-hot encoding: (nPerm, N, N)
    at::Tensor perms = at::zeros({nPerm, N, N}, opts_f32);
    for (int64_t p = 0; p < nPerm; ++p) {
        for (int64_t i = 0; i < N; ++i) {
            int64_t col = kPermIdx[p * N + i];
            perms[p][i][col] = 1.0f;
        }
    }

    // einsum('mr,rij->mij')  →  (M, N, N)
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

}  // namespace cmhc_binding
