"""Standard MHC — RMS Norm + Pre/Post/Res projection + Softmax-Permutation H_res.

Based on mHC: Manifold-Constrained Hyper-Connections (arXiv:2512.24880).
Reference: /home/y00889327/mHC-pytorch/mHC.ipynb

Key modification from standard mHC:
  Sinkhorn-Knopp on H_res → softmax + permutation matrix weighted sum (Birkhoff-von Neumann).

Formula:
  1. x_flat = x.reshape(B, S, N*C).float()
  2. Fused RMS: h_scaled = gamma * x_flat            (element-wise)
  3. Project:   H = h_scaled @ w                       w: (N*C, n! + 2N)
  4. r = ||x_flat||_2 / sqrt(N*C),  r_inv = 1 / r
  5. H_pre  = r_inv * H[:, :, :N]           * alpha[0] + bias[:N]          → sigmoid
  6. H_post = r_inv * H[:, :, N:2N]         * alpha[1] + bias[N:2N]        → 2*sigmoid
  7. H_res  = r_inv * H[:, :, 2N:2N+n!]    * alpha[2] + bias[2N:2N+n!]    → softmax
     res_coeff = softmax(H_res, dim=-1)           # (B, S, n!)
     perms = _get_permutations(dtype)              # (n!, N, N), gamma-blended
     H_res = einsum('bsr,rij->bsij', res_coeff, perms)  # (B, S, N, N)
  8. h_in = sum_n(x_streams[n] * H_pre[n])       # (B, S, C)

Gamma blending (Birkhoff-von Neumann):
    perm_mats = cmhc_gamma * P_p + (1 - cmhc_gamma) / N * J
  where P_p = pure permutation matrix, J = all-ones matrix.
  cmhc_gamma = 1.0 → pure permutations; < 1.0 → blended toward uniform.
"""

import itertools
import json
import math
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    """Standard MHC with softmax-permutation H_res (replaces Sinkhorn-Knopp).

    Permutation matrices are pre-computed once in ``__init__`` as a pure base
    (``_perm_base``), then gamma-blended with the uniform distribution on first
    access per dtype via ``_get_permutations()``.  This mirrors the MindSpore
    ``HyperConnectionModule`` pattern and avoids regenerating the same matrices
    on every forward pass.
    """

    def __init__(self, N: int = 4, C: int = 3584, cmhc_gamma: float = 1.0):
        super().__init__()
        self.N = N
        self.C = C
        self._cmhc_gamma = cmhc_gamma
        nC = N * C                # 512
        n_perm = math.factorial(N)  # 24
        out_dim = n_perm + 2 * N    # 24 + 8 = 32

        # phi: (outDim, nC) = (32, 14336) — transposed for Cube MatMul efficiency
        self.phi = nn.Parameter(torch.empty(out_dim, nC))
        # alpha: (3,) — pre, post, res scaling
        self.alpha = nn.Parameter(torch.ones(3) * 0.01)
        # bias: (n! + 2N,)
        self.bias = nn.Parameter(torch.zeros(out_dim) * 0.01)
        # gamma: RMSNorm weight (nC,)
        self.gamma = nn.Parameter(torch.ones(nC))

        # Pre-compute pure permutation base [n!, n, n] — heavy numpy once.
        perms = list(itertools.permutations(range(N)))
        perm_base = torch.zeros(n_perm, N, N)
        for p_idx, p in enumerate(perms):
            for i, j in enumerate(p):
                perm_base[p_idx, i, j] = 1.0
        self.register_buffer('_perm_base', perm_base, persistent=False)
        self._perms_cache = {}

        self._init_weights()

    def _init_weights(self):
        state = torch.get_rng_state()
        torch.manual_seed(12345)
        nn.init.xavier_uniform_(self.phi)
        torch.set_rng_state(state)

    def _get_permutations(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return cached permutation matrices cast to *dtype*, gamma-blended.

        On first access per dtype, the pure permutation base is blended with
        the uniform distribution when ``_cmhc_gamma < 1.0`` and cached.

        Args:
            dtype: Target torch dtype for the returned Tensor.

        Returns:
            Tensor of shape ``(n!, N, N)``, either pure permutations or
            gamma-blended with the uniform distribution.
        """
        if dtype not in self._perms_cache:
            perm_mats = self._perm_base.to(dtype=dtype)
            if self._cmhc_gamma < 1.0:
                perm_mats = (
                    self._cmhc_gamma * perm_mats +
                    (1.0 - self._cmhc_gamma) / self.N
                )
            self._perms_cache[dtype] = perm_mats
        return self._perms_cache[dtype]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, S, N, C) input.

        Returns:
            h_in:   (B, S, C)       aggregated layer input (x dtype).
            h_post: (B, S, N)       post weights (float32).
            h_res:  (B, S, N, N)    residual connection matrix (float32).
        """
        B, S, N_val, C_val = x.shape
        N = self.N
        C = self.C
        nC = N * C
        n_perm = math.factorial(N)
        input_dtype = x.dtype
        eps = 1e-6

        # 1. Flatten
        x_flat = x.reshape(B, S, nC).float()  # (B, S, nC)

        # 2. Fused RMSNorm: pre-scale with gamma, project, then apply RMS norm
        #    Mathematically equivalent to RMSNorm(x) @ w, but avoids
        #    creating intermediate x_normed array.
        h_scaled = self.gamma * x_flat  # (B, S, nC)

        # 3. Projection: H = (gamma * x) @ phi^T   where phi is (outDim, nC)
        H = torch.matmul(h_scaled, self.phi.t())  # (B, S, n!+2N) = (B, S, 32)

        # 4. RMS norm factor from original x (not gamma-scaled)
        r = x_flat.norm(dim=-1, keepdim=True) / math.sqrt(nC)  # (B, S, 1)
        r_inv = 1.0 / (r + eps)

        # 5. Split and apply RMS correction + alpha + bias
        H_pre_raw = r_inv * H[:, :, :N] * self.alpha[0] + self.bias[:N]
        H_post_raw = r_inv * H[:, :, N:2*N] * self.alpha[1] + self.bias[N:2*N]
        H_res_raw = r_inv * H[:, :, 2*N:2*N+n_perm] * self.alpha[2] + self.bias[2*N:2*N+n_perm]

        # 6. Activations
        H_pre = torch.sigmoid(H_pre_raw)              # (B, S, N)
        H_post = 2.0 * torch.sigmoid(H_post_raw)       # (B, S, N)

        # 7. Softmax + permutation einsum (replaces Sinkhorn-Knopp)
        #    Birkhoff-von Neumann: doubly stochastic matrix as convex combination
        #    of permutation matrices weighted by softmax coefficients.
        #    Uses _get_permutations() to support gamma blending (cmhc_gamma < 1.0).
        res_coeff = torch.softmax(H_res_raw, dim=-1)                     # (B, S, n!)
        perms = self._get_permutations(dtype=torch.float32).to(x.device)  # (n!, N, N), gamma-blended
        H_res = torch.einsum('bsr,rij->bsij', res_coeff, perms)          # (B, S, N, N)

        # 8. Aggregation: h_in = sum_n(x_streams[n] * H_pre[n])
        x_streams = x_flat.reshape(B, S, N, C)         # (B, S, N, C)
        h_in = (x_streams * H_pre.unsqueeze(-1)).sum(dim=2)  # (B, S, C)

        return h_in.to(input_dtype), H_post, H_res


# =========================================================================
# Benchmark entry-points
# =========================================================================

def get_input_groups():
    """Read JSON test cases and generate input tensors."""
    json_path = os.path.join(os.path.dirname(__file__), "mhc_pre_sinkhorn.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    input_groups = []
    for case in cases:
        x_info = case["inputs"][0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        input_groups.append([x])

    return input_groups


def get_init_inputs():
    """Constructor args for Model()."""
    return []


# =========================================================================
# Smoke test
# =========================================================================

if __name__ == "__main__":
    print("=== MHC Standard — smoke test ===\n")

    torch.manual_seed(42)

    for gamma in [1.0, 0.9, 0.5]:
        model = Model(N=4, C=3584, cmhc_gamma=gamma)
        model.eval()
        print(f"--- cmhc_gamma = {gamma} ---")

        for idx, (x,) in enumerate(get_input_groups()):
            with torch.no_grad():
                h_in, h_post, h_res = model(x)

            print(f"Case {idx}: x={list(x.shape)}  dtype={x.dtype}")
            print(f"  h_in   shape={list(h_in.shape)}   dtype={h_in.dtype}   "
                  f"min={h_in.min().item():.4f}  max={h_in.max().item():.4f}")
            print(f"  h_post shape={list(h_post.shape)} dtype={h_post.dtype} "
                  f"min={h_post.min().item():.4f}  max={h_post.max().item():.4f}")
            print(f"  h_res  shape={list(h_res.shape)}  dtype={h_res.dtype}  "
                  f"min={h_res.min().item():.4f}  max={h_res.max().item():.4f}")
            # Verify doubly stochastic: each row/col should sum to 1
            row_sums = h_res.sum(dim=-1)
            col_sums = h_res.sum(dim=-2)
            print(f"  h_res  row_sums: [{row_sums.min().item():.4f}, {row_sums.max().item():.4f}]"
                  f"  col_sums: [{col_sums.min().item():.4f}, {col_sums.max().item():.4f}]")
            print()
        print()
