#!/usr/bin/env python3
"""Precision verification: mhc_pre_cmhc_backward C++ binding vs PyTorch reference.

Uses torch.utils.cpp_extension.load_inline for CPU testing.
For NPU, the binding .so (PrivateUse1) is used via torch.ops.npu.mhc_pre_cmhc_backward.

Usage:
    python compare_backward.py          # CPU test (load_inline)
    python compare_backward.py --npu    # NPU test (torch.ops.npu.*)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch

_PROJECT_DIR = Path(__file__).resolve().parent


def load_binding_cpu():
    """Compile and load C++ binding via load_inline (works on CPU)."""
    cpp_path = _PROJECT_DIR / "binding" / "mhc_pre_cmhc_backward_torch_host.cpp"
    with open(cpp_path) as f:
        cpp_source = f.read()

    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name="mhc_pre_cmhc_backward_cpu_test",
        cpp_sources=[cpp_source],
        functions=["mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward"],
        extra_cflags=["-std=c++17", "-O2"],
        extra_include_paths=[str(_PROJECT_DIR / "binding")],
        verbose=False,
    )


def load_binding_npu():
    """Load the pre-built .so for NPU (PrivateUse1)."""
    libs = list((_PROJECT_DIR / "binding").glob("mhc_pre_cmhc_backward_torch_ops.*.so"))
    if libs:
        torch.ops.load_library(str(libs[0]))
    else:
        print("ERROR: mhc_pre_cmhc_backward_torch_ops.*.so not found. Build first:")
        print("  cd binding && python setup.py build_ext --inplace")
        sys.exit(1)


# Load Python reference
sys.path.insert(0, str(_PROJECT_DIR))
from backward_reference import mhc_pre_cmhc_forward, mhc_pre_cmhc_backward_manual, get_permutation_matrices


def run_comparison(B=2, S=4, N=4, C=64, seed=42, use_npu=False):
    torch.manual_seed(seed)

    nC = N * C
    nPerm = math.factorial(N)
    outDim = nPerm + 2 * N
    device = 'cpu'

    # Compile/load binding
    if use_npu:
        load_binding_npu()
        backend_fn = torch.ops.npu.mhc_pre_cmhc_backward
    else:
        module = load_binding_cpu()
        backend_fn = getattr(module, "mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward")

    # 1. Create CPU tensors, compute reference
    x_cpu = torch.randn(B, S, N, C)
    phi_cpu = torch.randn(outDim, nC)
    alpha_cpu = torch.ones(3) * 0.01
    bias_cpu = torch.zeros(outDim) * 0.01
    perms = get_permutation_matrices(N, 'cpu', torch.float32)

    (h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms,
     res_coeff, h_pre_raw, h_post_raw, h_res_logits,
     x_flat, x_streams) = mhc_pre_cmhc_forward(x_cpu, phi_cpu, alpha_cpu, bias_cpu, perms)

    grad_h_in = torch.ones_like(h_in)
    grad_h_post = torch.ones_like(h_post)
    grad_h_res = torch.ones_like(h_res)

    # Python reference (always on CPU for ground truth)
    ref_grad_x, ref_grad_phi, ref_grad_alpha, ref_grad_bias = mhc_pre_cmhc_backward_manual(
        x_cpu, phi_cpu, alpha_cpu, bias_cpu, perms,
        grad_h_in, grad_h_post, grad_h_res,
        h_pre, hc_before_norm, inv_rms,
        res_coeff, h_pre_raw, h_post_raw, h_res_logits,
        x_flat, x_streams,
    )

    # 2. If NPU, run C++ binding on NPU with same data
    if use_npu:
        # Move to NPU (use npu:0 — ASCEND_RT_VISIBLE_DEVICES controls physical mapping)
        x_npu = x_cpu.npu()
        phi_npu = phi_cpu.npu()
        alpha_npu = alpha_cpu.npu()
        bias_npu = bias_cpu.npu()
        ghi_npu = grad_h_in.npu()
        ghp_npu = grad_h_post.npu()
        ghr_npu = grad_h_res.npu()
        hp_npu = h_pre.npu()
        hbn_npu = hc_before_norm.npu()
        ir_npu = inv_rms.npu()
        perms_npu = perms.npu()

        result = backend_fn(ghi_npu, ghp_npu, ghr_npu,
                            x_npu, phi_npu, alpha_npu, bias_npu,
                            hp_npu, hbn_npu, ir_npu,
                            perms_npu, 1e-6)
        binding_out = [t.cpu() for t in result]
        backend_label = "NPU"
    else:
        result = backend_fn(grad_h_in, grad_h_post, grad_h_res,
                            x_cpu, phi_cpu, alpha_cpu, bias_cpu,
                            h_pre, hc_before_norm, inv_rms,
                            perms, 1e-6)
        binding_out = list(result)
        backend_label = "CPU"

    # Comparison
    print("=" * 70)
    print(f"mhc_pre_cmhc_backward — C++ Binding ({backend_label}) vs Python Reference")
    print("=" * 70)
    print(f"  B={B}, S={S}, N={N}, C={C}, outDim={outDim}, nPerm={nPerm}")
    print()

    def compare(name, ref, binding, atol=1e-4):
        diff = (ref - binding).abs()
        ok = torch.allclose(ref, binding, atol=atol)
        status = "PASS" if ok else "FAIL"
        print(f"  {name:16s}  max_diff={diff.max().item():.2e}  "
              f"mean_diff={diff.mean().item():.2e}  [{status}]")
        if not ok:
            worst_idx = diff.argmax()
            print(f"     Worst: ref={ref.flatten()[worst_idx].item():.6f}  "
                  f"binding={binding.flatten()[worst_idx].item():.6f}")
        return ok

    all_ok = True
    all_ok &= compare("grad_x",     ref_grad_x,     binding_out[0])
    all_ok &= compare("grad_phi",   ref_grad_phi,   binding_out[1])
    all_ok &= compare("grad_alpha", ref_grad_alpha, binding_out[2])
    all_ok &= compare("grad_bias",  ref_grad_bias,  binding_out[3])

    print(f"\n{'='*70}")
    print(f"  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print(f"{'='*70}")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", action="store_true", help="Use NPU backend (PrivateUse1)")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S", type=int, default=4)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--C", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ok = run_comparison(B=args.B, S=args.S, N=args.N, C=args.C, seed=args.seed, use_npu=args.npu)
    sys.exit(0 if ok else 1)
