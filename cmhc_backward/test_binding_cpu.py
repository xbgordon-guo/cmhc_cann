#!/usr/bin/env python3
"""Test mhc_pre_cmhc_backward C++ binding on CPU via torch.utils.cpp_extension.load_inline.

This compiles the backward kernel at runtime and compares against the
Python reference. No CANN/NPU required.
"""

import math
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

# Read the C++ source
CPP_PATH = Path(__file__).resolve().parent / "binding" / "mhc_pre_cmhc_backward_torch_host.cpp"
with open(CPP_PATH) as f:
    cpp_source = f.read()

# Load as a module
print("Compiling mhc_pre_cmhc_backward C++ module (CPU backend)...")
module = load_inline(
    name="mhc_pre_cmhc_backward_cpu_test",
    cpp_sources=[cpp_source],
    functions=["mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward"],
    extra_cflags=["-std=c++17", "-O2"],
    extra_include_paths=[str(Path(__file__).resolve().parent / "binding")],
    verbose=False,
)
print("Done.\n")

# Load Python reference
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backward_reference import mhc_pre_cmhc_forward, mhc_pre_cmhc_backward_manual, get_permutation_matrices


def run_comparison(B=2, S=4, N=4, C=64, seed=42):
    torch.manual_seed(seed)

    nC = N * C
    nPerm = math.factorial(N)
    outDim = nPerm + 2 * N
    device = 'cpu'

    x_ref = torch.randn(B, S, N, C, device=device)
    phi_ref = torch.randn(outDim, nC, device=device)
    alpha_ref = torch.ones(3, device=device) * 0.01
    bias_ref = torch.zeros(outDim, device=device) * 0.01
    perms = get_permutation_matrices(N, device, torch.float32)

    # Forward
    (h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms,
     res_coeff, h_pre_raw, h_post_raw, h_res_logits,
     x_flat, x_streams) = mhc_pre_cmhc_forward(x_ref, phi_ref, alpha_ref, bias_ref, perms)

    grad_h_in = torch.ones_like(h_in)
    grad_h_post = torch.ones_like(h_post)
    grad_h_res = torch.ones_like(h_res)

    # Python reference
    ref_grad_x, ref_grad_phi, ref_grad_alpha, ref_grad_bias = mhc_pre_cmhc_backward_manual(
        x_ref, phi_ref, alpha_ref, bias_ref, perms,
        grad_h_in, grad_h_post, grad_h_res,
        h_pre, hc_before_norm, inv_rms,
        res_coeff, h_pre_raw, h_post_raw, h_res_logits,
        x_flat, x_streams,
    )

    # C++ binding
    fn = getattr(module, "mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward")
    result = fn(
        grad_h_in, grad_h_post, grad_h_res,
        x_ref, phi_ref, alpha_ref, bias_ref,
        h_pre, hc_before_norm, inv_rms,
        perms,
        1e-6,
    )
    cpp_grad_x, cpp_grad_phi, cpp_grad_alpha, cpp_grad_bias = result

    # Comparison
    print("=" * 70)
    print("mhc_pre_cmhc_backward — C++ Binding vs Python Reference (CPU)")
    print("=" * 70)
    print(f"  B={B}, S={S}, N={N}, C={C}, outDim={outDim}, nPerm={nPerm}")
    print()

    def compare(name, ref, cpp, atol=1e-4):
        diff = (ref - cpp).abs()
        ok = torch.allclose(ref, cpp, atol=atol)
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name:16s}  max_diff={diff.max().item():.2e}  "
              f"mean_diff={diff.mean().item():.2e}  "
              f"ref_sum={ref.sum().item():.6f}  cpp_sum={cpp.sum().item():.6f}  [{status}]")
        if not ok:
            worst_idx = diff.argmax()
            print(f"     Worst idx={worst_idx}: ref={ref.flatten()[worst_idx].item():.6f}  "
                  f"cpp={cpp.flatten()[worst_idx].item():.6f}")
        return ok

    all_ok = True
    all_ok &= compare("grad_x",     ref_grad_x,     cpp_grad_x)
    all_ok &= compare("grad_phi",   ref_grad_phi,   cpp_grad_phi)
    all_ok &= compare("grad_alpha", ref_grad_alpha, cpp_grad_alpha)
    all_ok &= compare("grad_bias",  ref_grad_bias,  cpp_grad_bias)

    print(f"\n{'='*70}")
    print(f"  Overall: {'✓ ALL PASS' if all_ok else '✗ SOME FAILED'}")
    print(f"{'='*70}")
    return all_ok


if __name__ == "__main__":
    ok = run_comparison()
    sys.exit(0 if ok else 1)
