#!/usr/bin/env python3
"""Precision comparison: model.py (reference) vs model_new_ascendc.py (kernel).

Usage:
    # CPU-only test (reference model only)
    python compare_precision.py --cpu

    # NPU comparison (requires cmhc binding installed)
    python compare_precision.py --npu
"""

import argparse
import math
import torch

import model
import model_new_ascendc


def compare(ref_outputs, cand_outputs, names, atol=1e-4):
    """Compare reference vs candidate outputs."""
    for i, (r, c, name) in enumerate(zip(ref_outputs, cand_outputs, names)):
        r_f32 = r.float() if r.dtype != torch.float32 else r
        c_f32 = c.float() if c.dtype != torch.float32 else c
        diff = (r_f32 - c_f32).abs()
        ok = torch.allclose(r_f32, c_f32, atol=atol)
        match = (diff < atol).float().mean().item() * 100
        print(f"  {name}: ok={ok}  max_diff={diff.max().item():.6f}  "
              f"match={match:.1f}%  ref_sum={r_f32.sum().item():.4f}  "
              f"cand_sum={c_f32.sum().item():.4f}")
        if not ok:
            print(f"    !!! MISMATCH: max element diff = {diff.max().item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="CPU-only reference test")
    parser.add_argument("--npu", action="store_true", help="NPU kernel comparison")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=2)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--C", type=int, default=3584)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.cpu and not args.npu:
        args.cpu = True  # default

    torch.manual_seed(args.seed)

    # --- Create models with identical weights ---
    ref = model.Model(N=args.N, C=args.C)
    cand = model_new_ascendc.ModelNew(N=args.N, C=args.C)

    # Copy phi weights: ref.phi → cand.phi
    cand.phi.data.copy_(ref.phi.data)
    cand.alpha.data.copy_(ref.alpha.data)
    cand.bias.data.copy_(ref.bias.data)

    # --- Run CPU reference ---
    x_cpu = torch.randn(args.B, args.S, args.N, args.C, dtype=torch.float32)
    print(f"\n=== Reference model.py (CPU) ===")
    print(f"x: {list(x_cpu.shape)}, phi: {list(ref.phi.shape)}")
    with torch.no_grad():
        r = ref(x_cpu)
    print(f"h_in:   sum={r[0].sum().item():.4f}  shape={list(r[0].shape)}")
    print(f"h_post: sum={r[1].sum().item():.4f}  min={r[1].min().item():.4f} max={r[1].max().item():.4f}")
    print(f"h_res:  sum={r[2].sum().item():.4f}  row_sum=[{r[2].sum(-1).min().item():.4f},{r[2].sum(-1).max().item():.4f}]")

    if args.npu:
        # --- Run NPU kernel ---
        x_npu = x_cpu.npu()
        print(f"\n=== Kernel model_new_ascendc.py (NPU) ===")
        try:
            with torch.no_grad():
                c = cand(x_npu)
            c_cpu = [t.cpu() for t in c]

            print(f"\n=== Comparison (ref CPU vs kernel NPU) ===")
            compare(r[:3], c_cpu[:3], ["h_in", "h_post", "h_res"])
        except Exception as e:
            print(f"\nKernel call FAILED: {e}")
            print("The cmhc binding .so needs to be built first:")
            print("  cd binding && python setup.py build_ext --inplace")
    else:
        print("\n(CPU-only mode. Use --npu for kernel comparison.)")

    # --- Cross-validate with lower precision ---
    print(f"\n=== BF16 cross-check (CPU only) ===")
    x_bf16 = x_cpu.to(torch.bfloat16)
    with torch.no_grad():
        r_bf16 = ref(x_bf16)
    for name, r_f32, r_b16 in zip(["h_in", "h_post", "h_res"], r, r_bf16):
        diff = (r_f32 - r_b16.float()).abs()
        print(f"  {name}: fp32↔bf16 max_diff={diff.max().item():.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
