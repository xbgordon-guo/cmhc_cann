/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * mhc_pre_cmhc_backward — torch.ops.npu.mhc_pre_cmhc_backward registration
 *
 * Permutation matrices are passed as input (pre-computed on Python side).
 */
#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("mhc_pre_cmhc_backward(Tensor grad_h_in, Tensor grad_h_post, Tensor grad_h_res, "
          "Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
          "Tensor h_pre, Tensor hc_before_norm, Tensor inv_rms, "
          "Tensor perm_mats, "
          "float hc_eps) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("mhc_pre_cmhc_backward", TORCH_FN(mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward));
}

}  // namespace
