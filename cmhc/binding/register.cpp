/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * cmhc — torch.ops.npu.mhc_pre_cmhc registration
 */
#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("mhc_pre_cmhc(Tensor x, Tensor phi, Tensor alpha, Tensor bias, Tensor perm_mats, "
          "int hc_mult, int num_iters, float hc_eps, float norm_eps, bool need_backward) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("mhc_pre_cmhc", TORCH_FN(mhc_pre_cmhc_binding::mhc_pre_cmhc));
}

}  // namespace
