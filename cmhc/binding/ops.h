/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * mhc_pre_cmhc torch.ops binding — header
 */
#ifndef MHC_PRE_CMHC_TORCH_OPS_H
#define MHC_PRE_CMHC_TORCH_OPS_H

#include <ATen/ATen.h>
#include <vector>

namespace mhc_pre_cmhc_binding {

std::vector<at::Tensor> mhc_pre_cmhc(
    const at::Tensor &x,
    const at::Tensor &phi,
    const at::Tensor &alpha,
    const at::Tensor &bias,
    const at::Tensor &perm_mats,
    int64_t hc_mult,
    int64_t num_iters,
    double hc_eps,
    double norm_eps,
    bool need_backward);

}  // namespace mhc_pre_cmhc_binding

#endif  // MHC_PRE_CMHC_TORCH_OPS_H
