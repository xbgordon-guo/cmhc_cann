/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * mhc_pre_cmhc_backward torch.ops binding — header
 *
 * Permutation matrices are passed as input (pre-computed on Python side),
 * matching the reference cmhc_cann/cmhc pattern.
 */
#ifndef MHC_PRE_CMHC_BACKWARD_TORCH_OPS_H
#define MHC_PRE_CMHC_BACKWARD_TORCH_OPS_H

#include <ATen/ATen.h>
#include <vector>

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
    double hc_eps);

}  // namespace mhc_pre_cmhc_backward_binding

#endif  // MHC_PRE_CMHC_BACKWARD_TORCH_OPS_H
