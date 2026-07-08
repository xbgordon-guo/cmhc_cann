/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * cmhc torch.ops binding — header
 */
#ifndef CMHC_TORCH_OPS_H
#define CMHC_TORCH_OPS_H

#include <ATen/ATen.h>
#include <vector>

namespace cmhc_binding {

std::vector<at::Tensor> cmhc(
    const at::Tensor &x,
    const at::Tensor &phi,
    const at::Tensor &alpha,
    const at::Tensor &bias,
    int64_t hc_mult,
    int64_t num_iters,
    double hc_eps,
    double norm_eps,
    bool need_backward);

}  // namespace cmhc_binding

#endif  // CMHC_TORCH_OPS_H
