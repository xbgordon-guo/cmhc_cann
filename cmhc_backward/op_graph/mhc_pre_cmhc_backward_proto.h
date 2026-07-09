/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_cmhc_backward_proto.h
 * \brief MhcPreCmhcBackward operator proto definition
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_CMHC_BACKWARD_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_CMHC_BACKWARD_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief MhcPreCmhcBackward — gradient of cmhc (Softmax-Permutation H_res).
 *
 * Inputs (11):
 *   grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
 *   h_pre, hc_before_norm, inv_rms, perm_mats
 *
 * Outputs (4):
 *   grad_x, grad_phi, grad_alpha, grad_bias
 *
 * Permutation matrices (perm_mats) are pre-computed on the Python side
 * and passed as input, matching the reference cmhc_cann/cmhc pattern.
 *
 * Differs from MhcPreSinkhornBackward:
 *   - Removes sum_out / norm_out (Sinkhorn intermediates, not needed for softmax-permutation)
 *   - Replaces Sinkhorn backward iteration with direct softmax-permutation gradient
 */
REG_OP(MhcPreCmhcBackward)
    .INPUT(grad_hin, "T1")
    .INPUT(grad_h_post, "T2")
    .INPUT(grad_h_res, "T2")
    .INPUT(x, "T1")
    .INPUT(phi, "T2")
    .INPUT(alpha, "T2")
    .INPUT(bias, "T2")
    .INPUT(h_pre, "T2")
    .INPUT(hc_before_norm, "T2")
    .INPUT(inv_rms, "T2")
    .INPUT(perm_mats, "T2")
    .OUTPUT(grad_x, "T1")
    .OUTPUT(grad_phi, "T2")
    .OUTPUT(grad_alpha, "T2")
    .OUTPUT(grad_bias, "T2")
    .DATATYPE(T1, TensorType({DT_BF16, DT_FLOAT16}))
    .DATATYPE(T2, TensorType({DT_FLOAT}))
    .ATTR(hc_eps, Float, 1e-6f)
    .OP_END_FACTORY_REG(MhcPreCmhcBackward)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_CMHC_BACKWARD_PROTO_H_
