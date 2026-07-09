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
 * \file mhc_pre_cmhc_backward_infershape.cpp
 * \brief MhcPreCmhcBackward infershape implementation
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t INDEX_GRAD_HIN = 0;
constexpr size_t INDEX_GRAD_HPOST = 1;
constexpr size_t INDEX_GRAD_HRES = 2;
constexpr size_t INDEX_X = 3;
constexpr size_t INDEX_PHI = 4;
constexpr size_t INDEX_ALPHA = 5;
constexpr size_t INDEX_BIAS = 6;
constexpr size_t INDEX_H_PRE = 7;
constexpr size_t INDEX_HC_BEFORE_NORM = 8;
constexpr size_t INDEX_INV_RMS = 9;
constexpr size_t INDEX_PERM_MATS = 10;

constexpr size_t OUTPUT_GRAD_X = 0;
constexpr size_t OUTPUT_GRAD_PHI = 1;
constexpr size_t OUTPUT_GRAD_ALPHA = 2;
constexpr size_t OUTPUT_GRAD_BIAS = 3;
} // namespace

static ge::graphStatus InferShapeForMhcPreCmhcBackward(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MhcPreCmhcBackward infershape.");

    const gert::Shape *xShape = context->GetInputShape(INDEX_X);
    const gert::Shape *phiShape = context->GetInputShape(INDEX_PHI);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, phiShape);

    // x: (B, S, N, C)
    int64_t batch = xShape->GetDim(0);
    int64_t seqLen = xShape->GetDim(1);
    int64_t n = xShape->GetDim(2);
    int64_t c = xShape->GetDim(3);

    // grad_x: same shape as x
    gert::Shape *gradXShape = context->GetOutputShape(OUTPUT_GRAD_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradXShape);
    gradXShape->SetDimNum(4);
    gradXShape->SetDim(0, batch);
    gradXShape->SetDim(1, seqLen);
    gradXShape->SetDim(2, n);
    gradXShape->SetDim(3, c);

    // grad_phi: same shape as phi (outDim, n*c)
    gert::Shape *gradPhiShape = context->GetOutputShape(OUTPUT_GRAD_PHI);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradPhiShape);
    gradPhiShape->SetDimNum(2);
    gradPhiShape->SetDim(0, phiShape->GetDim(0));
    gradPhiShape->SetDim(1, phiShape->GetDim(1));

    // grad_alpha: (3,)
    gert::Shape *gradAlphaShape = context->GetOutputShape(OUTPUT_GRAD_ALPHA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradAlphaShape);
    gradAlphaShape->SetDimNum(1);
    gradAlphaShape->SetDim(0, 3);

    // grad_bias: (outDim,)
    gert::Shape *gradBiasShape = context->GetOutputShape(OUTPUT_GRAD_BIAS);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradBiasShape);
    gradBiasShape->SetDimNum(1);
    gradBiasShape->SetDim(0, phiShape->GetDim(0));

    OP_LOGD(context->GetNodeName(), "End to do MhcPreCmhcBackward infershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMhcPreCmhcBackward(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MhcPreCmhcBackward infer datatype.");
    const ge::DataType xDtype = context->GetInputDataType(INDEX_X);
    context->SetOutputDataType(OUTPUT_GRAD_X, xDtype);
    context->SetOutputDataType(OUTPUT_GRAD_PHI, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_GRAD_ALPHA, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_GRAD_BIAS, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do MhcPreCmhcBackward infer datatype.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPreCmhcBackward)
    .InferShape(InferShapeForMhcPreCmhcBackward)
    .InferDataType(InferDataTypeForMhcPreCmhcBackward);

} // namespace ops
