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
 * \file mhc_pre_cmhc_backward_arch22_tiling.cpp
 * \brief MhcPreCmhcBackward operator tiling implementation (arch22 / 910B)
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "mhc_pre_cmhc_backward_arch22_tiling.h"
#include "log/log.h"
#include "../../../op_kernel/arch22/mhc_pre_cmhc_backward_data_arch22.h"
#include "../../../op_kernel/arch22/mhc_pre_cmhc_backward_key_arch22.h"

#define CHECK_NULLPTR(ptr) \
    if (ptr == nullptr) {  \
        return ge::GRAPH_FAILED; \
    }

namespace {
constexpr uint8_t GRAD_HIN_IDX = 0;
constexpr uint8_t GRAD_H_POST_IDX = 1;
constexpr uint8_t GRAD_H_RES_IDX = 2;
constexpr uint8_t INPUT_X_IDX = 3;
constexpr uint8_t PHI_IDX = 4;
constexpr uint8_t ALPHA_IDX = 5;
constexpr uint8_t BIAS_IDX = 6;
constexpr uint8_t H_PRE_IDX = 7;
constexpr uint8_t HC_BEFORE_NORM_IDX = 8;
constexpr uint8_t INV_RMS_IDX = 9;
constexpr uint8_t PERM_MATS_IDX = 10;
constexpr uint8_t GRAD_X_IDX = 0;
constexpr uint8_t GRAD_PHI_IDX = 1;
constexpr uint8_t GRAD_ALPHA_IDX = 2;
constexpr uint8_t GRAD_BIAS_IDX = 3;
constexpr float DEFAULT_EPS = 1e-6f;
constexpr uint8_t BATCH_SIZE_DIM_IDX = 0;
constexpr uint8_t SEQ_LENGTH_DIM_IDX = 1;
constexpr uint8_t N_DIM_IDX = 2;
constexpr uint8_t C_DIM_IDX = 3;
constexpr int64_t N_SIZE_4 = 4;
constexpr int64_t ALPHA_SIZE_3 = 3;
constexpr int64_t C_V_RATIO = 2;
constexpr int64_t DEFAULT_KEY = 0;
constexpr int64_t ELEMENTS_SIZE_PER_BLOCK = 8;
} // namespace

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {

static ge::graphStatus ShapeVerify(gert::TilingContext *context, int64_t batchSize, int64_t seqLength,
                                   int64_t n, int64_t c)
{
    auto opName = context->GetNodeName();

    // Verify grad_hin: (B, S, C)
    auto gradHinShapePtr = context->GetInputShape(GRAD_HIN_IDX);
    OP_CHECK_IF(gradHinShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "grad_hin shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradHinShape = gradHinShapePtr->GetStorageShape();
    OP_CHECK_IF(gradHinShape.GetDimNum() != 3 ||
                    gradHinShape.GetDim(0) != batchSize || gradHinShape.GetDim(1) != seqLength ||
                    gradHinShape.GetDim(2) != c,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "grad_hin shape mismatch"), return ge::GRAPH_FAILED);

    // Verify grad_h_post: (B, S, N)
    auto gradHPostShapePtr = context->GetInputShape(GRAD_H_POST_IDX);
    OP_CHECK_IF(gradHPostShapePtr == nullptr,
                OP_LOGE(opName, "gradHPostShapePtr is null"), return ge::GRAPH_FAILED);
    auto gradHPostShape = gradHPostShapePtr->GetStorageShape();
    OP_CHECK_IF(gradHPostShape.GetDimNum() != 3 ||
                    gradHPostShape.GetDim(0) != batchSize || gradHPostShape.GetDim(1) != seqLength ||
                    gradHPostShape.GetDim(2) != n,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "grad_h_post shape mismatch"), return ge::GRAPH_FAILED);

    // Verify grad_h_res: (B, S, N, N)
    auto gradHResShapePtr = context->GetInputShape(GRAD_H_RES_IDX);
    OP_CHECK_IF(gradHResShapePtr == nullptr,
                OP_LOGE(opName, "gradHResShapePtr is null"), return ge::GRAPH_FAILED);
    auto gradHResShape = gradHResShapePtr->GetStorageShape();
    OP_CHECK_IF(gradHResShape.GetDimNum() != 4 ||
                    gradHResShape.GetDim(0) != batchSize || gradHResShape.GetDim(1) != seqLength ||
                    gradHResShape.GetDim(2) != n || gradHResShape.GetDim(3) != n,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "grad_h_res shape mismatch"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMhcPreCmhcBackwardArch22(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "MhcPreCmhcBackward arch22 tiling start.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE(context->GetNodeName(), "platformInfo is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // Get input shapes
    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_IF(xShapePtr == nullptr,
                OP_LOGE(context->GetNodeName(), "xShapePtr is null"), return ge::GRAPH_FAILED);
    auto xShape = xShapePtr->GetStorageShape();
    int64_t batchSize = xShape.GetDim(BATCH_SIZE_DIM_IDX);
    int64_t seqLength = xShape.GetDim(SEQ_LENGTH_DIM_IDX);
    int64_t n = xShape.GetDim(N_DIM_IDX);
    int64_t c = xShape.GetDim(C_DIM_IDX);
    int64_t nC = n * c;
    // cmhc forward: hcMix = nPerm + 2*n where nPerm = n! = 24 for n=4
    // This differs from Sinkhorn version (hcMix = n*n + 2*n = 24)
    int64_t nPerm = n * n;  // default: n! for n=4 is 24
    // Compute factorial: 4! = 24
    if (n == 4) {
        nPerm = 24;  // 4!
    }
    int64_t hcMix = nPerm + 2 * n;  // 24 + 8 = 32

    // Validate perm_mats shape: (n!, N, N)
    auto permMatsShapePtr = context->GetInputShape(PERM_MATS_IDX);
    OP_CHECK_IF(permMatsShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "perm_mats shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto permMatsShape = permMatsShapePtr->GetStorageShape();
    OP_CHECK_IF(permMatsShape.GetDimNum() != 3 ||
                    permMatsShape.GetDim(0) != nPerm ||
                    permMatsShape.GetDim(1) != n || permMatsShape.GetDim(2) != n,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "perm_mats shape must be (n!, N, N)"),
                return ge::GRAPH_FAILED);

    int64_t totalTasks = batchSize * seqLength;

    // Get attributes
    auto attrs = context->GetAttrs();
    float eps = DEFAULT_EPS;
    if (attrs != nullptr) {
        auto epsPtr = attrs->GetAttrPointer<float>(0);
        if (epsPtr != nullptr) {
            eps = *epsPtr;
        }
    }

    OP_LOGD(context->GetNodeName(), "MhcPreCmhcBackward: bs=%ld, seq=%ld, n=%ld, c=%ld, hcMix=%ld",
            batchSize, seqLength, n, c, hcMix);

    // Shape verification
    auto ret = ShapeVerify(context, batchSize, seqLength, n, c);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Shape verification failed"), return ret);

    // Tiling computation. c0 must be VL=256 — the kernel's ComputeGradPre uses a fixed
    // 256-element reduction layout. C must be a positive multiple of 128.
    int64_t c0 = 256;
    int64_t c1 = c / c0;
    int64_t cTail = c % c0;

    OP_CHECK_IF(c <= 0 || c >= 100000 || c % 128 != 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "c must be > 0, < 100000 and divisible by 128, but got %ld", c),
                return ge::GRAPH_FAILED);

    // (B,S) tasks each AIV core processes per tile. CMHC's hcMix (=32) is larger than
    // Sinkhorn's (=24), so UB is tighter -> use a smaller tile than Sinkhorn's 32.
    // IMPORTANT: tile must stay >= INNER_SPILT_NUM(8). The kernel computes
    // onceTask_ = tile/8 and uses it as an inner-loop stride; a tile < 8 yields
    // onceTask_=0 and an infinite loop. Keep it a constant (like Sinkhorn's tile=32),
    // padded out by totalTasksAligned for small workloads.
    int64_t tile = 16;

    // MM1: grad_H2 [tile*2, hcMix] @ phi [hcMix, n*c] -> grad_x_cube [tile*2, n*c]
    int64_t mm1M = tile * 2;
    int64_t mm1K = hcMix;
    int64_t mm1N = nC;

    // MM2: (grad_H2)^T [hcMix, tile*2] @ x_workspace [tile*2, n*c] -> grad_phi [hcMix, n*c]
    int64_t mm2M = hcMix;
    int64_t mm2K = tile * 2;
    int64_t mm2N = nC;

    MhcPreCmhcBackwardArch22TilingData *tilingData =
        context->GetTilingData<MhcPreCmhcBackwardArch22TilingData>();
    auto featureDataType = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::MatmulApiTiling mm1Tiling(ascendcPlatform);
    matmul_tiling::MatmulApiTiling mm2Tiling(ascendcPlatform);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetOrgShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    if (mm1Tiling.GetTiling(tilingData->mm1TilingData) == -1) {
        OP_LOGE(context->GetNodeName(), "mm1Tiling.GetTiling failed, M=%ld, N=%ld, K=%ld", mm1M, mm1N, mm1K);
        return ge::GRAPH_FAILED;
    }

    // A is transposed (grad_H2^T): isTransA = true (last positional arg).
    mm2Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType, true);
    mm2Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm2Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm2Tiling.SetOrgShape(mm2M, mm2N, mm2K);
    mm2Tiling.SetShape(mm2M, mm2N, mm2K);
    mm2Tiling.SetBias(false);
    mm2Tiling.SetBufferSpace(-1, -1, -1);
    if (mm2Tiling.GetTiling(tilingData->mm2TilingData) == -1) {
        OP_LOGE(context->GetNodeName(), "mm2Tiling.GetTiling failed, M=%ld, N=%ld, K=%ld", mm2M, mm2N, mm2K);
        return ge::GRAPH_FAILED;
    }

    tilingData->batchSize = batchSize;
    tilingData->seqLength = seqLength;
    tilingData->c = c;
    tilingData->n = n;
    tilingData->c0 = c0;
    tilingData->c1 = c1;
    tilingData->cTail = cTail;
    tilingData->aivNum = aivNum;
    tilingData->tileGradY = tile;
    tilingData->tileHHat2 = tile;
    tilingData->tileSize = tile;
    tilingData->nPerm = nPerm;
    tilingData->ubSize = static_cast<int64_t>(ubSize);
    tilingData->eps = eps;

    // Workspace = matmul-lib system workspace + gradH2 [bs*seq*hcMix]
    //           + xCast/gradXCube [bs*seq*n*c each].
    size_t gradHat2Workspace = static_cast<size_t>(batchSize) * seqLength * hcMix * sizeof(float);
    size_t xCastWorkspace = static_cast<size_t>(batchSize) * seqLength * n * c * sizeof(float) * 2;
    size_t systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = gradHat2Workspace + xCastWorkspace;

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr,
                OP_LOGE(context->GetNodeName(), "GetWorkspaceSizes() returned nullptr"), return ge::GRAPH_FAILED);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;

    context->SetTilingKey(DEFAULT_KEY);
    context->SetBlockDim(aicNum); // MIX_AIC_1_2: one block per AIC (1 AIC + 2 AIV)
    context->SetScheduleMode(1);  // batch mode

    OP_LOGD(context->GetNodeName(),
            "MhcPreCmhcBackward arch22 tiling done. tile=%ld aivNum=%lu aicNum=%lu hcMix=%ld nPerm=%ld",
            tile, aivNum, aicNum, hcMix, nPerm);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
