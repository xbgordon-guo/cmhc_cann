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
 * \file mhc_pre_cmhc_grad_kernel.h
 * \brief MhcPreCmhcBackward gradient kernel — Softmax-Permutation H_res grad (replaces Sinkhorn).
 *
 * Based on mhc_pre_grad_kernel.h with SinkhornGrad() replaced by SoftmaxPermutationGrad().
 * Removes: skIterCount_, skNormGlobal_, skSumGlobal_, Sinkhorn intermediate buffers.
 * Adds: permIdx[96] compile-time constant for Birkhoff-von Neumann backward.
 */
#ifndef MHC_PRE_CMHC_BACKWARD_OP_KERNEL_MHC_PRE_CMHC_GRAD_KERNEL_H
#define MHC_PRE_CMHC_BACKWARD_OP_KERNEL_MHC_PRE_CMHC_GRAD_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel/math_util.h"
#include "mhc_pre_cmhc_backward_key_arch22.h"
#include "mhc_pre_cmhc_backward_data_arch22.h"

using namespace AscendC;

namespace {
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK / sizeof(float);
constexpr int32_t BYTE_SIZE_PER_REPEAT = 256;
constexpr int32_t ELEMENTS_SIZE_PER_REPEAT = 256 / sizeof(float);
constexpr int32_t REPEAT_LENTH = ELEMENTS_SIZE_PER_REPEAT;
constexpr int32_t BLOCK_PER_REPEAT = 8;
constexpr uint64_t MASK_PRE[] = {0b0000111100001111000011110000111100001111000011110000111100001111};
constexpr uint64_t MASK_POST[] = {0b1111000011110000111100001111000011110000111100001111000011110000};
constexpr uint64_t MASK_POST_SCALE[] = {0b0000000000000000000000000000000000000000000000000000000011110000};
constexpr int32_t PING_PONG_NUM = 2;
constexpr int32_t PRE_POST_NUM = 2;
constexpr int32_t DOUBLE_RATIO = 2;

constexpr int32_t INNER_SPILT_NUM = 8;

constexpr MatmulConfig MHC_PRE_CMHC_GRAD_MM1_CFG = GetMDLConfig(true, false, 0, false, false, false, true);
constexpr MatmulConfig MHC_PRE_CMHC_GRAD_MM2_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

// Permutation matrices are passed from Python (perm_mats) as a tensor input,
// replacing the old hardcoded PERM_IDX compile-time table.
// Shape: (n!, N, N) = (24, 4, 4) for N=4 — loaded from GM into UB at init.

template <typename T>
__aicore__ inline void kahanCustom(LocalTensor<T> &inputTensor, LocalTensor<T> sumTensorList[2],
                                   const int32_t len, int32_t &outPos)
{
    LocalTensor<T> sumTensor = sumTensorList[outPos];
    LocalTensor<T> eTensor = sumTensorList[1 - outPos];
    PipeBarrier<PIPE_V>();
    Sub(inputTensor, inputTensor, eTensor, len);
    PipeBarrier<PIPE_V>();
    Add(eTensor, inputTensor, sumTensor, len);
    PipeBarrier<PIPE_V>();
    Sub(sumTensor, eTensor, sumTensor, len);
    PipeBarrier<PIPE_V>();
    Sub(sumTensor, sumTensor, inputTensor, len);
    PipeBarrier<PIPE_V>();
    outPos = 1 - outPos;
}
} // namespace

template <typename TYPE_X, typename T, bool DETERMINISTIC>
class MhcPreCmhcGradKernel {
public:
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, CType, CType, MHC_PRE_CMHC_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, CType, CType, MHC_PRE_CMHC_GRAD_MM2_CFG> mm2_;

    __aicore__ inline MhcPreCmhcGradKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hc_fn, GM_ADDR pre, GM_ADDR grad_y, GM_ADDR grad_post,
                                GM_ADDR grad_comb, GM_ADDR hc_scale, GM_ADDR hc_base, GM_ADDR h_hat2, GM_ADDR rsqrt,
                                GM_ADDR perm_mats,
                                GM_ADDR grad_x, GM_ADDR grad_hc_fn, GM_ADDR grad_hc_scale, GM_ADDR grad_hc_base,
                                GM_ADDR workspace, const MhcPreCmhcBackwardArch22TilingData *tilingData, TPipe *pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();

        InitTiling(tilingData);
        InitGM(x, hc_fn, pre, grad_y, grad_post, grad_comb, hc_scale, hc_base, h_hat2, rsqrt,
               perm_mats, grad_x, grad_hc_fn, grad_hc_scale, grad_hc_base, workspace);
        InitGradPreStageBuffer();
    }
    __aicore__ inline void Process();

protected:
    int64_t blkIdx_ = 0, aivNum_ = 0, aicNum_ = 0;
    TPipe *pipe_;
    int64_t batchSize_ = 0, seqLength_ = 0, totalTasks_ = 0, totalTasksAligned_ = 0, BSNN = 0, BSN = 0;
    int64_t c_ = 0, n_ = 0, c0_ = 0, c1_ = 0, cTail_ = 0, c0RepeatTime_ = 0, cTailAlign_ = 0, cTailBlockStride_ = 0,
            c1Align_ = 0;
    int64_t tileCoreBS_ = 0;
    int64_t nPerm_ = 0;   // ★ n! = 24 for N=4 (replaces n_*n_=16 in Sinkhorn version)
    int64_t hcMix_ = 0;   // ★ nPerm + 2N = 32 (vs 24 in Sinkhorn)
    int64_t ubSize_ = 0;
    int64_t mm1K_ = 0, mm1M_ = 0, mm1N_ = 0;
    int64_t mm2K_ = 0, mm2M_ = 0, mm2N_ = 0;
    int64_t tileRepeatTimes_ = 0;
    float eps_ = 1e-6f;
    int64_t needAdd = 0;
    event_t eventIdVToMTE3XCast;

    T hcScalePre_, hcScalePost_, hcScaleRes_;

    TQue<QuePosition::VECIN, 1> inputXInQueue;
    TQue<QuePosition::VECIN, 1> inputGradQueue;

    TQue<QuePosition::VECOUT, 1> OutQueue;

    TBuf<TPosition::VECCALC> fusedGradHPre2AndGradHPost2Buf_, gradRsqrtBuf_, gradBiasBuf_, onesBuf_, ScaleBuf_,
        hcBaseBuf_, tempBuf_, permMatsBuf_;

    LocalTensor<T> dBiasLocal_, gradRsqrtLocal_, dPrePostTempLocal_;
    LocalTensor<T> xCastLocal_, gradYCastLocal_, gradXCastLocal_;
    LocalTensor<T> scaleLocal_, dScaleLocal_, dBiasLocalTemp_, dScaleLocalTemp_;
    LocalTensor<T> dBiasLocalList_[2], dpreLocalList_[2];
    int32_t scalePos_ = 0, biasPos_ = 0;
    int32_t onceTask_ = 0;
    LocalTensor<T> gradHResLocal_;
    LocalTensor<T> hcBaseLocal_;
    LocalTensor<T> preBrcbLocal_, dRsqrtBrcbLocal_, rsqrtbrcbLocal_, tmpLocal_, hat2Scale, dhatBeforeNormLocal,
        gradHResTempLocal_, gradHResTempLocal2_, dhatLocal_, rsqrtTempLocal_, gradXCubeLocal_;
    LocalTensor<T> permMatsLocal_;  // ★ (nPerm, N, N) loaded from GM at init

    LocalTensor<T> hatLocal;
    LocalTensor<T> onesLocal_;

    GlobalTensor<TYPE_X> xGlobal_, gradYGlobal_;
    GlobalTensor<T> preGlobal_;
    GlobalTensor<TYPE_X> gradXGlobal_;
    GlobalTensor<T> gradPreGlobal_, gradPostGlobal_;
    GlobalTensor<T> hcScaleGlobal_, hcBaseGlobal_;
    GlobalTensor<T> rsqrtGlobal_;
    GlobalTensor<T> gradHcScaleGlobal_, gradHcBaseGlobal_;
    GlobalTensor<T> h2Global_;
    GlobalTensor<T> gradHResGlobal_;
    GlobalTensor<T> permMatsGlobal_;         // ★ (nPerm, N, N) passed from Python
    GlobalTensor<T> gradH2Global_;
    GlobalTensor<T> gradWeightGlobal_;
    GlobalTensor<T> weightGlobal_;
    GlobalTensor<T> gradHcBaseWSGlobal_;
    GlobalTensor<T> gradHcScaleWSGlobal_;
    GlobalTensor<T> gradWeightWSGlobal_;
    GlobalTensor<T> xWorkspaceGlobal_;
    GlobalTensor<T> gradXCubeGlobal_;

private:
    __aicore__ inline void InitTiling(const MhcPreCmhcBackwardArch22TilingData *tilingData)
    {
        batchSize_ = tilingData->batchSize;
        seqLength_ = tilingData->seqLength;
        aivNum_ = tilingData->aivNum;
        aicNum_ = tilingData->aivNum / DOUBLE_RATIO;
        c_ = tilingData->c;
        n_ = tilingData->n;
        c0_ = tilingData->c0;
        c1_ = tilingData->c1;
        nPerm_ = tilingData->nPerm;             // ★ n! = 24 for N=4
        hcMix_ = nPerm_ + PRE_POST_NUM * n_;    // ★ 24 + 8 = 32 (vs Sinkhorn's 24)
        BSNN = batchSize_ * seqLength_ * n_ * n_;
        BSN = batchSize_ * seqLength_ * n_;
        c1Align_ = Ops::Base::CeilDiv(c_, c0_);
        cTail_ = max((c_ - (c1Align_ - 1) * c0_), static_cast<int64_t>(0));

        cTailAlign_ = AlignUp(cTail_, ELEMENTS_SIZE_PER_BLOCK);
        cTailBlockStride_ = c0_ / ELEMENTS_SIZE_PER_BLOCK - cTailAlign_ / ELEMENTS_SIZE_PER_BLOCK;
        ubSize_ = tilingData->ubSize;
        eps_ = tilingData->eps;
        tileCoreBS_ = tilingData->tileSize;

        c0RepeatTime_ = c0_ / ELEMENTS_SIZE_PER_REPEAT;
        totalTasks_ = batchSize_ * seqLength_;
        totalTasksAligned_ = AlignUp(totalTasks_, aivNum_ * tileCoreBS_);
        if ASCEND_IS_AIC {
            mm1K_ = hcMix_;                      // ★ nPerm + 2N = 32
            mm1M_ = tileCoreBS_ * 2;
            mm1N_ = n_ * c_;

            mm2K_ = batchSize_ * seqLength_;
            mm2M_ = hcMix_;                      // ★ nPerm + 2N = 32
            mm2N_ = n_ * c_;
        }
    }

    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR hc_fn, GM_ADDR pre, GM_ADDR grad_y, GM_ADDR grad_post,
                                  GM_ADDR grad_comb, GM_ADDR hc_scale, GM_ADDR hc_base, GM_ADDR h_hat2, GM_ADDR rsqrt,
                                  GM_ADDR perm_mats,
                                  GM_ADDR grad_x, GM_ADDR grad_hc_fn, GM_ADDR grad_hc_scale, GM_ADDR grad_hc_base,
                                  GM_ADDR workspace)
    {
        // cube and vector shared
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X *>(x));
        gradWeightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_hc_fn));
        // Permutation matrices: passed from Python, pre-computed (n!, N, N)
        permMatsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(perm_mats));
        int64_t workspaceOffset = 0;
        gradH2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
        workspaceOffset += batchSize_ * seqLength_ * hcMix_;
        xWorkspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
        workspaceOffset += batchSize_ * seqLength_ * (n_ * c_);
        gradXCubeGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
        workspaceOffset += batchSize_ * seqLength_ * (n_ * c_);
        if constexpr (DETERMINISTIC == true) {
            gradWeightWSGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
            workspaceOffset += aicNum_ * hcMix_ * (n_ * c_);
        }

        weightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hc_fn));
        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X *>(grad_x));

        if ASCEND_IS_AIV {
            preGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(pre));
            gradYGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X *>(grad_y));

            gradPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_post));

            hcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hc_scale));
            hcBaseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hc_base));

            rsqrtGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rsqrt));
            gradHcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_hc_scale));
            gradHcBaseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_hc_base));
            if constexpr (DETERMINISTIC == true) {
                gradHcScaleWSGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
                workspaceOffset += aivNum_ * (ELEMENTS_SIZE_PER_BLOCK);
                gradHcBaseWSGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + workspaceOffset);
                workspaceOffset += aivNum_ * (hcMix_);
            }
            h2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(h_hat2));

            // Note: no skNormGlobal_ / skSumGlobal_ — Sinkhorn removed

            gradHResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad_comb));
            if constexpr (DETERMINISTIC == false) {
                if (blkIdx_ == aivNum_ - 1) {
                    InitOutput<T>(gradHcBaseGlobal_, (hcMix_), 0);
                    InitOutput<T>(gradHcScaleGlobal_, 3, 0);
                }
                for (int64_t taskOffset = blkIdx_ * tileCoreBS_; taskOffset < n_ * c_;
                    taskOffset += aivNum_ * tileCoreBS_) {
                    int32_t tileTaskCount =
                        min(static_cast<int32_t>(tileCoreBS_), static_cast<int32_t>(n_ * c_ - taskOffset));
                    InitOutput<T>(gradWeightGlobal_[taskOffset * (hcMix_)],
                                (hcMix_) * tileTaskCount, 0);
                }
                SyncAll<true>();
            }
        }
    }

    __aicore__ inline void InitGradPreStageBuffer()
    {
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(fusedGradHPre2AndGradHPost2Buf_, tileCoreBS_ * n_ * 2 * sizeof(float));
            pipe_->InitBuffer(gradRsqrtBuf_, tileCoreBS_ * (hcMix_) * sizeof(float) * 2);
            pipe_->InitBuffer(gradBiasBuf_, 2 * 2 * tileCoreBS_ * (hcMix_) * sizeof(float));
            pipe_->InitBuffer(onesBuf_, tileCoreBS_ * n_ * 2 * sizeof(float));
            pipe_->InitBuffer(hcBaseBuf_, (hcMix_) * sizeof(float));
            pipe_->InitBuffer(ScaleBuf_, BYTE_SIZE_PER_BLOCK * 2);
            pipe_->InitBuffer(inputXInQueue, 2, tileCoreBS_ * (hcMix_ + 2 * n_ + 1) * sizeof(float));
            pipe_->InitBuffer(inputGradQueue, 1, tileCoreBS_ * n_ * ELEMENTS_SIZE_PER_BLOCK * sizeof(float));
            // Removed SKInQueue — no Sinkhorn
            pipe_->InitBuffer(OutQueue, 2, tileCoreBS_ * n_ * c0_ * sizeof(float) / 8);
            auto ubSizeRemain =
                Ops::Base::CeilDiv(tileCoreBS_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)) *
                    ELEMENTS_SIZE_PER_REPEAT * sizeof(float) +
                Ops::Base::CeilDiv(tileCoreBS_ * n_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)) *
                    ELEMENTS_SIZE_PER_REPEAT * sizeof(float) +
                onceTask_ * n_ * c0_ * sizeof(float) * 2 + onceTask_ * c0_ * sizeof(float);
            auto ubSizeRemainHc = tileCoreBS_ * hcMix_ * 8 * sizeof(float);
            if (ubSizeRemainHc > ubSizeRemain) {
                ubSizeRemain = ubSizeRemainHc;
            }

            pipe_->InitBuffer(tempBuf_, ubSizeRemain);

            // ★ Load permutation matrices from GM into UB (nPerm * N * N floats)
            // nPerm = n! = 24 for N=4, total 384 floats = 1536 bytes
            int64_t permMatsSize = nPerm_ * n_ * n_;
            pipe_->InitBuffer(permMatsBuf_, permMatsSize * sizeof(float));
            permMatsLocal_ = permMatsBuf_.Get<T>();
            // Load all permutation data from GM to UB once
            DataCopyPad(permMatsLocal_, permMatsGlobal_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(permMatsSize * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
            PipeBarrier<PIPE_V>();

            eventIdVToMTE3XCast = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            dPrePostTempLocal_ = fusedGradHPre2AndGradHPost2Buf_.Get<T>();
            hcBaseLocal_ = hcBaseBuf_.Get<T>();
            onesLocal_ = onesBuf_.Get<T>();
            scaleLocal_ = ScaleBuf_.Get<T>();
            gradRsqrtLocal_ = gradRsqrtBuf_.Get<T>();
            dBiasLocal_ = gradBiasBuf_.Get<T>();
            dScaleLocal_ = dBiasLocal_[tileCoreBS_ * (hcMix_)];
            dBiasLocalTemp_ = dScaleLocal_[tileCoreBS_ * (hcMix_)];
            dScaleLocalTemp_ = dBiasLocalTemp_[tileCoreBS_ * (hcMix_)];
            dBiasLocalList_[0] = dBiasLocal_;
            dBiasLocalList_[1] = dBiasLocalTemp_;
            onceTask_ = tileCoreBS_ / INNER_SPILT_NUM;

            int32_t offset = 0;
            int32_t brcbAlign = static_cast<int32_t>(
                Ops::Base::CeilDiv(tileCoreBS_ * n_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));
            preBrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * ELEMENTS_SIZE_PER_REPEAT, offset);
            offset += brcbAlign * ELEMENTS_SIZE_PER_REPEAT * sizeof(float);
            brcbAlign = static_cast<int32_t>(
                Ops::Base::CeilDiv(tileCoreBS_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));
            dRsqrtBrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * ELEMENTS_SIZE_PER_REPEAT, offset);
            offset += brcbAlign * ELEMENTS_SIZE_PER_REPEAT * sizeof(float);
            gradYCastLocal_ = tempBuf_.GetWithOffset<T>(onceTask_ * c0_, offset);
            offset += onceTask_ * c0_ * sizeof(float);
            gradXCastLocal_ = tempBuf_.GetWithOffset<T>(onceTask_ * n_ * c0_, offset);
            offset += onceTask_ * n_ * c0_ * sizeof(float);
            xCastLocal_ = tempBuf_.GetWithOffset<T>(onceTask_ * n_ * c0_, offset);
            offset = 0;

            tmpLocal_ = gradXCastLocal_;
            dpreLocalList_[0] = tmpLocal_;
            dpreLocalList_[1] = tmpLocal_[onceTask_ * n_ * ELEMENTS_SIZE_PER_REPEAT];

            // SoftmaxPermutationGrad buffers (replaces SinkhornGrad buffers)
            gradHResTempLocal_ =
                tempBuf_.GetWithOffset<T>(tileCoreBS_ * n_ * ELEMENTS_SIZE_PER_BLOCK * 2, offset);
            offset += tileCoreBS_ * n_ * ELEMENTS_SIZE_PER_BLOCK * 2 * sizeof(float);
            gradHResTempLocal2_ = tempBuf_.GetWithOffset<T>(tileCoreBS_ * n_ * ELEMENTS_SIZE_PER_BLOCK, offset);
            offset = 0;

            dhatLocal_ =
                tempBuf_.GetWithOffset<T>(tileCoreBS_ * (hcMix_) * 2, offset);
            offset += tileCoreBS_ * (hcMix_) * sizeof(float) * 2;
            brcbAlign = static_cast<int32_t>(
                Ops::Base::CeilDiv(tileCoreBS_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));
            rsqrtbrcbLocal_ =
                tempBuf_.GetWithOffset<T>(brcbAlign * ELEMENTS_SIZE_PER_BLOCK * (hcMix_), offset);
            offset += brcbAlign * ELEMENTS_SIZE_PER_BLOCK * (hcMix_) * sizeof(float);
            hat2Scale = tempBuf_.GetWithOffset<T>(tileCoreBS_ * (hcMix_), offset);
            offset += tileCoreBS_ * (hcMix_) * sizeof(float);
            dhatBeforeNormLocal = tempBuf_.GetWithOffset<T>(tileCoreBS_ * (hcMix_), offset);
            offset += tileCoreBS_ * (hcMix_) * sizeof(float);
            hatLocal = tempBuf_.GetWithOffset<T>(tileCoreBS_ * (hcMix_), offset);
            offset += tileCoreBS_ * (hcMix_) * sizeof(float);
            rsqrtTempLocal_ = tempBuf_.GetWithOffset<T>(tileCoreBS_ * (hcMix_), offset);

            Duplicate(onesLocal_, 1.f, tileCoreBS_ * n_ * 2);
            Duplicate(dBiasLocal_, 0.f, 2 * 2 * tileCoreBS_ * (hcMix_));
        }
    }

    __aicore__ inline void ComputeGradPre(const int64_t taskOffset, const int32_t tileTaskCount, const int32_t innerId);

    __aicore__ inline void ComputeGradHHat2(const int64_t taskOffset, const int32_t tileTaskCount);
    // ★★★ SoftmaxPermutationGrad replaces SinkhornGrad ★★★
    __aicore__ inline void SoftmaxPermutationGrad(const int64_t taskOffset, const int32_t tileTaskCount);

    __aicore__ inline void ComputeGradX1(const int64_t taskOffset, const int32_t tileTaskCount, const int32_t innerId);
    __aicore__ inline void GetHcScaleAndHcBase();
    __aicore__ inline void ProcessMatmul1(const int64_t taskOffset, const int32_t mm1M);
    __aicore__ inline void ProcessMatmul2(const int64_t taskOffset, const int32_t mm2K);
    __aicore__ inline void ComputeScaleBias();
    __aicore__ inline void ComputeDeterministic(GlobalTensor<float> &inputGm, GlobalTensor<float> &outputGm,
                                                const int64_t dimR, const int64_t dimA);
};

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::GetHcScaleAndHcBase()
{
    hcScalePre_ = hcScaleGlobal_.GetValue(0);
    hcScalePost_ = hcScaleGlobal_.GetValue(1);
    hcScaleRes_ = hcScaleGlobal_.GetValue(2);
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
    Duplicate(scaleLocal_[8], hcScaleRes_, 8);
    Duplicate(scaleLocal_, hcScalePost_, 8);
    PipeBarrier<PIPE_V>();

    Duplicate(scaleLocal_, hcScalePre_, 4);

    DataCopyPad(hcBaseLocal_, hcBaseGlobal_,
                {static_cast<uint16_t>(1), static_cast<uint32_t>((hcMix_) * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});
}

// =========================================================================
// ★★★ SoftmaxPermutationGrad — replaces SinkhornGrad ★★★
//
// Forward:
//   h_res_logits = r_inv * hc_before_norm[2N:2N+nPerm] * alpha[2] + bias[2N:2N+nPerm]
//   res_coeff = softmax(h_res_logits)
//   h_res = einsum('mr,rij->mij', res_coeff, perms)
//
// Backward (given grad_h_res):
//   1. Recompute res_coeff from saved hc_before_norm + inv_rms + alpha + bias
//   2. grad_res_coeff[p] = sum_i grad_h_res[i, perm[p,i]]
//   3. grad_logits = res_coeff * (grad_res_coeff - sum(grad_res_coeff * res_coeff))
// =========================================================================
template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::SoftmaxPermutationGrad(
    const int64_t taskOffset, const int32_t tileTaskCount)
{
    // Load grad_h_res (shape: tileTaskCount * N * N, stored as tileTaskCount rows of N*N elements)
    // Note: grad_h_res comes in as (tileTaskCount, N, N) FLATTENED — each row is N*N=16 elements for N=4
    gradHResLocal_ = inputGradQueue.AllocTensor<T>();

    DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * n_ * n_],
                {static_cast<uint16_t>(tileTaskCount * n_), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    inputGradQueue.EnQue(gradHResLocal_);
    inputGradQueue.DeQue();

    // nPerm_ is passed from tiling data: n! = 24 for N=4
    // hcMix_ = nPerm_ + 2*n_ = 32 (vs Sinkhorn's n_*n_ + 2*n_ = 24)
    int64_t nPermEffective = nPerm_;

    // ============================================================
    // Step 1: Recompute res_coeff from saved forward intermediates
    // h_res_logits = r_inv * hc_before_norm[2N:2N+nPerm] * alpha[2] + bias[2N:2N+nPerm]
    //
    // We have:
    //   hatLocal = r_inv_broadcast * hc_before_norm * alpha_exp + bias
    //   (already computed in ComputeGradHHat2 before calling this)
    //   hatLocal[2N:2N+nPerm] = h_res_logits BEFORE softmax
    //
    // We need to compute: res_coeff = softmax(h_res_logits)
    // This is the forward softmax output that we need for the backward.
    // ============================================================

    // For each row r in tileTaskCount, compute softmax forward on h_res_logits
    // Store softmax output in gradHResTempLocal2_ as res_coeff
    // And max-subtracted values in gradHResTempLocal_

    // Softmax forward: exp(x - max) / sum(exp(x - max))
    for (int64_t r = 0; r < tileTaskCount; ++r) {
        int64_t logitsOff = r * (hcMix_) + PRE_POST_NUM * n_;
        // Find max (scalar)
        float mx = hatLocal.GetValue(logitsOff);
        for (int64_t k = 1; k < nPermEffective; ++k) {
            float v = hatLocal.GetValue(logitsOff + k);
            if (v > mx) mx = v;
        }
        // Copy logits to contiguous output buffer, then vector ops
        LocalTensor<T> logitsOut = gradHResTempLocal2_[r * nPermEffective];
        LocalTensor<T> logitsSrc = hatLocal[logitsOff];
        DataCopy(logitsOut, logitsSrc, nPermEffective * static_cast<uint32_t>(sizeof(T)));
        PipeBarrier<PIPE_V>();
        // x = x - max (use Adds with negative for scalar subtract)
        Adds(logitsOut, logitsOut, static_cast<T>(-mx), nPermEffective);
        // exp(x)
        Exp(logitsOut, logitsOut, nPermEffective);
        // Sum (scalar)
        float es = 0;
        for (int64_t k = 0; k < nPermEffective; ++k) {
            es += logitsOut.GetValue(k);
        }
        // Normalize: softmax output = res_coeff
        float ies = (es > 0) ? (1.0f / es) : 1.0f;
        Muls(logitsOut, logitsOut, static_cast<T>(ies), nPermEffective);
    }
    PipeBarrier<PIPE_V>();

    // ============================================================
    // Step 2: grad_res_coeff[p] = sum_i sum_j grad_h_res[i,j] * perm_mats[p,i,j]
    // grad_h_res is (tileTaskCount, N, N) stored flattened as (tileTaskCount * N, N)
    //   with each row padded to ELEMENTS_PER_BLOCK (8 elements).
    //   Element (r, i, j) is at: (r * N + i) * ELEMENTS_PER_BLOCK + j
    // perm_mats is (nPerm, N, N) loaded into permMatsLocal_ at init, stored flat.
    //   Element (p, i, j) is at: p * N * N + i * N + j
    // Uses perm_mats passed from Python (supports pure permutations or gamma-blended)
    // ============================================================
    int32_t nInt = static_cast<int32_t>(n_);
    int32_t nPermInt = static_cast<int32_t>(nPermEffective);
    int32_t esb = static_cast<int32_t>(ELEMENTS_SIZE_PER_BLOCK);
    for (int64_t r = 0; r < tileTaskCount; ++r) {
        for (int64_t p = 0; p < nPermEffective; ++p) {
            float sumVal = 0;
            for (int64_t i = 0; i < n_; ++i) {
                for (int64_t j = 0; j < n_; ++j) {
                    float pv = permMatsLocal_.GetValue(
                        static_cast<int>(p * nInt * nInt + i * nInt + j));
                    float gv = gradHResLocal_.GetValue(
                        static_cast<int>((r * nInt + i) * esb + j));
                    sumVal += gv * pv;
                }
            }
            gradHResTempLocal_.SetValue(r * nPermEffective + p, sumVal);
        }
    }
    PipeBarrier<PIPE_V>();

    // ============================================================
    // Step 3: Softmax backward
    // grad_logits = res_coeff * (grad_res_coeff - sum_k(grad_res_coeff_k * res_coeff_k))
    // ============================================================
    for (int64_t r = 0; r < tileTaskCount; ++r) {
        // Compute weighted sum: sum_k(grad_res_coeff_k * res_coeff_k)
        float weightedSum = 0;
        for (int64_t k = 0; k < nPermEffective; ++k) {
            float gradC = gradHResTempLocal_.GetValue(r * nPermEffective + k);
            float resC = gradHResTempLocal2_.GetValue(r * nPermEffective + k);
            weightedSum += gradC * resC;
        }
        // grad_logits_k = res_coeff_k * (grad_res_coeff_k - weightedSum)
        for (int64_t k = 0; k < nPermEffective; ++k) {
            float gradC = gradHResTempLocal_.GetValue(r * nPermEffective + k);
            float resC = gradHResTempLocal2_.GetValue(r * nPermEffective + k);
            float gradLogit = resC * (gradC - weightedSum);
            gradHResTempLocal_.SetValue(r * nPermEffective + k, gradLogit);
        }
    }
    PipeBarrier<PIPE_V>();

    // ============================================================
    // Step 4: Store grad_h_res_logits back to hatLocal (2N:2N+nPerm segment)
    // The downstream ComputeGradHHat2 will chain through alpha[2] and r_inv
    // ============================================================
    for (int64_t r = 0; r < tileTaskCount; ++r) {
        for (int64_t k = 0; k < nPermEffective; ++k) {
            float gl = gradHResTempLocal_.GetValue(r * nPermEffective + k);
            hatLocal.SetValue(r * (hcMix_) + PRE_POST_NUM * n_ + k, gl);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::ComputeGradHHat2(
    const int64_t taskOffset, const int32_t tileTaskCount)
{
    // ★★★ Call SoftmaxPermutationGrad instead of SinkhornGrad ★★★
    SoftmaxPermutationGrad(taskOffset, tileTaskCount);

    // Copy grad_h_res contribution into dhatLocal_ for pre/post section
    for (int32_t loopIdN = 0; loopIdN < 2; loopIdN += 1) {
        Copy(dhatLocal_[ELEMENTS_SIZE_PER_BLOCK + loopIdN * ELEMENTS_SIZE_PER_BLOCK],
             gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
             {static_cast<uint16_t>(3), static_cast<uint16_t>(2), static_cast<uint16_t>((2 + 1) * 8),
              static_cast<uint16_t>(2 * 8)});
    }

    inputGradQueue.FreeTensor(gradHResLocal_);

    // The rest of ComputeGradHHat2 is identical to the reference:
    // loads grad_h_post, hc_before_norm, rsqrt; computes sigmoid backward for pre/post;
    // chains through RMS norm backward; stores grad_H2 to workspace for Cube MatMul
    //
    // === Code below is adapted from MhcPreGradKernel::ComputeGradHHat2 ===

    auto gradHPostLocal_ = inputXInQueue.AllocTensor<T>();
    auto hat2LocalTemp = gradHPostLocal_[tileCoreBS_ * ELEMENTS_SIZE_PER_BLOCK];
    auto rsqrtLocal_ =
        gradHPostLocal_[tileCoreBS_ * ELEMENTS_SIZE_PER_BLOCK + tileCoreBS_ * (hcMix_)];

    DataCopyPad(gradHPostLocal_, gradPostGlobal_[taskOffset * n_],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});

    DataCopyPad(hat2LocalTemp, h2Global_[taskOffset * (hcMix_)],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>((hcMix_) * sizeof(T)),
                 0, 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(rsqrtLocal_, rsqrtGlobal_[taskOffset],
                {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});
    inputXInQueue.EnQue(gradHPostLocal_);
    inputXInQueue.DeQue();

    PipeBarrier<PIPE_V>();

    // GRAD_PRE*1_POST*2
    Axpy(dPrePostTempLocal_, gradHPostLocal_, float(2), tileTaskCount * 2 * n_);

    const uint32_t srcShape[2] = {static_cast<uint32_t>(tileTaskCount * n_), 1};
    const uint32_t dstShape[2] = {static_cast<uint32_t>(tileTaskCount * n_),
                                  static_cast<uint32_t>(hcMix_)};
    PipeBarrier<PIPE_V>();

    AscendC::Broadcast<float, 2, 1>(rsqrtbrcbLocal_, rsqrtLocal_, dstShape, srcShape);

    PipeBarrier<PIPE_V>();

    // GRAD_HAT_RSQRT
    Mul(hat2Scale, scaleLocal_, hat2LocalTemp, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {3, 0, 3, 24, 0, 24});
    Mul(hat2Scale[8], scaleLocal_[8], hat2LocalTemp[8], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
        {3, 0, 3, 24, 0, 24});
    Mul(hat2Scale[16], scaleLocal_[8], hat2LocalTemp[16], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
        {3, 0, 3, 24, 0, 24});
    PipeBarrier<PIPE_V>();
    Mul(hatLocal, hat2Scale, rsqrtbrcbLocal_, (hcMix_) * tileTaskCount);
    PipeBarrier<PIPE_V>();

    Add(hatLocal, hcBaseLocal_, hatLocal, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {3, 0, 3, 24, 0, 24});
    Add(hatLocal[8], hcBaseLocal_[8], hatLocal[8], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {3, 0, 3, 24, 0, 24});
    Add(hatLocal[16], hcBaseLocal_[16], hatLocal[16], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {3, 0, 3, 24, 0, 24});
    PipeBarrier<PIPE_V>();

    auto hatLocalTemp = dhatBeforeNormLocal;

    // Forward pre/post sigmoid
    Muls(hatLocalTemp, hatLocal, float(-1), ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {1, 3, 8, 24});
    PipeBarrier<PIPE_V>();

    Exp(hatLocal, hatLocalTemp, (2 * n_) * tileTaskCount);
    PipeBarrier<PIPE_V>();

    Adds(hatLocal, hatLocal, float(1), (2 * n_) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    Div(hatLocal, onesLocal_, hatLocal, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {1, 0, 1, 8, 0, 8});

    // Sigmoid gradient: x_sigmoid * (1 - x_sigmoid) * grad_output
    PipeBarrier<PIPE_V>();
    Mul(hatLocalTemp, hatLocal, hatLocal, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Sub(hatLocalTemp, hatLocal, hatLocalTemp, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    Mul(dhatLocal_, dPrePostTempLocal_, hatLocalTemp, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {3, 1, 1, 24, 8, 8});
    PipeBarrier<PIPE_V>();
    // dRsqrt = dhatLocal_ * beforenorm
    Mul(gradRsqrtLocal_, dhatLocal_, hat2Scale, (hcMix_) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    // rmsnorm gradient: (-rsqrt³) * grad_rsqrt / nC
    WholeReduceSum(rsqrtTempLocal_, gradRsqrtLocal_, (hcMix_), tileTaskCount, 1, 1, 3);
    PipeBarrier<PIPE_V>();
    Mul(rsqrtTempLocal_, rsqrtTempLocal_, rsqrtLocal_, tileTaskCount);
    PipeBarrier<PIPE_V>();

    Mul(rsqrtLocal_, rsqrtLocal_, rsqrtLocal_, tileTaskCount);
    PipeBarrier<PIPE_V>();
    Mul(rsqrtTempLocal_, rsqrtTempLocal_, rsqrtLocal_, tileTaskCount);

    PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_V>();
    Muls(gradRsqrtLocal_, rsqrtTempLocal_, float(-1) / (n_ * c_), tileTaskCount);

    // dscale = dhatLocal_ * hat_pre * rsqrt
    // dhat2 = dhatLocal_ * rsqrt * scale
    Mul(dhatBeforeNormLocal, rsqrtbrcbLocal_, dhatLocal_, (hcMix_) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    Duplicate(dhatLocal_[tileTaskCount * (hcMix_)], 0.f,
              (tileCoreBS_ - tileTaskCount) * (hcMix_));
    Duplicate(dhatLocal_[(tileCoreBS_ + tileTaskCount) * (hcMix_)], 0.f,
              (tileCoreBS_ - tileTaskCount) * (hcMix_));
    Mul(dhatLocal_[tileCoreBS_ * (hcMix_)], hat2LocalTemp, dhatBeforeNormLocal,
        (hcMix_) * tileTaskCount);
    PipeBarrier<PIPE_V>();

    kahanCustom(dhatLocal_, dBiasLocalList_, tileCoreBS_ * (hcMix_) * 2, scalePos_);

    auto dhat2Local = OutQueue.AllocTensor<float>();
    inputXInQueue.FreeTensor(gradHPostLocal_);

    Mul(dhat2Local, scaleLocal_, dhatBeforeNormLocal, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {3, 0, 3, 24, 0, 24});
    Mul(dhat2Local[8], scaleLocal_[8], dhatBeforeNormLocal[8], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
        {3, 0, 3, 24, 0, 24});
    Mul(dhat2Local[16], scaleLocal_[8], dhatBeforeNormLocal[16], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
        {3, 0, 3, 24, 0, 24});

    OutQueue.EnQue(dhat2Local);
    dhat2Local = OutQueue.DeQue<T>();
    DataCopyPad(gradH2Global_[taskOffset * (hcMix_)], dhat2Local,
                {static_cast<uint16_t>(1),
                 static_cast<uint32_t>(tileTaskCount * (hcMix_) * sizeof(float)), 0, 0, 0});

    OutQueue.FreeTensor(dhat2Local);
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::ComputeGradX1(
    const int64_t taskOffset, const int32_t tileTaskCount, const int32_t innerId)
{
    PipeBarrier<PIPE_V>();

    for (int32_t loopIdC = 0; loopIdC < c1Align_; loopIdC += 1) {
        int64_t copyLen = c0_;
        bool isPad = false;
        uint8_t padLen = 0;
        int64_t ubAlignC = c0_;
        if (loopIdC == c1_) {
            isPad = true;
            copyLen = cTail_;
            ubAlignC = cTailAlign_;
            padLen = static_cast<uint8_t>(cTailAlign_ - cTail_);
        }
        auto xLocal_ = inputXInQueue.AllocTensor<TYPE_X>();
        auto gradXCubeLocal_ = xLocal_.template ReinterpretCast<float>()[onceTask_ * n_ * c0_ / 2];
        auto gradYLocal_ = xLocal_[onceTask_ * n_ * c0_ * 3];

        DataCopyPad(gradYLocal_, gradYGlobal_[taskOffset * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0},
                    {isPad, 0, padLen, 0});
        DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0},
                    {isPad, 0, padLen, 0});
        DataCopyPad(gradXCubeLocal_, gradXCubeGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(T)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(T)), 0, 0},
                    {isPad, 0, padLen, 0});

        inputXInQueue.EnQue(xLocal_);
        inputXInQueue.DeQue();
        PipeBarrier<PIPE_V>();
        Cast(gradYCastLocal_, gradYLocal_, RoundMode::CAST_NONE, ubAlignC * tileTaskCount);

        Cast(xCastLocal_, xLocal_, RoundMode::CAST_NONE, ubAlignC * n_ * tileTaskCount);

        PipeBarrier<PIPE_V>();
        uint8_t blkStride1 = static_cast<uint8_t>(ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        uint8_t blkStride2 = static_cast<uint8_t>(n_ * ubAlignC / ELEMENTS_SIZE_PER_BLOCK);

        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask =
                    min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                Mul(gradXCastLocal_[loopIdN * ubAlignC + loopOffsetC0], gradYCastLocal_[loopOffsetC0],
                    preBrcbLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK + innerId * n_ * ELEMENTS_SIZE_PER_BLOCK], mask,
                    tileTaskCount, {1, 1, 0, blkStride2, blkStride1, static_cast<uint8_t>(n_)});
            }
        }

        PipeBarrier<PIPE_V>();
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask =
                    min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                MulAddDst(gradXCastLocal_[loopIdN * ubAlignC + loopOffsetC0],
                          xCastLocal_[loopIdN * ubAlignC + loopOffsetC0],
                          dRsqrtBrcbLocal_[innerId * ELEMENTS_SIZE_PER_BLOCK], mask, tileTaskCount,
                          {1, 1, 0, blkStride2, blkStride2, 1});
            }
        }

        Add(gradXCastLocal_, gradXCubeLocal_, gradXCastLocal_, ubAlignC * n_ * tileTaskCount);
        PipeBarrier<PIPE_V>();
        inputXInQueue.FreeTensor(xLocal_);

        auto gradXLocalOut = OutQueue.AllocTensor<TYPE_X>();

        Cast(gradXLocalOut, gradXCastLocal_, RoundMode::CAST_RINT, ubAlignC * n_ * tileTaskCount);
        OutQueue.EnQue(gradXLocalOut);
        gradXLocalOut = OutQueue.DeQue<TYPE_X>();

        DataCopyPad(gradXGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], gradXLocalOut,
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), 0,
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0});
        OutQueue.FreeTensor(gradXLocalOut);

        PipeBarrier<PIPE_V>();
    }
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::ComputeDeterministic(
    GlobalTensor<float> &inputGm, GlobalTensor<float> &outputGm, const int64_t dimR, const int64_t dimA)
{
    int64_t queMaxLen = tileCoreBS_ * n_ * c0_ / 8;
    int64_t totalLen = dimR * dimA;
    for (int64_t taskOffset = blkIdx_ * queMaxLen; taskOffset < dimA; taskOffset += aivNum_ * queMaxLen) {
        int64_t tileTaskCount = min(static_cast<int64_t>(queMaxLen), static_cast<int64_t>(dimA - taskOffset));
        if (tileTaskCount > 0) {
            auto localOut = OutQueue.AllocTensor<float>();
            Duplicate(localOut, 0.f, tileTaskCount);
            for (int64_t dimRId = 0; dimRId < dimR; dimRId += 1) {
                auto localIn = inputXInQueue.AllocTensor<float>();

                DataCopyPad(localIn, inputGm[dimRId * dimA + taskOffset],
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(float)), 0, 0, 0},
                            {false, 0, 0, 0});
                inputXInQueue.EnQue(localIn);
                inputXInQueue.DeQue();
                PipeBarrier<PIPE_V>();
                Add(localOut, localOut, localIn, tileTaskCount);
                PipeBarrier<PIPE_V>();
                inputXInQueue.FreeTensor(localIn);
            }
            OutQueue.EnQue(localOut);
            localOut = OutQueue.DeQue<float>();
            DataCopyPad(outputGm[taskOffset], localOut,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(float)), 0, 0, 0});
            OutQueue.FreeTensor(localOut);
        }
    }
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::ComputeGradPre(
    const int64_t taskOffset, const int32_t tileTaskCount, const int32_t innerId)
{
    Duplicate(dpreLocalList_[0], 0.f, tileTaskCount * n_ * ELEMENTS_SIZE_PER_REPEAT);
    Duplicate(dpreLocalList_[1], 0.f, tileTaskCount * n_ * ELEMENTS_SIZE_PER_REPEAT);
    int32_t outPos = 0;
    for (int32_t loopIdC = 0; loopIdC < c1Align_; loopIdC += 1) {
        int64_t copyLen = c0_;
        bool isPad = false;
        uint8_t padLen = 0;
        int64_t ubAlignC = c0_;
        if (loopIdC == c1_) {
            isPad = false;
            copyLen = cTail_;
            ubAlignC = cTailAlign_;
            padLen = static_cast<uint8_t>(cTailAlign_ - cTail_);
        }
        auto xLocal_ = inputXInQueue.AllocTensor<TYPE_X>();

        auto gradYLocal_ = xLocal_[onceTask_ * n_ * c0_];

        DataCopyPad(gradYLocal_, gradYGlobal_[taskOffset * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0},
                    {isPad, 0, padLen, 0});

        DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC],
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(TYPE_X)),
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0},
                    {isPad, 0, padLen, 0});

        inputXInQueue.EnQue(xLocal_);
        inputXInQueue.DeQue();
        PipeBarrier<PIPE_V>();
        auto xCastOutLocal = OutQueue.AllocTensor<float>();

        Cast(xCastOutLocal, xLocal_, RoundMode::CAST_NONE, ubAlignC * n_ * tileTaskCount);

        Cast(gradYCastLocal_, gradYLocal_, RoundMode::CAST_NONE, ubAlignC * tileTaskCount);
        inputXInQueue.FreeTensor(xLocal_);

        PipeBarrier<PIPE_V>();
        uint8_t blkStride = static_cast<uint8_t>(ubAlignC / ELEMENTS_SIZE_PER_BLOCK);

        uint8_t blkStride3 = static_cast<uint8_t>(n_ * ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask =
                    min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                Mul(xCastLocal_[loopIdN * ubAlignC + loopOffsetC0], xCastOutLocal[loopIdN * ubAlignC + loopOffsetC0],
                    gradYCastLocal_[loopOffsetC0], mask, tileTaskCount, {1, 1, 1, blkStride3, blkStride3, blkStride});
            }
        }
        OutQueue.EnQue(xCastOutLocal);
        xCastOutLocal = OutQueue.DeQue<T>();

        DataCopyPad(xWorkspaceGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], xCastOutLocal,
                    {static_cast<uint16_t>(n_ * tileTaskCount), static_cast<uint32_t>(copyLen * sizeof(float)), 0,
                     static_cast<uint32_t>((c_ - copyLen) * sizeof(float)), 0});
        OutQueue.FreeTensor(xCastOutLocal);

        PipeBarrier<PIPE_V>();

        int64_t reduceLen = ubAlignC;
        if (ubAlignC == c0_) {
            Add(xCastLocal_[64], xCastLocal_[64], xCastLocal_[128 + 64], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
            Add(xCastLocal_, xCastLocal_, xCastLocal_[128], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
            PipeBarrier<PIPE_V>();

            Add(xCastLocal_, xCastLocal_, xCastLocal_[64], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
        } else {
            if (cTail_ - (128 + 64) > 0) {
                uint64_t mask = min(static_cast<uint64_t>(cTail_ - (128 + 64)), static_cast<uint64_t>(REPEAT_LENTH));
                Add(xCastLocal_[64], xCastLocal_[64], xCastLocal_[128 + 64], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }
            if (cTail_ - (128) > 0) {
                uint64_t mask = min(static_cast<uint64_t>(cTail_ - (128)), static_cast<uint64_t>(REPEAT_LENTH));
                Add(xCastLocal_, xCastLocal_, xCastLocal_[128], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }
            PipeBarrier<PIPE_V>();
            if (cTail_ - (64) > 0) {
                uint64_t mask = min(static_cast<uint64_t>(cTail_ - (64)), static_cast<uint64_t>(REPEAT_LENTH));
                Add(xCastLocal_, xCastLocal_, xCastLocal_[64], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }
        }
        PipeBarrier<PIPE_V>();

        uint64_t mask = min(static_cast<uint64_t>(cTail_), static_cast<uint64_t>(REPEAT_LENTH));
        PipeBarrier<PIPE_V>();

        LocalTensor<T> sumTensor = dpreLocalList_[outPos];
        LocalTensor<T> eTensor = dpreLocalList_[1 - outPos];
        int64_t len = tileTaskCount * n_ * ELEMENTS_SIZE_PER_REPEAT;
        auto inputTensor = gradYCastLocal_;
        PipeBarrier<PIPE_V>();
        Sub(inputTensor, xCastLocal_, eTensor, mask, tileTaskCount * n_, {1, 1, 1, 8, blkStride, 8});
        PipeBarrier<PIPE_V>();
        Add(eTensor, inputTensor, sumTensor, len);
        PipeBarrier<PIPE_V>();
        Sub(sumTensor, eTensor, sumTensor, len);
        PipeBarrier<PIPE_V>();
        Sub(sumTensor, sumTensor, inputTensor, len);
        PipeBarrier<PIPE_V>();

        outPos = 1 - outPos;
    }
    PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_V>();

    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN += 1) {
        WholeReduceSum(dPrePostTempLocal_[loopIdN + innerId * n_ * 2], dpreLocalList_[outPos][loopIdN * REPEAT_LENTH],
                       REPEAT_LENTH, tileTaskCount, n_ * 2, 1, n_ * 8);
    }
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::ComputeScaleBias()
{
    auto dBiasLocal = dBiasLocalList_[scalePos_];
    auto dScaleLocal = dBiasLocal[tileCoreBS_ * (hcMix_)];

    Add(dScaleLocal[8], dScaleLocal[8], dScaleLocal[16], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,
        {3, 3, 3, 24, 24, 24});

    PipeBarrier<PIPE_V>();

    for (int32_t bsCount = tileCoreBS_ / 2; bsCount > 0; bsCount = bsCount / 2) {
        Add(dBiasLocal, dBiasLocal, dBiasLocal[bsCount * (hcMix_)],
            bsCount * (hcMix_));
        Add(dScaleLocal, dScaleLocal, dScaleLocal[bsCount * (hcMix_)],
            bsCount * (hcMix_));
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    LocalTensor<T> dscaleOut = tempBuf_.GetWithOffset<T>(ELEMENTS_SIZE_PER_BLOCK, 0);

    if constexpr (DETERMINISTIC == false) {
        SetAtomicAdd<T>();
        DataCopyPad(
            gradHcBaseGlobal_, dBiasLocal,
            {static_cast<uint16_t>(1), static_cast<uint32_t>((hcMix_) * sizeof(T)), 0, 0, 0});
    } else {
        DataCopyPad(
            gradHcBaseWSGlobal_[blkIdx_ * (hcMix_)], dBiasLocal,
            {static_cast<uint16_t>(1), static_cast<uint32_t>((hcMix_) * sizeof(T)), 0, 0, 0});
    }
    PipeBarrier<PIPE_V>();

    WholeReduceSum(dscaleOut, dScaleLocal, 4, 1, 1, 3, 8);
    WholeReduceSum(dscaleOut[1], dScaleLocal, MASK_POST_SCALE, 1, 1, 3, 8);
    WholeReduceSum(dscaleOut[2], dScaleLocal[8], 8, 1, 1, 3, 8);
    WholeReduceSum(dscaleOut[3], dScaleLocal, 8, 1, 1, 3, 8);

    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    if constexpr (DETERMINISTIC == false) {
        DataCopyPad(gradHcScaleGlobal_, dscaleOut,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>((3) * sizeof(T)), 0, 0, 0});
        SetAtomicNone();
    } else {
        DataCopyPad(gradHcScaleWSGlobal_[blkIdx_ * (3)], dscaleOut,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>((3) * sizeof(T)), 0, 0, 0});
    }
}

/**
    Constraints:
    c: c >= 64 && c % 64 == 0
    n: n == 4
*/

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::Process()
{
    if ASCEND_IS_AIV {
        GetHcScaleAndHcBase();

        int8_t ping = 0;

        for (int64_t taskOffset = blkIdx_ * tileCoreBS_; taskOffset < totalTasksAligned_;
             taskOffset += aivNum_ * tileCoreBS_) {
            int32_t tileTaskCount =
                min(static_cast<int32_t>(tileCoreBS_), static_cast<int32_t>(totalTasks_ - taskOffset));
            tileRepeatTimes_ = Ops::Base::CeilDiv(static_cast<int64_t>(tileTaskCount) * 2 * n_,
                                                  static_cast<int64_t>(ELEMENTS_SIZE_PER_REPEAT));
            if (tileTaskCount > 0) {
                int32_t innerId = 0;
                Duplicate(dPrePostTempLocal_, 0.f, tileCoreBS_ * n_ * 2);
                for (int64_t taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
                    int32_t tileTaskCountInner =
                        min(static_cast<int32_t>(onceTask_), static_cast<int32_t>(tileTaskCount - taskOffsetInner));

                    ComputeGradPre(taskOffset + taskOffsetInner, tileTaskCountInner, taskOffsetInner);
                    innerId++;
                }
                ComputeGradHHat2(taskOffset, tileTaskCount);
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
            CrossCoreWaitFlag<0x2>(1);
            ping = (ping + 1) % 10;
            if (tileTaskCount > 0) {
                int32_t innerId = 0;
                for (int64_t taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
                    int32_t tileTaskCountInner =
                        min(static_cast<int32_t>(onceTask_), static_cast<int32_t>(tileTaskCount - taskOffsetInner));
                    int32_t brcbAlign = static_cast<int32_t>(Ops::Base::CeilDiv(
                        static_cast<int64_t>(tileTaskCount) * n_, static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));
                    int32_t offset = 0;
                    auto preLocal_ = inputGradQueue.AllocTensor<T>();
                    DataCopyPad(
                        preLocal_, preGlobal_[taskOffset * n_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * n_ * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
                    inputGradQueue.EnQue(preLocal_);
                    inputGradQueue.DeQue();

                    const uint32_t srcShape[2] = {static_cast<uint32_t>(tileTaskCount * n_), 1};
                    const uint32_t dstShape[2] = {static_cast<uint32_t>(tileTaskCount * n_),
                                                  ELEMENTS_SIZE_PER_BLOCK};
                    Brcb(preBrcbLocal_, preLocal_, brcbAlign, {static_cast<uint8_t>(1), static_cast<uint8_t>(8)});
                    inputGradQueue.FreeTensor(preLocal_);

                    offset += brcbAlign * 8 * sizeof(float);
                    brcbAlign = static_cast<int32_t>(Ops::Base::CeilDiv(
                        static_cast<int64_t>(tileTaskCount), static_cast<int64_t>(ELEMENTS_SIZE_PER_BLOCK)));

                    offset += brcbAlign * 8 * sizeof(float);

                    Brcb(dRsqrtBrcbLocal_, gradRsqrtLocal_, brcbAlign,
                         {static_cast<uint8_t>(1), static_cast<uint8_t>(8)});
                    ComputeGradX1(taskOffset + taskOffsetInner, tileTaskCountInner, taskOffsetInner);
                    innerId++;
                }
            }
        }

        tileRepeatTimes_ = Ops::Base::CeilDiv(tileCoreBS_ * 2 * n_, static_cast<int64_t>(ELEMENTS_SIZE_PER_REPEAT));
        ComputeScaleBias();

        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    }

    if ASCEND_IS_AIC {
        int8_t ping = 0;
        for (int64_t taskOffset = blkIdx_ * 2 * tileCoreBS_; taskOffset < totalTasksAligned_;
             taskOffset += aicNum_ * 2 * tileCoreBS_) {
            int32_t tileTaskCount =
                min(static_cast<int32_t>(2 * tileCoreBS_), static_cast<int32_t>(totalTasks_ - taskOffset));
            CrossCoreWaitFlag<0x2>(0);

            if (tileTaskCount > 0) {
                ProcessMatmul1(taskOffset, tileTaskCount);
                ProcessMatmul2(taskOffset, tileTaskCount);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(1);

            ping = (ping + 1) % 10;
        }
    }
    if constexpr (DETERMINISTIC == true) {
        SyncAll<false>();
        if ASCEND_IS_AIV {
            int64_t useCoreNum =
                min(Ops::Base::CeilDiv(totalTasks_, tileCoreBS_), aivNum_);
            ComputeDeterministic(gradHcBaseWSGlobal_, gradHcBaseGlobal_, useCoreNum, (hcMix_));
            ComputeDeterministic(gradHcScaleWSGlobal_, gradHcScaleGlobal_, useCoreNum, 3);
            useCoreNum = Ops::Base::CeilDiv(useCoreNum, static_cast<int64_t>(2));
            ComputeDeterministic(gradWeightWSGlobal_, gradWeightGlobal_, useCoreNum,
                                 (hcMix_) * (n_ * c_));
        }
    }
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::ProcessMatmul1(
    const int64_t taskOffset, const int32_t mm1M)
{
    if (mm1M <= 0)
        return;

    mm1_.SetTensorA(gradH2Global_[taskOffset * mm1K_]);
    mm1_.SetTensorB(weightGlobal_);
    mm1_.SetHF32(true, 1);
    mm1_.SetOrgShape(mm1M, mm1N_, mm1K_);
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    mm1_.template IterateAll<false>(gradXCubeGlobal_[taskOffset * (n_ * c_)]);
    mm1_.End();
}

template <typename TYPE_X, typename T, bool DETERMINISTIC>
__aicore__ inline void MhcPreCmhcGradKernel<TYPE_X, T, DETERMINISTIC>::ProcessMatmul2(
    const int64_t taskOffset, const int32_t mm2K)
{
    if (mm2K <= 0)
        return;

    mm2_.SetTensorA(gradH2Global_[taskOffset * mm2M_], true);
    mm2_.SetTensorB(xWorkspaceGlobal_[taskOffset * mm2N_]);
    mm2_.SetHF32(true, 1);
    mm2_.SetOrgShape(mm2M_, mm2N_, mm2K);
    mm2_.SetSingleShape(mm2M_, mm2N_, mm2K);
    if constexpr (DETERMINISTIC == true) {
        mm2_.template IterateAll<false>(gradWeightWSGlobal_[blkIdx_ * (mm2M_ * mm2N_)], needAdd);
        needAdd = 1;
    } else {
        mm2_.template IterateAll<false>(gradWeightGlobal_, 1);
    }
    mm2_.End();
}

#endif // MHC_PRE_CMHC_BACKWARD_OP_KERNEL_MHC_PRE_CMHC_GRAD_KERNEL_H
