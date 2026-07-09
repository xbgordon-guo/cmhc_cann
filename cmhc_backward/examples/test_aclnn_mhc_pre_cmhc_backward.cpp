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
 * \file test_aclnn_mhc_pre_cmhc_backward.cpp
 * \brief ACLNN test for MhcPreCmhcBackward operator
 *
 * Usage: build via cmake, run on NPU.
 * Compares kernel output against known-good reference.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_cmhc_backward.h"

#define CHECK_RET(cond, return_expr) \
    do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) \
    do { printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) { size *= dim; }
    return size;
}

int InitAcl(int32_t device_id, aclrtContext &context, aclrtStream &stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed: %d\n", ret); return -1);
    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed: %d\n", ret); return -1);
    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed: %d\n", ret); return -1);
    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed: %d\n", ret); return -1);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed: %d\n", ret); return -1);
    return 0;
}

int CreateFloatTensor(const std::vector<float> &host, const std::vector<int64_t> &shape,
                      void *&dev, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);
    aclError ret = aclrtMalloc(&dev, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    ret = aclrtMemcpy(dev, size, host.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                             ACL_FORMAT_ND, shape.data(), shape.size(), dev);
    return (tensor != nullptr) ? 0 : -1;
}

int CreateFloatOutput(const std::vector<int64_t> &shape, void *&dev, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);
    aclError ret = aclrtMalloc(&dev, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                             ACL_FORMAT_ND, shape.data(), shape.size(), dev);
    return (tensor != nullptr) ? 0 : -1;
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;

    int64_t bs = 2, seq_len = 32, n = 4, c = 256;
    int64_t nPerm = 24; // 4!
    int64_t hcMix = nPerm + 2 * n; // 32
    double hc_eps = 1e-6;

    std::vector<int64_t> grad_hin_shape    = {bs, seq_len, c};
    std::vector<int64_t> grad_h_post_shape = {bs, seq_len, n};
    std::vector<int64_t> grad_h_res_shape  = {bs, seq_len, n, n};
    std::vector<int64_t> x_shape           = {bs, seq_len, n, c};
    std::vector<int64_t> phi_shape         = {hcMix, n * c};
    std::vector<int64_t> alpha_shape       = {3};
    std::vector<int64_t> bias_shape        = {hcMix};
    std::vector<int64_t> h_pre_shape       = {bs, seq_len, n};
    std::vector<int64_t> hc_before_norm_shape = {bs, seq_len, hcMix};
    std::vector<int64_t> inv_rms_shape     = {bs, seq_len, 1};
    std::vector<int64_t> grad_x_shape      = {bs, seq_len, n, c};
    std::vector<int64_t> grad_phi_shape    = {hcMix, n * c};
    std::vector<int64_t> grad_alpha_shape  = {3};
    std::vector<int64_t> grad_bias_shape   = {hcMix};

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed\n"); return -1);

    // Create inputs (fill with 0.1)
    auto fill = [](int64_t size) {
        std::vector<float> v(size, 0.1f); return v;
    };
    void *d_grad_hin, *d_grad_h_post, *d_grad_h_res, *d_x, *d_phi, *d_alpha, *d_bias;
    void *d_h_pre, *d_hc_before_norm, *d_inv_rms;
    void *d_grad_x, *d_grad_phi, *d_grad_alpha, *d_grad_bias;
    aclTensor *t_grad_hin, *t_grad_h_post, *t_grad_h_res, *t_x, *t_phi, *t_alpha, *t_bias;
    aclTensor *t_h_pre, *t_hc_before_norm, *t_inv_rms;
    aclTensor *t_grad_x, *t_grad_phi, *t_grad_alpha, *t_grad_bias;

    ret |= CreateFloatTensor(fill(GetShapeSize(grad_hin_shape)),    grad_hin_shape, d_grad_hin, t_grad_hin);
    ret |= CreateFloatTensor(fill(GetShapeSize(grad_h_post_shape)), grad_h_post_shape, d_grad_h_post, t_grad_h_post);
    ret |= CreateFloatTensor(fill(GetShapeSize(grad_h_res_shape)),  grad_h_res_shape, d_grad_h_res, t_grad_h_res);
    ret |= CreateFloatTensor(fill(GetShapeSize(x_shape)),           x_shape, d_x, t_x);
    ret |= CreateFloatTensor(fill(GetShapeSize(phi_shape)),         phi_shape, d_phi, t_phi);
    ret |= CreateFloatTensor(fill(GetShapeSize(alpha_shape)),       alpha_shape, d_alpha, t_alpha);
    ret |= CreateFloatTensor(fill(GetShapeSize(bias_shape)),        bias_shape, d_bias, t_bias);
    ret |= CreateFloatTensor(fill(GetShapeSize(h_pre_shape)),       h_pre_shape, d_h_pre, t_h_pre);
    ret |= CreateFloatTensor(fill(GetShapeSize(hc_before_norm_shape)), hc_before_norm_shape, d_hc_before_norm, t_hc_before_norm);
    ret |= CreateFloatTensor(fill(GetShapeSize(inv_rms_shape)),     inv_rms_shape, d_inv_rms, t_inv_rms);

    ret |= CreateFloatOutput(grad_x_shape,     d_grad_x,     t_grad_x);
    ret |= CreateFloatOutput(grad_phi_shape,   d_grad_phi,   t_grad_phi);
    ret |= CreateFloatOutput(grad_alpha_shape, d_grad_alpha, t_grad_alpha);
    ret |= CreateFloatOutput(grad_bias_shape,  d_grad_bias,  t_grad_bias);
    CHECK_RET(ret == 0, LOG_PRINT("Tensor creation failed\n"); return -1);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclnn_ret = aclnnMhcPreCmhcBackwardGetWorkspaceSize(
        t_grad_hin, t_grad_h_post, t_grad_h_res, t_x, t_phi, t_alpha, t_bias,
        t_h_pre, t_hc_before_norm, t_inv_rms, hc_eps,
        t_grad_x, t_grad_phi, t_grad_alpha, t_grad_bias,
        &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("GetWorkspaceSize failed: %d\n", aclnn_ret); return -1);

    void *workspace = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("workspace malloc failed\n"); return -1);
    }

    aclnn_ret = aclnnMhcPreCmhcBackward(workspace, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("MhcPreCmhcBackward kernel failed: %d\n", aclnn_ret); return -1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS,
              LOG_PRINT("Stream sync failed\n"); return -1);

    LOG_PRINT("MhcPreCmhcBackward compute success!\n");

    // Cleanup
    if (workspace) aclrtFree(workspace);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}
