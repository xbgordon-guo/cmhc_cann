# mhc_pre_cmhc_backward — cmhc 反向梯度算子

基于 mhc_pre_sinkhorn_backward，将 Sinkhorn-Knopp 迭代反向替换为 Softmax-Permutation 直接反向。

## 关键差异（vs mhc_pre_sinkhorn_backward）

| 项目 | mhc_pre_sinkhorn_backward | mhc_pre_cmhc_backward |
|------|--------------------------|---------------|
| H_res 算法 | Sinkhorn-Knopp 迭代 | Softmax + Permutation einsum |
| 输入数量 | 12 (含 sum_out, norm_out) | 10 (移除 Sinkhorn 中间态) |
| hcMix | n²+2n = 24 | n!+2n = 32 |
| nPerm | 无 (用 n² = 16) | n! = 24 |
| SinkhornGrad | ~200行迭代反传 | SoftmaxPermutationGrad ~100行直接计算 |

## 工程结构

```
mhc_pre_cmhc_backward/
├── CMakeLists.txt
├── backward_reference.py                # PyTorch 反向验证
├── op_graph/mhc_pre_cmhc_backward_proto.h       # IR 定义
├── op_host/
│   ├── CMakeLists.txt
│   ├── mhc_pre_cmhc_backward_def.cpp            # OpDef 注册
│   ├── mhc_pre_cmhc_backward_infershape.cpp     # 形状推导
│   └── op_tiling/
│       ├── mhc_pre_cmhc_backward_tiling.h/.cpp  # Tiling 分发
│       └── arch22/
│           ├── mhc_pre_cmhc_backward_arch22_tiling.h
│           └── mhc_pre_cmhc_backward_arch22_tiling.cpp
├── op_kernel/
│   ├── mhc_pre_cmhc_backward.cpp               # Kernel 入口
│   └── arch22/
│       ├── mhc_pre_cmhc_backward_data_arch22.h  # TilingData 结构
│       ├── mhc_pre_cmhc_backward_key_arch22.h   # 模板 Key
│       └── mhc_pre_cmhc_grad_kernel.h          # ★ 核心梯度计算
└── examples/
    └── test_aclnn_mhc_pre_cmhc_backward.cpp
```

## 输入输出

### 输入 (10)
- grad_hin:    (B,S,C)      BF16/FP16 — loss对h_in的梯度
- grad_h_post: (B,S,N)      FP32      — loss对h_post的梯度
- grad_h_res:  (B,S,N,N)    FP32      — loss对h_res的梯度
- x:           (B,S,N,C)    BF16/FP16 — 原始输入
- phi:         (hcMix,N*C)  FP32      — 权重矩阵
- alpha:       (3,)         FP32      — 缩放系数
- bias:        (hcMix,)     FP32      — 偏置
- h_pre:       (B,S,N)      FP32      — 正向pre激活值
- hc_before_norm: (B,S,hcMix) FP32    — 正向投影结果H
- inv_rms:     (B,S)        FP32      — 正向RMS因子

### 输出 (4)
- grad_x:     (B,S,N,C)    BF16/FP16
- grad_phi:   (hcMix,N*C)  FP32
- grad_alpha: (3,)         FP32
- grad_bias:  (hcMix,)     FP32

## 反向计算图

```
grad_h_in → [加权求和反向] → grad_x_streams, grad_h_pre
grad_h_post → [sigmoid反向] → grad_H_post_raw
grad_h_res → [Permutation einsum反向] → grad_res_coeff
                                       → [softmax反向] → grad_H_res_logits
  
grad_H_raw = concat(grad_H_pre_raw, grad_H_post_raw, grad_H_res_logits)
  → [RMS反向] → grad_r_inv
  → [chain through alpha, r_inv] → grad_H
  → [MatMul反向] → grad_x_matmul, grad_phi
  → [scale/bias reduce] → grad_alpha, grad_bias

grad_x = grad_x_matmul + grad_x_rms + grad_x_streams
```

## 验证

```bash
python3 backward_reference.py
```
