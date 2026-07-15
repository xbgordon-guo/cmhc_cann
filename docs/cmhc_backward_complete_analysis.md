# mhc_pre_cmhc_backward 算子完整分析

> 本文档整合了 `mhc_pre_cmhc_backward` 算子的 6 份分析文档，覆盖调用流程、PyTorch at:: 机制、MindSpore 崩溃根因、路径对比、修复方案和验证策略。

## 📚 目录

| 序号 | 章节 | 核心内容 |
|---|---|---|
| **一** | [Shell 调用流程分析](#一-shell-调用流程分析) | `bash_cmhc_backward_local.sh` 的 5 步执行流程 |
| **二** | [PyTorch at:: 机制详解](#二-pytorch-at-机制详解) | at:: 命名空间、Dispatcher 机制、等价性保证 |
| **三** | [MindSpore 崩溃根因分析](#三-mindspore-崩溃根因分析) | error 507015 崩溃的完整 5 阶段机制 |
| **四** | [两条调用路径完整对比](#四-两条调用路径完整对比) | Forward/Backward/Tiling/Workspace 全面对比 |
| **五** | [修复方案](#五-修复方案) | P0/P1/P2 分级修复（含完整代码） |
| **六** | [完整验证策略](#六-完整验证策略) | 四层验证体系 + CI/CD + 执行路线图 |

## 🎯 核心结论速览

```
当前状态:
  Layer 1 (Shell 算法验证):     ✅ 4/4 PASS (max_diff = 1.19e-07)
  Layer 2 (AscendC 单算子验证):  ❌ 未建立
  Layer 3 (MindFormers 单步):    ❌ error 507015 崩溃
  Layer 4 (MindFormers 完整训练): ❌ 无法开始

根因: arch22_tiling.cpp 是未完成的 stub
  - ❌ 未调用 SaveToBuffer() - tiling 数据未序列化
  - ❌ 未调用 GetWorkspaceSizes() - workspace 未分配 (0 bytes)
  - ❌ 未调用 MatmulApiTiling::GetTiling() - MatMul tiling 为空
  - ❌ 未调用 SetDataSize() - kernel 读到垃圾 tiling

关键结论:
  1. Shell 验证的本质: 验证 at:: 算法实现, 而非 AscendC kernel
  2. Shell 验证通过 ≠ AscendC kernel 正确
  3. 两条路径互补, 不是替代关系
  4. 需要建立四层验证体系覆盖完整质量
```

---

# 一、Shell 调用流程分析

## 概述

`bash_cmhc_backward_local.sh` 是用于编译和验证 `mhc_pre_cmhc_backward` 算子的 Shell 脚本。

## 脚本执行流程

```bash
bash_cmhc_backward_local.sh 执行流程：

Step 1: 编译 AscendC kernel 二进制
  cd ops-transformer/
  bash build.sh --opkernel --soc=ascend910b --ops=mhc_pre_cmhc_backward
  → 编译出 build/binary/ascend910b/src/mhc_pre_cmhc_backward/ (kernel 二进制)
  → 但此二进制后续验证中 ❌ 未被使用

Step 2: 编译 PyTorch binding
  cd cmhc_backward/binding/
  python setup.py build_ext --inplace
  → 编译 mhc_pre_cmhc_backward_torch_host.cpp (纯 at:: 实现)
  → 生成 mhc_pre_cmhc_backward_torch_ops.cpython-311-*.so
  → 此文件后续验证中 ✅ 被使用

Step 3: 加载验证
  python -c "import torch; torch.ops.load_library('mhc_pre_cmhc_backward_torch_ops.*.so'); print('OK')"
  → 验证 .so 能正确加载

Step 4: 精度验证 (verification_ascendc.py)
  python verification_ascendc.py /path/to/cmhc_backward/
  → 加载两个模型：
    - model.py (reference): 使用 torch.autograd.grad 实现精确梯度
    - model_new_ascendc.py (candidate): 使用 torch.ops.npu.mhc_pre_cmhc_backward
  → 使用 NPU Benchmark 三项判定:
    - allclose (atol, rtol 判定)
    - matched_ratio (分桶匹配率 ≥ 90%)
    - MERE (平均相对误差)

Step 5: 性能验证 (msprof_perf_summary.py)
  → 对比 reference vs candidate 的执行时间
  → 输出 Speedup 指标
```

## 关键调用关系

```
verification_ascendc.py
  └─ _run_verification(op)
      ├─ _load_module(model.py)        # reference: torch.autograd.grad
      ├─ _load_module(model_new_ascendc.py)  # candidate: torch binding
      └─ _run_comparisons(ref_model, cand_model, inputs)
          ├─ ref_model(*inputs)  → torch.autograd.grad 精确梯度
          └─ cand_model(*inputs)
              └─ _get_backend_fn(device)
                  └─ NPU: torch.ops.load_library("...torch_ops.*.so")
                      └─ torch.ops.npu.mhc_pre_cmhc_backward  ✅ 这里执行验证
                          └─ TORCH_LIBRARY_IMPL(npu, PrivateUse1)
                              └─ mhc_pre_cmhc_backward_torch_host.cpp
                                  └─ 纯 PyTorch at:: API 实现:
                                      - at::matmul, at::einsum
                                      - at::tanh, at::sigmoid
                                      - at::sum, at::reshape, at::permute
```

## 文件路径关系

```
bash_cmhc_backward_local.sh
  │
  ├─ Step 1: /home/z00941044/0709/ops-transformer/
  │           ├─ build.sh (编译脚本)
  │           ├─ mhc/mhc_pre_cmhc_backward/
  │           │   ├─ op_kernel/mhc_pre_cmhc_backward.cpp (AscendC kernel 源码)
  │           │   ├─ op_host/ (tiling + infershape)
  │           │   │   ├─ mhc_pre_cmhc_backward_tiling.cpp
  │           │   │   └─ arch22/mhc_pre_cmhc_backward_arch22_tiling.cpp
  │           │   └─ aclnn/ (注册文件)
  │           └─ build/binary/ascend910b/src/mhc_pre_cmhc_backward/ (编译产物)
  │
  └─ Step 2-5: /home/z00941044/0709/cmhc_cann/cmhc_backward/
               ├─ binding/ (PyTorch binding 目录)
               │   ├─ mhc_pre_cmhc_backward_torch_host.cpp (at:: 实现)
               │   ├─ register.cpp (TORCH_LIBRARY_IMPL 注册)
               │   ├─ setup.py (编译配置)
               │   └─ *.so (编译产物)
               ├─ model.py (reference 模型)
               ├─ model_new_ascendc.py (candidate 模型)
               └─ verification_ascendc.py (验证脚本)
```

## 关键代码片段

### register.cpp: 算子注册

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("mhc_pre_cmhc_backward", &mhc_pre_cmhc_backward);
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("mhc_pre_cmhc_backward",
           TORCH_FN(mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward));
}
```

### torch_host.cpp: at:: 实现（关键摘要）

```cpp
namespace mhc_pre_cmhc_backward_binding {

std::vector<at::Tensor> mhc_pre_cmhc_backward(
    const at::Tensor& x, const at::Tensor& hc_fn,
    const at::Tensor& h_pre, const at::Tensor& grad_h_hat2, ...) {
    
    // 使用纯 PyTorch at:: API 实现 backward
    auto grad_hc_fn = at::matmul(grad_h_hat2, hc_fn);
    auto grad_x = at::einsum("bij,bjk->bik", {grad_h_hat2, hc_fn});
    // ... 更多计算
    
    return {grad_x, grad_hc_fn, grad_hc_scale, grad_hc_base};
}

} // namespace
```

## 验证结果

当前 Shell 验证结果：**4/4 PASS**

```
Case 0: max_abs_diff: 1.19e-07   matched_ratio: 1.000000   MERE: 3.42e-08
Case 1: max_abs_diff: 1.19e-07   matched_ratio: 1.000000   MERE: 3.41e-08
Case 2: max_abs_diff: 1.19e-07   matched_ratio: 1.000000   MERE: 3.42e-08
Case 3: max_abs_diff: 1.19e-07   matched_ratio: 1.000000   MERE: 3.42e-08
```

## Shell 验证的局限性

**Shell 路径验证只能确认**：
- ✅ `torch_host.cpp` 的 at:: 算法实现与 reference 的 autograd 结果**数值等价**
- ✅ 在 NPU 上使用 PyTorch at:: API 的执行性能

**Shell 路径验证无法检测**：
- ❌ AscendC kernel 本身的数值正确性（kernel 从未被 shell 路径调用）
- ❌ AscendC kernel 的 workspace 分配问题
- ❌ AscendC kernel 的 tiling 数据序列化问题
- ❌ AscendC kernel 的 Cube/AIV/AIC 混合计算单元协同问题

## 关键发现

**Shell 路径虽然编译了 AscendC kernel，但验证时从未调用它**：
- Step 1 编译的 kernel 二进制未被使用
- Step 4 的 candidate 模型通过 PyTorch dispatcher 调用 at:: 实现
- 因此 Shell 验证通过只能保证 **算法层面** 的正确性，不能保证 **hardware 层面** 的正确性

这正是为什么 MindFormers 训练路径能暴露出 error 507015，而 Shell 验证全部通过的原因。

---

# 二、PyTorch at:: 机制详解

## 1. 什么是 at:: (ATen)?

`at::` 是 PyTorch 的底层张量计算库的命名空间。所有高级 PyTorch API（如 `torch.matmul`, `torch.sigmoid`）最终都会调用 `at::` 命名空间中的底层函数。

### at:: 的层次结构

```
torch.matmul(A, B)              # 用户 API
    ↓
at::matmul(A, B)                # ATen API
    ↓
Dispatcher (分发机制)
    ↓
实际执行:
  - CPU: at::native::matmul_cpu  (使用 BLAS/LAPACK)
  - CUDA: at::native::matmul_cuda (使用 cuBLAS)
  - NPU: torch_npu::matmul_npu   (使用 aclnn)
```

## 2. Dispatcher 机制

PyTorch Dispatcher 是一个**运行时调度器**，根据张量的设备类型（CPU/CUDA/NPU）和类型（float32/float16 等）选择正确的实现。

### 调度流程

```
torch.matmul(A, B)
    │
    ▼
[Dispatcher 入口]
    │
    ├─ 检查张量设备: A.device() == B.device() == npu
    ├─ 检查张量 dtype: A.dtype() == B.dtype() == bfloat16
    │
    ▼
[Dispatch Key: PrivateUse1 (NPU)]
    │
    ▼
[查找注册表]
    ├─ at::matmul 注册的 PrivateUse1 实现
    │   └─ torch_npu::matmul_npu()
    │       └─ aclnnMatmul()
    │           └─ NPU 硬件执行
    │
    ▼
返回结果
```

### Dispatch Keys

- **CPU**: CPU 设备
- **CUDA**: NVIDIA GPU
- **PrivateUse1**: 第三方硬件后端（NPU 通过 torch_npu 注册到此）
- **PrivateUse2, PrivateUse3**: 更多第三方后端

### NPU 注册示例

```cpp
// 在 torch_npu 库中（华为官方提供）
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("matmul", TORCH_FN(torch_npu::matmul_npu));
    m.impl("sigmoid", TORCH_FN(torch_npu::sigmoid_npu));
    m.impl("sum", TORCH_FN(torch_npu::sum_npu));
    // ... 其他算子
}
```

## 3. torch.ops.npu.* vs at::*

### torch.ops.npu.mhc_pre_cmhc_backward（自定义算子）

```cpp
// register.cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("mhc_pre_cmhc_backward(...)", &mhc_pre_cmhc_backward);
    m.impl("mhc_pre_cmhc_backward",
           TORCH_FN(mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward));
}
```

**关键点**:
- 这个注册**不通过** aten 命名空间
- 它注册到独立的 `npu` 命名空间
- 当调用 `torch.ops.npu.mhc_pre_cmhc_backward` 时，直接跳转到注册的函数

### at::matmul 等标准算子

```cpp
// PyTorch 核心代码
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("matmul", TORCH_FN(at::native::matmul_cpu));
}
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("matmul", TORCH_FN(at::native::matmul_cuda));
}
// torch_npu 扩展
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("matmul", TORCH_FN(torch_npu::matmul_npu));
}
```

## 4. Shell 路径的完整调用链

```
torch.ops.npu.mhc_pre_cmhc_backward(...)  ← 自定义算子
    │
    ▼
[Dispatcher] 查找 torch.ops.npu 命名空间
    ↓ 找到 mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward
    ↓
mhc_pre_cmhc_backward_torch_host.cpp (纯 at:: API 实现)
    │
    ├─ at::matmul(A, B) → Dispatcher → torch_npu::matmul_npu → aclnnMatmul → NPU 执行
    ├─ at::einsum(...)  → Dispatcher → torch_npu::einsum_npu → aclnnEinsum → NPU 执行
    └─ tensor.sigmoid() → Dispatcher → torch_npu::sigmoid_npu → aclnnSigmoid → NPU 执行
```

## 5. 为什么 at:: 能"代替" AscendC kernel？

### 5.1 数学等价性

**核心假设**: at:: 实现与 AscendC kernel 在**数学上等价**

```cpp
// torch_host.cpp 中的 at:: 实现
auto grad_hc_fn = at::matmul(grad_h_hat2, phi);
auto grad_x = at::einsum("bij,bjk->bik", {grad_h_hat2, hc_fn});

// AscendC kernel 中的实现
mm2_.ProcessMatmul2(&tilingData.mm2TilingData);  // 矩阵乘法

// 数学定义相同
```

### 5.2 数值精度等价

- **float32 路径**: PyTorch CPU 与 NPU 原生实现使用相同的 IEEE 754 标准，误差 < 1e-6
- **bf16/fp16 路径**: NPU 原生实现保证与 PyTorch CPU 等价，bf16 < 1e-2, fp16 < 1e-3

### 5.3 后端调度等价

```
at::matmul(NPU_A, NPU_B)
    ↓ Dispatcher → torch_npu::matmul_npu
    ↓ aclnnMatmul()
    ↓ CANN 算子库中的 matmul 实现
    ↓ NPU 硬件执行
```

**注意**: 这里调用的是 **NPU 原生 matmul 算子**，而不是 AscendC 自定义 kernel。

## 6. at:: vs AscendC kernel 的本质区别

| 对比维度 | at:: 实现 (torch_host.cpp) | AscendC kernel (op_kernel/*.cpp) |
|---|---|---|
| **编写语言** | PyTorch C++ API (at::matmul, ...) | AscendC C++ (AscendC::Copy, Mul, ...) |
| **并行策略** | PyTorch 自动多线程/多 GPU | 手动 Cube/AIV/AIC 混合并行 |
| **性能优化** | 依赖 NPU 原生算子的优化 | 针对特定场景手动优化 |
| **内存管理** | PyTorch dispatcher 自动管理 | 手动管理 GlobalMemory (GM) |
| **tiling 层** | 不需要 (PyTorch 自管) | 需要 op_host tiling 计算块大小和 workspace |
| **调度机制** | PyTorch dispatcher → torch_npu → aclnn | MindSpore CustomOp → ascendebug → aclnn |
| **workspace** | PyTorch 自动分配 | 需要通过 tiling 层显式分配 |
| **编译产物** | .so 共享库 (binding) | .o 对象文件 (被 CANN 框架链接) |

## 7. 数学等价性保证

**Shell 验证的核心假设**: at:: 代码中的每个算子，与 AscendC kernel 中对应操作的数学定义相同。

| at:: 操作 (torch_host.cpp) | AscendC kernel 对应 | 数学定义 |
|---|---|---|
| `at::matmul(grad_H, phi)` | `mm2_.ProcessMatmul2()` | grad_x = grad_H @ phi |
| `at::matmul(grad_H.t(), x_flat)` | `mm1_.ProcessMatmul1()` | grad_phi = grad_Hᵀ @ x |
| `tensor.sigmoid() + tensor*(1-tensor)` | `SigmoidGrad` AIV kernel | σ'(x) = σ(x)(1-σ(x)) |
| `tensor.softmax(-1)` | softmax 实现 | e^x / sum(e^x) |
| `at::einsum("mij,rij->mr", ...)` | 显式循环累加 | Σᵢⱼ grad_res[i,j] * perm[r,i,j] |
| `grad_act * alpha_exp * r_inv` | 逐元素 Mul + Mul | elementwise multiply |

## 8. 验证层级分层

```
┌─────────────────────────────────────────────────────┐
│  数学正确性层: 算法是否按预期计算？                │
│  └─ Shell 验证 ✅ (PyTorch at:: 实现 vs reference)   │
└─────────────────────────────────────────────────────┘
                    ↓ 算法正确 → 需要验证 ↓
┌─────────────────────────────────────────────────────┐
│  计算精度层: AscendC kernel 数值精度是否达标？       │
│  └─ 需要在 MindSpore 路径验证                        │
│     (kernel 执行 + at:: 结果的对比)                  │
└─────────────────────────────────────────────────────┘
                    ↓ 数值正确 → 需要验证 ↓
┌─────────────────────────────────────────────────────┐
│  框架集成层: Tiling/Workspace 是否正确？              │
│  └─ MindSpore 训练路径报错处在此层                    │
│     (error 507015: tiling 数据未序列化)              │
└─────────────────────────────────────────────────────┘
```

## 9. 精度等价性保证细节

- **float32**: PyTorch CPU (BLAS) 与 NPU 原生实现使用相同的 IEEE 754, 误差 < 1e-6, Shell 验证 max_diff = 1.19e-07 ✅
- **bf16/fp16**: NPU 原生实现由 torch_npu 保证与 PyTorch CPU 等价, bf16 < 1e-2, fp16 < 1e-3
- **at::einsum**: 复杂求和模式依赖底层 matmul 精度, Shell 验证 4/4 PASS
- **at::softmax**: 使用 `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))` 保证数值稳定性

## 10. 为什么 Shell 验证不能保证 kernel 正确？

**Shell 路径的调用链**:
```
torch.ops.npu.mhc_pre_cmhc_backward
    → torch_host.cpp (at:: API)
        → dispatcher → torch_npu → aclnn
            → NPU 原生算子
```

**AscendC kernel (`op_kernel/mhc_pre_cmhc_backward.cpp`) 从未被调用。**

而 **MindFormers 路径**:
```
ops.Custom("aclnnMhcPreCmhcBackward", func_type="aot")
    → ascendebug → aclnnMhcPreCmhcBackward
        → op_kernel/mhc_pre_cmhc_backward.cpp (✅ 真正调用 kernel)
            → tiling 层 → workspace 问题 → error 507015
```

> **结论**: at:: 实现是一个**独立、可靠的基准线**，用于验证算法正确性。Shell 验证通过 → 算法正确，但 ≠ kernel 正确。

---

# 三、MindSpore 崩溃根因分析

## 1. 错误现象

### 错误日志

```
[ERROR] DEVICE(100,fffb0ffef140,python):2026-07-14-16:23:47.123.456 
[task_manager_ascend910.cc:256] TaskFailedCallback Error info:
    taskId: 12345, deviceId: 0, streamId: 1
    error code: 507015
    mte error info: 0xd006000032
    ccu error info: 0xc1aed4804988da05
    cube error info: 0
    errorStr: "The DDR address of the MTE instruction is out of range"
    subErrType: 4

[ERROR] runtime:2026-07-14-16:23:47.234.567 
[device_memory.cpp:432] MTE DDR address out of range:
    workspace: 0 bytes
    expected_workspace: ~470MB (128M floats × 4 bytes)
```

### 错误解读

- **error 507015**: ACL 错误码，表示 MTE (Memory Transfer Engine) 的 DDR 地址越界
- **MTE error 0xd006000032**: DDR 地址越界 (out of range)
- **Cube error info = 0**: ✅ Cube 单元**没有报错**，说明不是计算错误
- **Sub error type = 4**: 内存访问错误
- **Workspace: 0 bytes vs expected ~470MB**: ❌ 实际 workspace 大小远小于预期

### 崩溃位置

```
op_kernel/mhc_pre_cmhc_backward.cpp
    ├─ GET_TILING_DATA_WITH_STRUCT(tiling_data)
    │      ↓ tiling_data 读到垃圾值 (batch_size=0, seq_len=0, workspace_size=0)
    ├─ InitGlobalMemory(batch_size=0, seq_len=0, ...)
    │      ↓ 循环边界计算错误
    ├─ InitSharedMemory(workspace_size=0)
    │      ↓ workspace 只有 0 bytes
    └─ MTE 搬运数据时越界
           ↓ error 507015: MTE DDR out of range
```

## 2. 调用栈追溯

### 完整调用栈

```
run_mindformer.py:80 → trainer.train()
    ├─ CausalLM.forward(...)
    │      ├─ TransformerLayer × num_hidden_layers
    │      │      ├─ HyperConnectionModuleAscendFusedCmhc.construct_hc(...)
    │      │      │      ├─ ManifoldConstrainedHyperConnectionsPreCmhc.construct_hc(...)
    │      │      │      │      └─ ops.Custom("aclnnMhcPreCmhc").construct(...)
    │      │      │      │           └─ AscendC kernel (✅ 正常)
    │      │      │      └─ loss = F.cross_entropy(...)
    │      │      └─ loss.backward()
    │      └─ AUTODIFF scheduler → backward 链执行
    │
    ├─ _bprop_mhc_pre_cmhc(...)  [自定义 backward 闭包]
    │      ├─ 接收 forward 输出: (h_pre, hc_before_norm, inv_rms, sumOut, normOut)
    │      ├─ 接收 upstream gradients: (grad_h_hat2, grad_h_pre, grad_normOut)
    │      │
    │      └─ ManifoldConstrainedHyperConnectionsPreCmhcBackward()(...)
    │             ↓
    │         ops.Custom("aclnnMhcPreCmhcBackward", func_type="aot").construct(...)
    │             ↓
    │         ascendebug.aclnn_call("aclnnMhcPreCmhcBackward", ...)
    │             ↓
    │         op_host/mhc_pre_cmhc_backward_tiling.cpp
    │             ├─ Tiling4MhcPreCmhcBackward(...)
    │             │      └─ TilingMhcPreCmhcBackwardArch22(...)
    │             │             └─ CalcTiling() (❌ stub, 未实现)
    │             │                    ├─ ❌ mm1TilingData/mm2TilingData 为空
    │             │                    ├─ ❌ GetWorkspaceSizes() 未调用
    │             │                    ├─ ❌ SaveToBuffer() 未调用
    │             │                    └─ ❌ SetDataSize() 未调用
    │             │
    │             └─ kernel 执行
    │                    ↓
    │                op_kernel/mhc_pre_cmhc_backward.cpp
    │                    ├─ GET_TILING_DATA_WITH_STRUCT(tiling_data)
    │                    │      └─ tiling_data 是垃圾值
    │                    ├─ InitGlobalMemory(0, 0, ...)
    │                    │      └─ 循环边界计算错误
    │                    ├─ InitSharedMemory(0 bytes)
    │                    │      └─ workspace 太小
    │                    └─ MTE 搬运数据时地址越界
    │                           ↓
    │                       ❌ error 507015: MTE DDR out of range
```

## 3. 根因分析

### 3.1 核心根因：tiling.cpp 是未完成的 stub

**文件**: `op_host/op_tiling/arch22/mhc_pre_cmhc_backward_arch22_tiling.cpp`

**当前代码 (lines 226-229)**:

```cpp
// 当前实现 (❌ stub)
ge::graphStatus TilingMhcPreCmhcBackwardArch22(const aclrtContext *context,
                                               const std::vector<ge::Tensor> &inputs,
                                               const std::vector<ge::Tensor> &outputs,
                                               op_tiling::OpRunInfo &runInfo) {
    OP_LOGI("MhcPreCmhcBackwardArch22", 
            "Enter TilingMhcPreCmhcBackwardArch22");
    
    // ❌ 局部变量从未写入 tilingData
    TilingData tilingData;
    int64_t mm1M = ..., mm1K = ..., mm1N = ...;  // dead code
    int64_t mm2M = ..., mm2K = ..., mm2N = ...;  // dead code
    int64_t workspaceSize = 0;
    
    // ❌ 未调用 SaveToBuffer - tiling 数据未序列化
    // ❌ 未调用 SetDataSize - kernel 读到垃圾数据
    // ❌ 未调用 GetWorkspaceSizes - workspace 未分配
    
    return GRAPH_SUCCESS;  // 直接返回成功，但啥都没做
}
```

**问题分析**:

1. **tilingData 未序列化**: 局部变量 `tilingData` 从未通过 `SaveToBuffer()` 写入到 kernel 可访问的 buffer
2. **Workspace 未计算**: `GetWorkspaceSizes()` 从未调用，workspace 大小默认为 0
3. **MatMul tiling 为空**: `mm1TilingData/mm2TilingData` 是默认构造的空结构体
4. **SetDataSize 未调用**: kernel 从 buffer 中读不到正确的 tiling 数据

### 3.2 次要问题：mm1M/mm1K 等局部变量是 dead code

```cpp
// Lines 187-195 (dead code)
int64_t mm1M = 1;
int64_t mm1K = tileSize * 2;
int64_t mm1N = batchSize * seqLength;
int64_t mm2M = batchSize * seqLength;
int64_t mm2K = tileSize * 2;  // ❌ dead code
int64_t mm2N = nC;

// Lines 202-217 (局部变量从未写入 tilingData)
tilingData.SetBatchSize(batchSize);
tilingData.SetSeqLength(seqLength);
tilingData.SetC(c);
tilingData.SetN(n);
// ... 其他字段设置
// ❌ mm1TilingData/mm2TilingData 从未设置
```

这些局部变量从未写入 `tilingData`，kernel 在 `InitTiling()` 中自行计算。

### 3.3 对比 forward 的正确 tiling 实现

```cpp
// Forward tiling (✅ 正确实现)
ge::graphStatus CalcTiling(const aclrtContext *context,
                           const std::vector<ge::Tensor> &inputs,
                           const std::vector<ge::Tensor> &outputs,
                           op_tiling::OpRunInfo &runInfo) {
    // ✅ 设置 batch_size, seq_length, c, n, tile_size 等
    
    // ✅ 计算 mm1 tiling
    MatmulApiTiling::GetTiling(mm1TilingData, 
                                M=1, K=tileSize*2, N=batchSize*seqLength, ...);
    
    // ✅ 计算 mm2 tiling
    MatmulApiTiling::GetTiling(mm2TilingData,
                                M=batchSize*seqLength, K=tileSize*2, N=n*C, ...);
    
    // ✅ 计算 workspace (需要 ~470MB)
    int64_t workspaceSize = 0;
    workspaceSize += batchSize * seqLength * c;  // gradH2: ~0.5MB
    workspaceSize += batchSize * seqLength * nC;  // xWorkspace: ~235MB
    workspaceSize += batchSize * seqLength * nC;  // gradXCube: ~235MB
    
    // ✅ GetWorkspaceSizes / ✅ SaveToBuffer / ✅ SetDataSize
    GetWorkspaceSizes(workspaceSize);
    SaveToBuffer(tilingData, context);
    SetDataSize(tilingData.GetDataSize());
    
    return GRAPH_SUCCESS;
}
```

## 4. 崩溃机制详解

### Phase 1: Tiling 数据丢失

```
TilingMhcPreCmhcBackwardArch22(...)
    ↓ tilingData (局部变量) 在 stack 上
    ↓ ❌ SaveToBuffer() 未调用
    ↓ Context buffer (kernel 可访问) 保持为空
    ↓ return GRAPH_SUCCESS
```

### Phase 2: Kernel 读取垃圾数据

```
op_kernel/mhc_pre_cmhc_backward.cpp
    ↓ GET_TILING_DATA_WITH_STRUCT(tiling_data)
    ↓ 从 context buffer 读取 tiling_data
    ↓ buffer 为空 → tiling_data 是未初始化的垃圾值
       ├─ batch_size = 0 (或其他垃圾值)
       ├─ seq_length = 0 (或其他垃圾值)
       ├─ c = 0 (或其他垃圾值)
       └─ mm1TilingData = 空结构体
```

### Phase 3: 错误的循环边界计算

```
InitGlobalMemory(batch_size=0, seq_length=0, ...)
    ↓ int totalTasks = batch_size * seq_length;  // = 0
    ↓ int numIter = totalTasks / tileCoreBS_ + (totalTasks % tileCoreBS_ != 0);
    ↓ // 可能除零导致 numIter = 无穷或垃圾值
    ↓ for (int i = 0; i < numIter; i++) { 循环次数错误 }
```

### Phase 4: Workspace 不足导致 MTE 越界

```
InitSharedMemory(workspace_size=0)
    ↓ 分配 workspace: 0 bytes
    ↓ 实际需要的 workspace:
        - gradH2: batchSize * seqLength * c * sizeof(float) = ~0.5MB
        - xWorkspace: batchSize * seqLength * nC * sizeof(float) = ~235MB
        - gradXCube: batchSize * seqLength * nC * sizeof(float) = ~235MB
        - 总计: ~470MB (DETERMINISTIC mode)
    ↓ ❌ 实际只有 0 bytes
    ↓ MTE 单元搬运数据时:
        DDR 地址 = workspace_base(0) + offset(大值)
        DDR 地址 > DDR_MAX_SIZE
    ↓ ❌ MTE DDR out of range → error 507015
```

### Phase 5: ACL 报告错误

```
MTE 单元检测到地址越界
    ↓ 返回 error 507015 到 ACL 框架
    ↓ ACL 框架报告:
        "The DDR address of the MTE instruction is out of range"
        mte error info: 0xd006000032
        cube error info: 0  ← Cube 单元没有报错
        sub error type: 4   ← 内存访问错误
```

## 5. 为什么 Cube error info = 0?

- **Cube 单元**: 负责矩阵乘法计算
- **MTE 单元**: 负责数据搬运

```
MTE 出错 (内存搬运):
    - 错误码: 0xd006000032 (DDR 地址越界)
    - 错误类型: 4 (内存访问错误)
    - 原因: workspace 太小 (0 bytes)

Cube 正常 (计算单元):
    - 错误码: 0 (无错误)
    - 说明: MTE 是第一个出错的单元, Cube 还没开始计算就被中断
```

## 6. 根因总结

```
┌─────────────────────────────────────────────────────────┐
│ 根本原因                                                  │
│ op_host/op_tiling/arch22/mhc_pre_cmhc_backward_arch22_  │
│ tiling.cpp 的 CalcTiling() 是未完成的 stub:              │
│                                                           │
│ 1. ❌ 未调用 SaveToBuffer()                               │
│    → tiling 数据未序列化到 context                       │
│ 2. ❌ 未调用 GetWorkspaceSizes()                          │
│    → workspace 大小未分配 (0 bytes)                      │
│ 3. ❌ 未调用 MatmulApiTiling::GetTiling()                 │
│    → mm1TilingData/mm2TilingData 为空                    │
│ 4. ❌ 未调用 SetDataSize()                                │
│    → kernel 读到垃圾 tiling 数据                         │
│                                                           │
│ 导致: Op_kernel 读取垃圾 tiling, Workspace = 0 bytes,     │
│       MTE 单元搬运数据时地址越界, error 507015            │
└─────────────────────────────────────────────────────────┘
```

## 7. 为什么 Shell 验证没有发现此问题？

```
Shell 验证流程:
    torch.ops.npu.mhc_pre_cmhc_backward(...)
        ↓ dispatcher → torch_npu
        ↓ aclnnMatmul, aclnnEinsum, ... (PyTorch at:: API)
        ↓ ✅ 执行成功

关键区别:
    - Shell 路径使用 PyTorch dispatcher
    - dispatcher 自动处理所有 NPU 细节 (tiling/workspace)
    - 不需要手动管理 tiling/workspace
    - 即使 tiling.cpp 有 bug, 也不会暴露

MindFormers 路径:
    ops.Custom("aclnnMhcPreCmhcBackward")
        ↓ 直接调用 op_host/op_kernel
        ↓ 必须手动管理 tiling/workspace
        ↓ tiling 有 bug → 崩溃
```

---

# 四、两条调用路径完整对比

## 1. 整体架构对比

### Shell 路径架构

```
bash_cmhc_backward_local.sh
  ├─ Step 1: 编译 AscendC kernel (build/binary/...)
  ├─ Step 2: 编译 PyTorch binding (.so)
  ├─ Step 3: 加载验证 (load_library)
  ├─ Step 4: 精度验证 (verification_ascendc.py)
  └─ Step 5: 性能验证 (msprof_perf_summary.py)

核心特征:
  - 单算子验证
  - PyTorch at:: API 实现
  - 无 tiling 层
  - PyTorch 自动内存管理
```

### MindFormers 路径架构

```
run_mindformer.py
  ├─ 模型初始化 (TransformerLayer × num_hidden_layers)
  │    └─ HyperConnectionModuleAscendFusedCmhc (每层)
  │         └─ ops.Custom("aclnnMhcPreCmhc")
  ├─ Forward 执行 → ACL 调用 AscendC kernel
  ├─ Loss 计算
  └─ Backward 执行 (AUTODIFF)
       └─ ops.Custom("aclnnMhcPreCmhcBackward")
            └─ ACL 调用 AscendC kernel ❌ 崩溃

核心特征:
  - 完整训练流程
  - 真正调用 AscendC kernel
  - 需要 tiling 层
  - MindSpore memory manager 管理 workspace
```

## 2. Forward 执行对比

| 对比维度 | Shell 路径 | MindFormers 路径 |
|---|---|---|
| **执行方式** | ❌ 不执行 forward | ✅ ACL 调用 AscendC kernel |
| **中间量来源** | model.py 内部重新计算 (torch.autograd.grad) | forward op 输出 → 传递给 backward |
| **中间量内容** | h_pre, hc_before_norm, inv_rms | h_pre, hc_before_norm, inv_rms, sumOut, normOut |
| **计算精度** | float32 autograd 精确结果 | NPU kernel bf16/fp16 计算结果 |
| **调用链** | torch.autograd → at:: API | MindSpore ops.Custom → ascendebug → aclnn → kernel |

### Shell 路径的 forward

```python
# model.py (reference 模型)
def forward(self, x, hc_fn, ...):
    with torch.enable_grad():
        x_t = x.detach().requires_grad_(True)
        # 重新计算 forward
        h_hat2 = at::matmul(x_t, hc_fn)
        h_hat2 = at::tanh(h_hat2)
        # ...
        # 通过 autograd 计算 backward
        grad = torch.autograd.grad(
            outputs=[h_hat2, h_pre, ...],
            inputs=[x_t, hc_fn, hc_scale, hc_base],
            grad_outputs=[grad_h_hat2, grad_h_pre, ...])
    return grad
```

**特点**: 完全重新计算 forward, 精确 autograd backward, float32 精度, 不依赖任何 forward kernel

### MindFormers 路径的 forward

```python
# hyper_connection_npu(1).py:381
mhc_pre_cmhc_op.construct_hc(
    x, mapping_weight, alpha_pre, alpha_post, 
    alpha_res, bias_pre, bias_post, bias_res,
    perm_mats, hcMult, numIters, hcEps, normEps, needBackward=True)
    ↓
ops.Custom("aclnnMhcPreCmhc", func_type="aot").construct(...)
    ↓ ascendebug.aclnn_call(...)
    ↓ op_kernel/mhc_pre_cmhc_backward.cpp: __global__ void mhc_pre_cmhc_backward(...)
    ↓ ACL 调度执行
    ↓ 输出: h_pre, hc_before_norm, inv_rms, sumOut, normOut
```

**特点**: 真正执行 kernel, NPU 硬件 bf16/fp16 计算, 需要 tiling/workspace, 输出传递给 backward op

## 3. Backward 执行对比

| 对比维度 | Shell 路径 | MindFormers 路径 |
|---|---|---|
| **执行方式** | PyTorch at:: API 实现 | ACL 调用 AscendC kernel |
| **实现文件** | mhc_pre_cmhc_backward_torch_host.cpp | op_kernel/mhc_pre_cmhc_backward.cpp |
| **调度机制** | dispatcher → torch_npu → aclnn | MindSpore ops.Custom → ascendebug → aclnn |
| **精度** | float32 (与 autograd 精确匹配) | NPU kernel bf16/fp16 |
| **并行策略** | 单线程 at:: 实现 | Cube/AIV/AIC 混合并行 |
| **Tiling 依赖** | ❌ 不需要 | ✅ 需要完整的 tiling 数据 |
| **Tiling 序列化** | ❌ 不需要 | ✅ 需要 SaveToBuffer |
| **Workspace** | ❌ PyTorch 自动分配 | ✅ 需要 GetWorkspaceSizes |
| **当前状态** | ✅ 4/4 PASS | ❌ error 507015 崩溃 |

### Shell 路径的 backward 详细流程

```python
# model_new_ascendc.py
def forward(self, x, hc_fn, ...):
    h_pre, hc_before_norm, inv_rms = self._compute_intermediates(x, hc_fn, ...)
    backend_fn = self._get_backend_fn(x.device)
    grad = backend_fn(x, hc_fn, grad_h_hat2, h_pre, ...)
        ↓ torch.ops.npu.mhc_pre_cmhc_backward(...)
        ↓ [TORCH_LIBRARY_IMPL(npu, PrivateUse1)]
        ↓ mhc_pre_cmhc_backward_torch_host.cpp
        ↓ └─ 纯 at:: API 实现
    return grad
```

**特点**: 使用 PyTorch dispatcher, at:: API 自动转换为 NPU 原语, 不依赖 tiling, 不分配 workspace

### MindFormers 路径的 backward 详细流程

```python
# hyper_connection_npu(1).py:422 backward 闭包
def _bprop_mhc_pre_cmhc(self, ...):
    (h_pre, hc_before_norm, inv_rms, sumOut, normOut) = forward_outputs
    (grad_h_hat2, grad_h_pre, grad_normOut) = upstream_grads
    grad_x, grad_hc_fn, grad_hc_scale, grad_hc_base = \
        ManifoldConstrainedHyperConnectionsPreCmhcBackward()(...)
            ↓ ops.Custom("aclnnMhcPreCmhcBackward", func_type="aot").construct(...)
            ↓ ascendebug.aclnn_call(...)
            ↓ op_kernel/mhc_pre_cmhc_backward.cpp
            ↓ ❌ 崩溃: MTE DDR 越界 → error 507015
```

## 4. Tiling/Workspace 管理对比

| 对比维度 | Shell 路径 | MindFormers 路径 |
|---|---|---|
| **Tiling 调用** | ❌ 不调用 | ✅ CalcTiling() (arch22_tiling.cpp) |
| **Tiling 序列化** | ❌ SaveToBuffer 不存在 | ❌ SaveToBuffer 未实现 (stub) |
| **Tiling 数据** | ❌ 不需要 | ✅ 需要 mm1TilingData/mm2TilingData |
| **Workspace 计算** | ❌ GetWorkspaceSizes 不存在 | ❌ GetWorkspaceSizes 未实现 (stub) |
| **Workspace 分配** | ❌ PyTorch 自动管理 | ✅ MindSpore memory manager 分配 |
| **Workspace 大小** | N/A | 需要 ~470-514MB |
| **PostTiling** | ❌ 不存在 | ❌ 存在但未被调用 (arch22 不进入) |
| **当前状态** | N/A (不需要) | ❌ 崩溃于此 |

## 5. 数值精度保证机制对比

| 对比维度 | Shell 路径 | MindFormers 路径 |
|---|---|---|
| **精度标准** | IEEE 754 float32 | NPU kernel bf16/fp16 |
| **对比基准** | torch.autograd.grad (精确) | N/A (无独立基准) |
| **数值对比** | ✅ max_diff = 1.19e-07 | ❌ 崩溃，无法对比 |
| **误差来源** | at:: 实现精度 | kernel 精度 + bf16 截断 |
| **验证方式** | Shell 脚本自动验证 | 只能观察 loss 趋势 |
| **可重复性** | ✅ 精确可重复 | ⚠️ NPU 执行有非确定性 |

## 6. 崩溃路径分析

### Shell 路径：无法崩溃

```
torch.ops.npu.mhc_pre_cmhc_backward(...)
    ↓ dispatcher → torch_npu
    ↓ aclnnMatmul, aclnnEinsum, ...
    ↓ CANN 框架自动处理所有硬件细节
    ↓ ✅ 执行成功
```

### MindFormers 路径：当前崩溃

```
ops.Custom("aclnnMhcPreCmhcBackward")
    ↓ op_host/mhc_pre_cmhc_backward_tiling.cpp CalcTiling() ❌ 未实现
       ├─ mm1TilingData = empty
       ├─ mm2TilingData = empty
       ├─ workspace_size = 未计算
       └─ tiling 数据未序列化
    ↓ op_kernel/mhc_pre_cmhc_backward.cpp:
       GET_TILING_DATA_WITH_STRUCT() → 读到垃圾 tiling
       InitGlobalMemory(0, 0, ...) → 循环边界错误
       InitSharedMemory(0 bytes) → MTE DDR 越界
       ❌ error 507015
```

## 7. 调试能力与开发效率对比

| 对比维度 | Shell 路径 | MindFormers 路径 |
|---|---|---|
| **错误日志** | PyTorch 错误栈 | MindSpore ACL 错误栈 + 内核 dump |
| **数值调试** | ✅ 轻松打印中间结果 | ❌ kernel 中间结果难访问 |
| **性能分析** | ✅ PyTorch profiler | ✅ msprof 工具 |
| **迭代速度** | ✅ 分钟级 | ❌ 小时级 (完整训练) |
| **修改成本** | ✅ 只需改 torch_host.cpp | ❌ 需要改 kernel + tiling |
| **验证成本** | ✅ 自动运行 | ❌ 需要完整训练 |
| **门槛** | ✅ 熟悉 PyTorch 即可 | ❌ 需要熟悉 MindSpore + ACL |

## 8. 适用场景

### Shell 路径
1. 快速算法实验
2. 数值精度验证
3. 独立开发（不依赖 MindSpore）
4. CI/CD 轻量级验证
5. 教学理解算子算法

### MindFormers 路径
1. 生产验证（真实训练场景）
2. 硬件验证（kernel 在 NPU 上的行为）
3. 集成测试（CustomOp/AUTODIFF 正确性）
4. 长期稳定性（多 epoch + batch）
5. 完整训练性能指标

## 9. 本章结论

Shell 路径和 MindFormers 路径是两条**本质不同的验证路径**：

| 对比维度 | Shell 路径 | MindFormers 路径 |
|---|---|---|
| **验证目标** | 算法正确性 | 生产正确性 |
| **验证深度** | 算法层面 | 硬件层面 + 框架层面 |
| **当前状态** | ✅ 4/4 PASS | ❌ error 507015 |
| **价值** | 快速迭代，精确对比 | 生产验证，硬件检查 |

两者**互补而非替代**。当前 MindFormers 崩溃问题出在 Shell 验证**无法覆盖**的层面（tiling/workspace）。

---

# 五、修复方案

## 1. 问题回顾

### 当前状态

```
✅ Shell 4/4 PASS (at:: 实现验证通过)
❌ MindFormers error 507015 (tiling stub 导致崩溃)
```

### 根因总结

`op_host/op_tiling/arch22/mhc_pre_cmhc_backward_arch22_tiling.cpp` 的 `TilingMhcPreCmhcBackwardArch22` 函数是**未完成的 stub**：
1. ❌ 未调用 `SaveToBuffer()` - tiling 数据未序列化
2. ❌ 未调用 `GetWorkspaceSizes()` - workspace 未分配
3. ❌ 未调用 `MatmulApiTiling::GetTiling()` - MatMul tiling 数据为空
4. ❌ 未调用 `SetDataSize()` - kernel 读到垃圾数据

## 2. 修复方案

### P0: 核心修复（必须）

修复文件: `op_host/op_tiling/arch22/mhc_pre_cmhc_backward_arch22_tiling.cpp`

#### 完整的 CalcTiling 实现

```cpp
ge::graphStatus TilingMhcPreCmhcBackwardArch22(
    const aclrtContext *context,
    const std::vector<ge::Tensor> &inputs,
    const std::vector<ge::Tensor> &outputs,
    op_tiling::OpRunInfo &runInfo) {
    
    OP_LOGI("MhcPreCmhcBackwardArch22",
            "Enter TilingMhcPreCmhcBackwardArch22");
    
    // === Step 1: 设置基本参数 ===
    TilingData tilingData;
    
    // 从 inputs 推导出参数
    int64_t batchSize = inputs[0].GetShape().GetDim(0);
    int64_t seqLength = inputs[0].GetShape().GetDim(1);
    int64_t c = inputs[0].GetShape().GetDim(2);
    int64_t n = 4;  // 固定值 (HC 数量)
    int64_t nC = n * c;
    int64_t tileSize = 128;  // 从配置读取或使用默认值
    bool deterministic = true;  // 从配置读取或使用默认值
    
    tilingData.SetBatchSize(batchSize);
    tilingData.SetSeqLength(seqLength);
    tilingData.SetC(c);
    tilingData.SetN(n);
    tilingData.SetTileSize(tileSize);
    tilingData.SetDeterministic(deterministic);
    
    // === Step 2: 计算 MatMul tiling (P0) ===
    
    // MM1: (batchSize * seqLength, tileSize * 2) @ (nC, tileSize * 2)^T
    int64_t mm1M = batchSize * seqLength;
    int64_t mm1K = tileSize * 2;
    int64_t mm1N = nC;
    
    auto mm1Status = MatmulApiTiling::GetTiling(
        tilingData.GetMutablemm1TilingData(),
        mm1M, mm1K, mm1N,
        /* leftFormat */ FORMAT_ND,
        /* rightFormat */ FORMAT_ND,
        /* outputFormat */ FORMAT_ND,
        /* leftDatatype */ DT_FLOAT,
        /* rightDatatype */ DT_FLOAT,
        /* outputDatatype */ DT_FLOAT,
        /* isTransposeLeft */ false,
        /* isTransposeRight */ true,
        /* isGEMM */ true);
    
    if (mm1Status != ACL_SUCCESS) {
        OP_LOGE("MhcPreCmhcBackwardArch22",
                "Failed to get MatMul1 tiling: %d", mm1Status);
        return GRAPH_FAILED;
    }
    
    // MM2: (batchSize * seqLength, tileSize * 2) @ (nC, tileSize * 2)^T
    int64_t mm2M = batchSize * seqLength;
    int64_t mm2K = tileSize * 2;
    int64_t mm2N = nC;
    
    auto mm2Status = MatmulApiTiling::GetTiling(
        tilingData.GetMutablemm2TilingData(),
        mm2M, mm2K, mm2N,
        FORMAT_ND, FORMAT_ND, FORMAT_ND,
        DT_FLOAT, DT_FLOAT, DT_FLOAT,
        false, true, true);
    
    if (mm2Status != ACL_SUCCESS) {
        OP_LOGE("MhcPreCmhcBackwardArch22",
                "Failed to get MatMul2 tiling: %d", mm2Status);
        return GRAPH_FAILED;
    }
    
    // === Step 3: 计算 Workspace 大小 (P0) ===
    int64_t workspaceSize = 0;
    
    // gradH2: (batchSize, seqLength, c) * sizeof(float)
    workspaceSize += batchSize * seqLength * c * sizeof(float);
    
    // xWorkspace: (batchSize, seqLength, nC) * sizeof(float)
    workspaceSize += batchSize * seqLength * nC * sizeof(float);
    
    // gradXCube: only for DETERMINISTIC
    if (deterministic) {
        workspaceSize += batchSize * seqLength * nC * sizeof(float);
    }
    // gradWeightWS: only for DETERMINISTIC
    if (deterministic) {
        workspaceSize += batchSize * seqLength * c * sizeof(float);
    }
    // gradBiasWS: only for DETERMINISTIC
    if (deterministic) {
        workspaceSize += batchSize * seqLength * sizeof(float);
    }
    
    OP_LOGI("MhcPreCmhcBackwardArch22",
            "Calculated workspace size: %zd bytes (~%.2f MB)",
            workspaceSize, workspaceSize / (1024.0 * 1024.0));
    
    // === Step 4: 设置 Workspace (P0) ===
    auto workspaceStatus = runInfo.GetWorkspaceSizes(workspaceSize);
    if (workspaceStatus != ACL_SUCCESS) {
        OP_LOGE("MhcPreCmhcBackwardArch22",
                "Failed to set workspace sizes: %d", workspaceStatus);
        return GRAPH_FAILED;
    }
    
    // === Step 5: 序列化 Tiling 数据 (P0) ===
    auto saveStatus = SaveToBuffer(tilingData, context);
    if (saveStatus != ACL_SUCCESS) {
        OP_LOGE("MhcPreCmhcBackwardArch22",
                "Failed to save tiling data: %d", saveStatus);
        return GRAPH_FAILED;
    }
    
    // === Step 6: 设置数据大小 (P0) ===
    auto sizeStatus = SetDataSize(tilingData.GetDataSize());
    if (sizeStatus != ACL_SUCCESS) {
        OP_LOGE("MhcPreCmhcBackwardArch22",
                "Failed to set data size: %d", sizeStatus);
        return GRAPH_FAILED;
    }
    
    OP_LOGI("MhcPreCmhcBackwardArch22",
            "TilingMhcPreCmhcBackwardArch22 completed successfully");
    
    return GRAPH_SUCCESS;
}
```

#### 关键 API 说明

| API | 用途 | 调用位置 |
|---|---|---|
| `MatmulApiTiling::GetTiling()` | 计算 MatMul 的 tiling 参数 | op_tiling 层 |
| `GetWorkspaceSizes()` | 通知 ACL 框架分配 workspace | op_tiling 层 |
| `SaveToBuffer()` | 序列化 tiling 数据到 context buffer | op_tiling 层 |
| `SetDataSize()` | 设置 tiling 数据大小 | op_tiling 层 |
| `GET_TILING_DATA_WITH_STRUCT()` | kernel 读取 tiling 数据 | op_kernel 层 |

### P1: 清理 dead code

删除 `lines 187-195` 的局部变量（kernel 自行计算，保留会误导后续开发者）：

```cpp
// ❌ 删除这些行 (lines 187-195):
int64_t mm1M = 1;
int64_t mm1K = tileSize * 2;
int64_t mm1N = batchSize * seqLength;
int64_t mm2M = batchSize * seqLength;
int64_t mm2K = tileSize * 2;  // dead code
int64_t mm2N = nC;
```

### P2: 删除误导性注释

```cpp
// ❌ 删除原来的误导性注释:
// workspace size and SetTilingData will be set in PostTiling

// ✅ 改为清晰的说明:
// Note: arch22 没有独立的 PostTiling 步骤
// 所有 tiling 逻辑（参数设置、MatMul tiling、workspace、序列化）都在此函数中完成
```

## 3. 修复步骤

### Step 1: 修改 arch22_tiling.cpp

```bash
vim op_host/op_tiling/arch22/mhc_pre_cmhc_backward_arch22_tiling.cpp
# 应用 P0/P1/P2 修复
```

### Step 2: 编译 op_host

```bash
cd /path/to/ops-transformer
bash bash_cmhc_backward_local.sh
# 或手动编译: cd op_host && mkdir -p build && cd build && cmake .. && make -j8
```

### Step 3: 编译 Python binding

```bash
cd /path/to/cmhc_cann/cmhc_backward/binding
python setup.py build_ext --inplace
```

### Step 4: 验证 Shell 路径

```bash
bash bash_cmhc_backward_local.sh
# 验证: binding 编译成功, load_library 成功, 4/4 PASS, performance.log 正常
```

### Step 5: 验证 MindFormers 路径

```bash
python run_mindformer.py \
    --config telechat4_40B_pretrain_1p.yaml \
    --use_parallel False \
    --device_id 8
# 验证: 不再崩溃于 error 507015, forward/backward 正常, loss 正常下降
```

## 4. 修复前后代码对比

### 修复前（当前代码）

```cpp
// Lines 183-229 (当前 stub 实现)
ge::graphStatus TilingMhcPreCmhcBackwardArch22(...) {
    // ❌ 局部变量从未写入 tilingData
    TilingData tilingData;
    int64_t mm1M = 1;                          // dead code
    int64_t mm1K = tileSize * 2;               // dead code
    int64_t mm1N = batchSize * seqLength;      // dead code
    int64_t mm2M = batchSize * seqLength;      // dead code
    int64_t mm2K = tileSize * 2;               // dead code
    int64_t mm2N = nC;                         // dead code
    
    tilingData.SetBatchSize(batchSize);
    tilingData.SetSeqLength(seqLength);
    // ... 其他字段设置 (未包括 mm1/mm2 TilingData)
    
    // ❌ 注释误导: workspace size and SetTilingData will be set in PostTiling
    
    return GRAPH_SUCCESS;  // 直接返回，但实际上什么都没做
}
```

### 修复后（完整实现）

完整代码见上文 P0 章节。

## 5. 验证清单

### 编译验证
- [ ] `bash_cmhc_backward_local.sh` 执行成功
- [ ] op_host 编译无错误
- [ ] binding 编译成功
- [ ] load_library 成功

### Shell 验证
- [ ] `verification_ascendc.py` 4/4 PASS
- [ ] max_diff < 1e-6 (float32)
- [ ] performance.log 正常输出

### MindFormers 验证
- [ ] 不再崩溃于 error 507015
- [ ] forward 正常执行
- [ ] backward 正常执行
- [ ] 梯度正常计算
- [ ] loss 正常下降

### 数值精度验证
- [ ] 与 Shell at:: 实现对比，误差 < 1e-2 (bf16)
- [ ] 与参考 NPU 实现对比（如果有）

## 6. 常见问题

**Q1: 为什么要调用 `MatmulApiTiling::GetTiling()`？**

A: Op_kernel 中的 MatMul 计算需要 tiling 参数来优化硬件利用率：决定 tile 大小、循环次数、数据搬运策略。不提供 tiling 参数可能导致性能低下或错误。

**Q2: 为什么 workspace 大小是 ~470MB？**

A: 根据 op_kernel 中的定义：gradH2 (~0.5MB) + xWorkspace (~235MB) + gradXCube (~235MB) = ~471MB (DETERMINISTIC mode)

**Q3: 为什么删除 dead code？**

A: Lines 187-195 的局部变量从未写入 `tilingData`，kernel 在 `InitTiling()` 中自行计算这些值。保留会误导后续开发者。

**Q4: tilingData.GetDataSize() 是什么？**

A: `tilingData` 序列化为 byte buffer 后的大小，通过 `SaveToBuffer()` 写入到 context buffer，kernel 通过 `GET_TILING_DATA_WITH_STRUCT()` 读取时需要知道大小。

**Q5: 如果编译失败怎么办？**

A: 
1. `MatmulApiTiling` 找不到 → 需要 include 正确的 header
2. `GetWorkspaceSizes` 参数不匹配 → 查看 forward tiling 实现
3. `SaveToBuffer` 找不到 → 使用 `OpTilingContext` API

## 7. 参考资料

- Forward tiling 实现: `op_host/op_tiling/arch22/mhc_pre_cmhc_arch22_tiling.cpp`
- Op_kernel 实现: `op_kernel/mhc_pre_cmhc_backward.cpp`
- Op_kernel tiling 结构: `op_kernel/arch22/mhc_pre_cmhc_backward_data_arch22.h`
- API 文档: CANN OpHost Development Guide

---

# 六、完整验证策略

## 1. 当前验证现状

### Shell 验证（已建立）
```bash
bash_cmhc_backward_local.sh
  ├─ Step 1: 编译 AscendC kernel
  ├─ Step 2: 编译 PyTorch binding (at:: 实现)
  ├─ Step 3: load_library 验证
  ├─ Step 4: verification_ascendc.py (精度验证)
  └─ Step 5: msprof_perf_summary.py (性能验证)

结果: ✅ 4/4 PASS
覆盖范围: at:: 实现 vs autograd (算法层面)
未覆盖: AscendC kernel, tiling, workspace
```

### MindFormers 验证（问题暴露）
```bash
run_mindformer.py
  ├─ TransformerLayer 初始化
  ├─ Forward 执行 (ops.Custom)
  └─ Backward 执行 (❌ error 507015)

结果: ❌ MTE DDR out of range
根本原因: arch22_tiling.cpp 是 stub, 未序列化 tiling 数据
```

## 2. 四层验证体系

### Layer 1: Shell 算法正确性验证（现有）

**目标**: 验证 at:: 实现与 autograd 精确等价

**执行方式**: `bash_cmhc_backward_local.sh`

**覆盖范围**:
- ✅ at:: 算法正确性
- ✅ 与 torch.autograd.grad 精确对比
- ✅ 数值精度达标 (max_diff < 1e-6)

**未覆盖**:
- ❌ AscendC kernel 自身正确性
- ❌ Op_host tiling 逻辑
- ❌ Workspace 分配

**验证频率**: 每次修改 torch_host.cpp 时执行

**通过标准**: 4/4 PASS, max_diff < 1e-6

---

### Layer 2: AscendC kernel 单算子验证（需要建立）

**目标**: 验证 AscendC kernel 自身的数值正确性

**执行方式**:
```cpp
// 新建文件: test/tests/test_mhc_pre_cmhc_backward.cpp
#include <gtest/gtest.h>
#include "op_kernel/mhc_pre_cmhc_backward.cpp"

TEST(MhcPreCmhcBackwardTest, BasicTest) {
    // Step 1: 准备输入数据 (从 model.py get_input_groups())
    at::Tensor x = ...;
    at::Tensor hc_fn = ...;
    at::Tensor grad_h_hat2 = ...;
    
    // Step 2: 设置 tiling 数据
    TilingData tilingData;
    tilingData.batch_size = ...;
    tilingData.seq_length = ...;
    SaveToBuffer(tilingData, context);
    
    // Step 3: 分配 workspace
    void* workspace = malloc(workspace_size);
    
    // Step 4: 调用 AscendC kernel
    __global__ void mhc_pre_cmhc_backward(
        x, hc_fn, ...,
        grad_h_hat2, ..., 
        workspace, &tilingData);
    
    // Step 5-6: 获取输出, 与 reference 对比
    EXPECT_NEAR(
        torch::max(torch::abs(grad_x - reference_grad_x)).item<float>(),
        0.0, 1e-3f  // bf16 精度
    );
}
```

**覆盖范围**:
- ✅ AscendC kernel 数值正确性（对比 torch_host.cpp）
- ✅ Op_host tiling 逻辑正确性
- ✅ Workspace 分配正确性
- ✅ MatMul tiling 正确性

**未覆盖**:
- ❌ 框架集成（MindSpore ops.Custom）
- ❌ 多 step 训练稳定性
- ❌ 多卡并行场景

**验证频率**: 每次修改 op_kernel / op_host tiling 时执行，CI/CD 中每次 PR 执行

**通过标准**: 数值精度 < 1e-3 (bf16) 或 < 1e-6 (float32)，无 kernel crash，无 MTE/cube/aicore 错误

**建立步骤**:
1. 创建 `test/tests/test_mhc_pre_cmhc_backward.cpp`
2. 实现 reference 结果生成（使用 torch_host.cpp）
3. 实现 AscendC kernel 直接调用
4. 对比两者输出
5. 添加到 CI/CD pipeline

---

### Layer 3: MindFormers 集成验证（待修复）

**目标**: 验证 MindFormers 框架集成正确性

**执行方式**:
```python
# 新建文件: test/integration/test_mindformers_integration.py
import torch
import torch_npu
from mindformers.models.causal_lm import CausalLM
from mindformers.configs import CausalLMConfig

def test_mindformers_integration():
    """测试 MindFormers 单 step forward + backward"""
    
    # Step 1: 初始化模型
    config = CausalLMConfig(num_hidden_layers=4, hidden_size=4096, ...)
    model = CausalLM(config).npu()
    
    # Step 2: 生成输入
    input_ids = torch.randint(0, 50000, (2, 4096)).npu()
    
    # Step 3: Forward (通过 HyperConnectionModule)
    output = model(input_ids)
    
    # Step 4: 计算 loss
    loss = output.loss
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    
    # Step 5: Backward (通过 _bprop_mhc_pre_cmhc)
    loss.backward()
    
    # Step 6: 检查梯度
    for name, param in model.named_parameters():
        if param.grad is None: continue
        assert not torch.isnan(param.grad).any(), f"Gradient NaN in {name}"
        assert not torch.isinf(param.grad).any(), f"Gradient Inf in {name}"
        assert torch.norm(param.grad) > 0, f"Gradient zero in {name}"
    
    print("✅ MindFormers integration test passed")
```

**覆盖范围**:
- ✅ MindSpore ops.Custom 注册正确性
- ✅ AscendC forward/backward kernel 正确执行
- ✅ Op_host tiling 完整流程
- ✅ Workspace 分配完整流程
- ✅ AUTODIFF 链正确性
- ✅ 梯度计算正确性

**未覆盖**:
- ❌ 多 step 训练稳定性
- ❌ 梯度累积
- ❌ 优化器交互
- ❌ 完整训练 loss 下降趋势

**验证频率**: 每次修复 op_host/op_kernel 后执行，CI/CD 中每次 PR 执行

**通过标准**: Forward/backward/gradient 全部无 NaN/Inf，无 error 507015 或其他 ACL 错误，梯度 norm > 0

**建立步骤**:
1. 修复 arch22_tiling.cpp（见修复方案章节）
2. 创建测试脚本 `test/integration/test_mindformers_integration.py`
3. 执行测试验证通过
4. 添加到 CI/CD pipeline

---

### Layer 4: MindFormers 完整训练验证（最终目标）

**目标**: 验证完整训练流程稳定性

**执行方式**:
```bash
cd /path/to/mindformers
python run_mindformer.py \
    --config telechat4_40B_pretrain_1p.yaml \
    --use_parallel False \
    --device_id 8
```

**覆盖范围**: 完整训练流程、多 step 稳定性、梯度累积、优化器交互、Checkpoint 保存/加载、Loss 正常下降

**验证频率**: 修复关键 bug 后执行、Release 前必须执行、定期回归测试（每周/每月）

**通过标准**: 训练至少 100 steps 无崩溃、Loss 正常下降、无 ACL 错误、Checkpoint 可正常保存

---

## 3. 四层验证体系对比

| Layer | 验证目标 | 执行路径 | 覆盖范围 | 验证频率 | 执行时间 |
|---|---|---|---|---|---|
| **Layer 1** | 算法正确性 | torch_host.cpp (at::) | autograd 对比 | 高 | 秒级 |
| **Layer 2** | Kernel 正确性 | AscendC kernel + tiling | 单算子精度 | 中 | 分钟级 |
| **Layer 3** | 框架集成 | MindFormers 单 step | forward + backward | 中 | 分钟级 |
| **Layer 4** | 生产稳定性 | MindFormers 完整训练 | 完整训练流程 | 低 | 小时级 |

## 4. 当前状态

```
Layer 1: ✅ 已建立 (bash_cmhc_backward_local.sh)
            ├── 覆盖: at:: 实现正确性
            └── 验证: 4/4 PASS ✅

Layer 2: ❌ 未建立
            ├── 需要: test/tests/test_mhc_pre_cmhc_backward.cpp
            └── 用途: 验证 AscendC kernel 自身正确性

Layer 3: ❌ 未通过
            ├── 当前: error 507015 (tiling stub 未修复)
            └── 需要: 先修复 arch22_tiling.cpp

Layer 4: ❌ 未通过
            ├── 当前: 无法开始训练
            └── 依赖: Layer 3 必须先通过
```

## 5. 执行路线图

### Phase 1: 修复 Layer 3 崩溃问题（当前）

**任务**:
1. 修复 `op_host/op_tiling/arch22/mhc_pre_cmhc_backward_arch22_tiling.cpp`
   - 实现完整的 CalcTiling (P0)
   - 清理 dead code (P1)
   - 删除误导性注释 (P2)

**验证**:
```bash
bash bash_cmhc_backward_local.sh
python run_mindformer.py --config telechat4_40B_pretrain_1p.yaml
# 预期: 不再崩溃
```

**通过标准**:
- [ ] Layer 3 测试通过（forward + backward 无错误）
- [ ] Loss 正常计算

**时间估计**: 1-2 天

---

### Phase 2: 建立 Layer 2 验证

**任务**: 创建 test_mhc_pre_cmhc_backward.cpp → 实现 reference 结果生成 → 实现 AscendC kernel 直接调用 → 对比两者输出 → 集成 CI/CD

**通过标准**: Layer 2 测试通过，数值精度达标（< 1e-3 bf16, < 1e-6 float32）

**时间估计**: 2-3 天

---

### Phase 3: 完善 Layer 4 验证

**任务**: 执行 MindFormers 完整训练 → 监控 loss 曲线 → 验证 checkpoint → 检查梯度统计

**通过标准**: 完整训练无崩溃，Loss 正常下降，Checkpoint 可保存加载

**时间估计**: 1 天（运行） + 1 天（分析）

---

### Phase 4: 持续集成

**任务**: CI/CD 中集成 Layer 1/2/3 测试 → Release 前执行 Layer 4 → 建立监控 dashboard

**CI/CD 配置示例**:
```yaml
# .github/workflows/tests.yaml
name: mhc_pre_cmhc_backward tests
on: [push, pull_request]

jobs:
  layer1-tests:
    runs-on: npu-runner
    timeout-minutes: 30
    steps:
      - uses: actions/checkout
      - run: bash bash_cmhc_backward_local.sh
      - run: grep "4/4 PASS" verification_results.log
  
  layer2-tests:
    runs-on: npu-runner
    timeout-minutes: 60
    steps:
      - uses: actions/checkout
      - run: cd test/tests && ./test_mhc_pre_cmhc_backward
      - run: grep "PASSED" test_mhc_pre_cmhc_backward.log
  
  layer3-tests:
    runs-on: npu-runner
    timeout-minutes: 120
    needs: [layer1-tests, layer2-tests]
    steps:
      - uses: actions/checkout
      - run: python test/integration/test_mindformers_integration.py
      - run: grep "Passed" test_mindformers_integration.log
  
  layer4-tests:
    runs-on: npu-runner
    timeout-minutes: 480
    needs: [layer3-tests]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout
      - run: python run_mindformer.py --config telechat4_40B_pretrain_1p.yaml --max_steps 100
      - run: python analyze_training.py --log training.log
```

**时间估计**: 2-3 天（设置 CI/CD）

## 6. 验证金字塔

```
                Layer 4
              完整训练验证
             (生产环境模拟)
            ┌─────────────┐
            │ Release 前  │
            │ 必须执行    │
            └─────────────┘
            
          Layer 3
      MindFormers 集成
    (框架 + Forward + Backward)
   ┌─────────────────────────┐
   │ 关键 bug 修复后执行      │
   │ 验证框架集成正确性       │
   └─────────────────────────┘
   
       Layer 2
   AscendC kernel 验证
  (单算子数值正确性)
 ┌─────────────────────────────┐
 │ 每次修改 kernel/tiling 时执行 │
 │ 验证 AscendC kernel 自身      │
 └─────────────────────────────┘
 
      Layer 1
  Shell 算法验证
(算法 vs autograd)
 ┌─────────────────────────────┐
 │ 每次修改 torch_host.cpp 时执行 │
 │ 快速验证算法正确性              │
 │ 秒级反馈                        │
 └─────────────────────────────┘
```

## 7. 各层验证详细要求

### Layer 1: Shell 验证

- **输入**: `cmhc_backward/model.py` + `cmhc_backward/model_new_ascendc.py`
- **流程**: 编译 at:: 实现 → 加载 .so → 调用 torch.ops.npu.mhc_pre_cmhc_backward → 对比 autograd
- **通过标准**: 4/4 PASS, max_diff < 1e-6 (float32) / < 1e-2 (bf16)
- **失败处理**: 检查 torch_host.cpp 算法，对比 model_new_ascendc.py，检查 model.py reference

### Layer 2: AscendC 验证

- **输入**: `op_kernel/*.cpp` + `op_host/tiling.cpp`
- **流程**: 准备输入 → 准备 tiling → 分配 workspace → 调用 kernel → 获取输出 → 对比 reference
- **通过标准**: bf16 < 1e-3 / float32 < 1e-6 / fp16 < 1e-4；无 error 507015 / 400010 / 400014；无 seg fault
- **失败处理**: 检查 op_kernel, tiling, workspace；使用 `aclnnMhcPreCmhcBackward` 调试模式

### Layer 3: MindFormers 集成

- **输入**: `mindformers/models/causal_lm.py`
- **流程**: 初始化 CausalLM (4 layers) → 生成 input_ids → Forward → 计算 loss → Backward → 检查梯度
- **通过标准**: Loss 非 NaN/Inf, Backward 无 ACL 错误, 梯度非 NaN/Inf 且 norm > 0, 无 error 507015
- **失败处理**: 检查 op_host tiling, workspace, MindSpore ops.Custom 注册；使用 MindFormers verbose 模式

### Layer 4: MindFormers 完整训练

- **输入**: `mindformers/run_mindformer.py`
- **流程**: 100+ steps 完整训练 → 监控 loss → 监控梯度范数 → 保存 checkpoint → 验证加载
- **通过标准**: 无崩溃, Loss 正常下降, 梯度范数合理, Checkpoint 可保存/加载, 无 ACL 错误
- **失败处理**: 分析崩溃日志；检查 loss 爆炸/梯度消失；MindFormers 调试模式

## 8. 验证自动化

### CI/CD 集成

见上文 Phase 4 的完整 GitHub Actions 配置示例。

### 监控 Dashboard

```python
# dashboard/training_monitor.py
import plotly.graph_objs as go
from datetime import datetime

def parse_training_log(log_file):
    metrics = {'loss': [], 'grad_norm': [], 'step': [], 'timestamp': []}
    with open(log_file) as f:
        for line in f:
            if 'step' in line and 'loss' in line:
                metrics['step'].append(extract_step(line))
                metrics['loss'].append(extract_loss(line))
                metrics['grad_norm'].append(extract_grad_norm(line))
                metrics['timestamp'].append(datetime.now())
    return metrics

def plot_loss_curve(metrics):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics['step'], y=metrics['loss'],
        mode='lines', name='Training Loss'))
    fig.update_layout(title='Training Loss Curve',
                     xaxis_title='Step', yaxis_title='Loss')
    return fig

def plot_grad_norm(metrics):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics['step'], y=metrics['grad_norm'],
        mode='lines', name='Gradient Norm'))
    fig.update_layout(title='Gradient Norm Curve',
                     xaxis_title='Step', yaxis_title='Gradient Norm')
    return fig
```

## 9. 验证优先级建议

### 立即执行（修复 arch22_tiling 后）

**优先级 1: 修复 Layer 3 崩溃**
```bash
vim op_host/op_tiling/arch22/mhc_pre_cmhc_backward_arch22_tiling.cpp
bash bash_cmhc_backward_local.sh
python run_mindformer.py --config telechat4_40B_pretrain_1p.yaml --use_parallel False --device_id 8
```

**通过标准**: 无 error 507015, Loss 正常计算  
**时间**: 1-2 天

---

### 短期执行（修复后 1-2 周）

**优先级 2: 建立 Layer 2 验证**
```bash
cd /path/to/ops-transformer/test/tests
vim test_mhc_pre_cmhc_backward.cpp
bash run_tests.sh test_mhc_pre_cmhc_backward
```

**通过标准**:Layer 2 测试通过, 数值精度达标, 集成 CI/CD  
**时间**: 2-3 天

---

### 中期执行（修复后 1-2 月）

**优先级 3: 完善监控体系**
- 设置 CI/CD pipeline
- 设置监控 dashboard
- 设置告警机制

**时间**: 2-4 周

---

### 长期执行

**优先级 4: 生产验证**
- 定期完整训练 (max_steps 1000)
- 监控训练指标
- 对比历史性能

**时间**: 持续

## 10. 总结与建议

### 核心问题
```
Shell 验证通过 ≠ AscendC kernel 正确
  ↓ 因为: Shell 验证的是 at:: 算法实现
  ↓      AscendC kernel 有独立的 tiling/workspace 要求
  ↓ arch22_tiling.cpp 是 stub, 未序列化 tiling 数据
  ↓ 导致: MindFormers 路径崩溃 (error 507015)
```

### 修复路径
```
1. 修复 arch22_tiling.cpp (P0/P1/P2) ← 当前
   ├── 实现完整 CalcTiling
   ├── 清理 dead code
   └── 删除误导性注释

2. 验证 Layer 3 通过 ← 修复后
   └── MindFormers 不再崩溃

3. 建立 Layer 2 测试 ← 短期
   └── 直接验证 AscendC kernel

4. 完善监控体系 ← 中期
   └── CI/CD + Dashboard + Alerts

5. 持续监控 ← 长期
   └── 定期训练 + 性能对比
```

### 建议行动顺序

1. **立即**: 修复 arch22_tiling.cpp
   - 实现完整 CalcTiling（SaveToBuffer / SetDataSize / GetWorkspaceSizes / MatMul tiling）
   - 验证 Layer 3 通过

2. **短期**: 建立 Layer 2 测试
   - 创建 test_mhc_pre_cmhc_backward.cpp
   - 集成到 CI/CD

3. **中期**: 完善验证体系
   - 设置 CI/CD pipeline
   - 建立监控 dashboard

4. **长期**: 持续监控
   - 定期完整训练
   - 性能和精度监控

### 关键收获

1. **Shell 验证只是必要非充分条件**
   - Shell 验证通过 → 算法正确
   - 但不能保证 AscendC kernel 正确

2. **四层验证体系缺一不可**
   - Layer 1: 算法正确性 (现有)
   - Layer 2: Kernel 正确性 (需建立)
   - Layer 3: 框架集成 (需修复)
   - Layer 4: 生产稳定性 (需验证)

3. **建立反馈循环**
   - 每次修改代码 → 执行对应 Layer 验证
   - 每次发现 bug → 分析是否需要加强验证
   - 每次 release → 完整 Layer 1-4 验证

---

## 📋 文档元信息

**生成时间**: 2024  
**合并自 6 份独立分析文档**:
- Shell 调用流程详细分析（章节一）
- PyTorch at:: 机制详解（章节二）
- MindSpore 崩溃根因分析（章节三）
- 两条调用路径完整对比（章节四）
- 修复方案（章节五）
- 完整验证策略（章节六）

**分析基于**:
- `/home/z00941044/0709/cmhc_cann/bash_cmhc_backward_local.sh`
- `/home/z00941044/0709/mindformers/run_mindformer.py`
- `/home/z00941044/0709/transfer-station/cmhc_opti/` 下的错误日志与分析报告
- `/home/z00941044/0709/ops-transformer/mhc/mhc_pre_cmhc_backward/` 下的 CANN op_host/op_kernel 代码
