# CMHC Kernel 代码逻辑分析

> 分析对象：`path/cmhc_cann/cmhc`（前向）+ `path/cmhc_cann/cmhc_backward`（反向）

---

## 目录

- [1. 概述](#1-概述)
- [2. 前向 Kernel (cmhc)](#2-前向-kernel-cmhc)
  - [2.1 输入与输出](#21-输入与输出)
  - [2.2 计算流程](#22-计算流程)
  - [2.3 关键设计要点](#23-关键设计要点)
  - [2.4 代码架构](#24-代码架构)
- [3. 反向 Kernel (cmhc_backward)](#3-反向-kernel-cmhc_backward)
  - [3.1 输入与输出](#31-输入与输出)
  - [3.2 反向传播的 8 个步骤](#32-反向传播的-8-个步骤)
  - [3.3 AscendC Kernel 架构](#33-ascendc-kernel-架构)
  - [3.4 与旧版 Sinkhorn 反向的差异](#34-与旧版-sinkhorn-反向的差异)
- [4. 两份实现对比](#4-两份实现对比)

---

## 1. 概述

`cmhc` 算子实现了 **mHC: Manifold-Constrained Hyper-Connections**（论文 arXiv:2512.24880）的前向与反向计算。核心思想是通过多流（multi-stream）注意力/MLP 的超连接机制，用 **Birkhoff-von Neumann 置换矩阵组合** 替代传统的 Sinkhorn-Knopp 迭代归一化。

关键参数：
- **N = 4**：残差流的数量（hc_mult）
- **C**：每条流的通道维度（如 3584、4096、7168）
- **n! = 24**：4×4 置换矩阵的总数
- **outDim = n! + 2N = 32**：投影输出维度

---

## 2. 前向 Kernel (cmhc)

### 2.1 输入与输出

```
输入 (5 个):
  x         (B, S, N, C)      输入张量（bf16/fp16）
  phi       (outDim, N*C)     投影权重矩阵（fp32），outDim = n! + 2N = 32
  alpha     (3,)              缩放参数（fp32）
  bias      (outDim,)         偏置参数（fp32）
  perm_mats (n!, N, N)        置换矩阵，Python 侧预计算（fp32）

输出 (6 个):
  h_in           (B, S, C)         聚合后的层输入
  h_post         (B, S, N)         post 权重
  h_res          (B, S, N, N)      残差连接矩阵（双随机矩阵）
  h_pre          (B, S, N)         pre 权重
  hc_before_norm (B, S, outDim)    投影中间结果 H = x @ phi^T
  inv_rms        (B, S)            RMS norm 因子

属性 (5 个):
  hc_mult   = 4     N（残差流数量）
  num_iters = 20    保留参数，Softmax-Permutation 方案不使用
  hc_eps    = 1e-6  防除零参数
  norm_eps  = 1e-6  RMSNorm 防除零参数
  need_backward      是否输出反向所需中间结果
```

### 2.2 计算流程

设 `M = B * S`，`nC = N * C`。

```
Step 1: Flatten + Cast
  x_flat = x.reshape(M, nC).float()          → shape (M, nC)

Step 2: RMSNorm
  r = sqrt(mean(x_flat²))                     → shape (M, 1)
  r_inv = 1 / (r + eps)                       → shape (M, 1)

Step 3: 线性投影 (MatMul)
  H = x_flat @ phi^T                           → shape (M, outDim=32)
  (phi 形状: (outDim, nC)，已在 Python 侧预转置)

Step 4: RMS 修正 + Alpha/Bias 缩放
  alpha_exp = [α₀]*N + [α₁]*N + [α₂]*nPerm    → shape (1, 32)
  act = r_inv * H * alpha_exp + bias           → shape (M, 32)

Step 5: 拆分 + 激活
  H 的 32 维拆为三段：
  ├─ [0:4)   → H_pre_raw  → sigmoid           → h_pre   (M, 4)
  ├─ [4:8)   → H_post_raw → 2 × sigmoid       → h_post  (M, 4)
  └─ [8:32)  → H_res_raw  → softmax           → res_coeff (M, 24)

Step 6: Birkhoff-von Neumann 置换矩阵组合
  h_res = einsum('mr, rij -> mij', res_coeff, perm_mats)
  即 24 个 4×4 置换矩阵的加权凸组合 → shape (M, 4, 4)
  结果保证是双随机矩阵（每行每列和为 1）

Step 7: 加权聚合
  x_streams = x_flat.reshape(M, N, C)          → shape (M, 4, C)
  h_in = Σ_n x_streams[n] * h_pre[n]           → shape (M, C)

Step 8: Reshape 输出
  将所有 (M, ...) 张量 reshape 回 (B, S, ...)
```

### 2.3 关键设计要点

#### 为什么是 32 维输出？

`outDim = n! + 2N = 24 + 8 = 32`：
- **N=4 个 pre 权重**：控制每条流进入下一层的权重（sigmoid → (0, 1)）
- **N=4 个 post 权重**：控制每条流输出的缩放（2×sigmoid → (0, 2)）
- **n! = 24 个 res 系数**：通过 softmax 得到 24 个置换矩阵的凸组合系数

#### Birkhoff-von Neumann 定理

根据 Birkhoff-von Neumann 定理，**任意双随机矩阵都可以表示为置换矩阵的凸组合**。这里用 softmax 学习 24 个组合系数，生成的 `h_res` 是一个 `(N, N) = (4, 4)` 的双随机矩阵，用于在 4 条流之间做残差重排。

#### Gamma Blending（正则化）

```python
perm_mats = gamma * P_p + (1 - gamma) / N * J
```

- `P_p`：纯置换矩阵（每行每列恰好一个 1）
- `J`：全 1 矩阵
- `gamma = 1.0`：使用纯置换矩阵
- `gamma < 1.0`：向均匀分布混合，增加正则化

置换矩阵在 **Python 侧预计算并缓存**，作为输入张量 `perm_mats` 传入 kernel，避免每次前向重新生成，也避免了在 kernel 内部硬编码 Gamma 混合逻辑。

### 2.4 代码架构

```
cmhc/
├── op_graph/
│   └── mhc_pre_cmhc_proto.h          # 算子 IR 定义（输入/输出/属性）
├── op_host/                           # Host 侧：Tiling + Infershape
├── op_kernel/                         # Device 侧：AscendC kernel 实现
├── binding/
│   ├── mhc_pre_cmhc_torch_host.cpp    # C++ PyTorch 绑定（at:: 实现，匹配 kernel）
│   ├── register.cpp                   # torch.ops 注册
│   └── setup.py                       # 编译脚本
├── model.py                           # PyTorch 纯参考实现（含 gamma + RMSNorm weight）
├── model_new_ascendc.py               # AscendC kernel 的 nn.Module 封装
├── mhc_pre_sinkhorn.json              # 测试用例
└── docs/
    └── aclnnMhcPreSinkhorn.md         # CANN 官方 API 文档
```

---

## 3. 反向 Kernel (cmhc_backward)

### 3.1 输入与输出

反向需要前向的 3 个输出梯度 + 8 个前向中间结果：

```
输入 (11 个):
  上游梯度:
    grad_h_in      (B, S, C)         loss 对 h_in 的梯度
    grad_h_post    (B, S, N)         loss 对 h_post 的梯度
    grad_h_res     (B, S, N, N)      loss 对 h_res 的梯度

  前向输入/参数:
    x              (B, S, N, C)      前向输入（bf16/fp16）
    phi            (outDim, N*C)     前向权重（fp32）
    alpha          (3,)             前向缩放（fp32）
    bias           (outDim,)        前向偏置（fp32）

  前向中间结果:
    h_pre          (B, S, N)        前向 sigmoid 输出（fp32）
    hc_before_norm (B, S, outDim)   前向投影 H = x @ phi^T（fp32）
    inv_rms        (B, S)           前向 RMS norm 因子 r_inv（fp32）
    perm_mats      (n!, N, N)       置换矩阵，Python 侧预计算（fp32）

输出 (4 个):
  grad_x         (B, S, N, C)     loss 对 x 的梯度
  grad_phi       (outDim, N*C)    loss 对 phi 的梯度
  grad_alpha     (3,)            loss 对 alpha 的梯度
  grad_bias      (outDim,)       loss 对 bias 的梯度
```

### 3.2 反向传播的 8 个步骤

设 `M = B * S`，`nC = N * C`。反向过程沿正向计算图反向遍历：

#### Step 1: 加权聚合的反向

正向：`h_in = Σ_n x_streams[n] * h_pre[n]`

```
grad_x_streams[n,:] = grad_h_in * h_pre[n]              → shape (M, N, C)
grad_h_pre[n]       = Σ_c grad_h_in_c * x_streams[n,c]  → shape (M, N)
```

即：`grad_h_in` 按 `h_pre` 广播到每条流的 C 维 → `grad_x_streams`；`grad_h_in` 与 `x_streams` 内积 → `grad_h_pre`。

#### Step 2: Sigmoid 反向（pre 分支）

正向：`h_pre = sigmoid(h_pre_raw)`

```
sigmoid_grad   = h_pre * (1 - h_pre)            → shape (M, N)
grad_h_pre_raw = grad_h_pre * sigmoid_grad      → shape (M, N)
```

Sigmoid 导数性质：`dσ(x)/dx = σ(x)(1 - σ(x))`。

#### Step 3: 2×Sigmoid 反向（post 分支）

正向：`h_post = 2 * sigmoid(h_post_raw)`

```
h_post_raw       = r_inv * H[:, N:2N] * alpha[1] + bias[N:2N]   (重算前向中间值)
h_post_sigmoid   = sigmoid(h_post_raw)
sigmoid_grad     = sigmoid * (1 - sigmoid)
grad_h_post_raw  = grad_h_post * 2 * sigmoid_grad                → shape (M, N)
```

系数 `2` 来源于正向 `h_post = 2 * sigmoid(h_post_raw)`。

#### Step 4: Softmax-Permutation 反向 ★核心★

这是与旧版 Sinkhorn 反向的**最大区别**。正向：`h_res = einsum('mr, rij -> mij', softmax(h_res_logits), perm_mats)`

**4a. 前向 Softmax 重算**

```
h_res_logits = r_inv * H[:, 2N:2N+nPerm] * alpha[2] + bias[2N:2N+nPerm]
res_coeff    = softmax(h_res_logits)        → shape (M, 24)
// nPerm = n! = 24 (N=4 时)
```

在 AscendC kernel 中，softmax 通过三步实现：
1. 找最大值 `mx = max(h_res_logits)`
2. `exp(x - mx)` + 求和
3. 归一化（除以和）

**4b. Einsum 反向：grad_res_coeff**

```
grad_res_coeff[p] = Σ_i,j grad_h_res[i,j] * perm_mats[p,i,j]
// 等价于 einsum('mij, rij -> mr', grad_h_res, perm_mats) → shape (M, 24)
```

这一步将 `(4, 4)` 的梯度矩阵通过每个置换矩阵的 mask 投影回 24 维系数空间。对于纯置换矩阵（gamma=1），`perm_mats[p,i,j]` 只在置换映射 `i→j` 的位置为 1，其余为 0，所以等价于：

```
grad_res_coeff[p] = Σ_i grad_h_res[i, perm[p,i]]
// 即按置换 p 对应的 (行→列) 映射收集梯度
```

**4c. Softmax 反向**

```
weighted       = Σ_k(grad_res_coeff_k * res_coeff_k)          // 标量，每行一个
grad_logits_k  = res_coeff_k * (grad_res_coeff_k - weighted)
```

这是标准的 softmax 雅可比矩阵公式：`∂L/∂z_i = y_i * (∂L/∂y_i - Σ_j y_j * ∂L/∂y_j)`。

#### Step 5: 拼接 + Alpha/Bias 链式法则

将三路梯度拼接，然后反向穿过 `act = r_inv * H * alpha_exp + bias`：

```
grad_act = [grad_h_pre_raw, grad_h_post_raw, grad_h_res_logits]  → shape (M, 32)

grad_H[:,i]       = grad_act[:,i] * alpha_exp[i] * r_inv    → shape (M, 32)
grad_r_inv        = Σ_i(grad_act[:,i] * H[:,i] * alpha_exp[i])
grad_bias[i]      = Σ_m grad_act[m,i]                       → shape (32,)

grad_alpha[0]     = Σ_{m, i∈[0,4)}  grad_act * r_inv * H    → pre 段
grad_alpha[1]     = Σ_{m, i∈[4,8)}  grad_act * r_inv * H    → post 段
grad_alpha[2]     = Σ_{m, i∈[8,32)} grad_act * r_inv * H    → res 段
```

#### Step 6: MatMul 反向

正向：`H = x_flat @ phi^T`，形状 `(M, outDim) = (M, nC) @ (nC, outDim)^T`

```
grad_x_matmul = grad_H @ phi        → shape (M, nC)
grad_phi      = grad_H^T @ x_flat   → shape (outDim, nC)
```

#### Step 7: RMS Norm 反向

正向：`r_inv = 1 / (sqrt(mean(x²)) + eps)`

```
r = sqrt(mean(x_flat²))
grad_x_rms = -x_flat * grad_r_inv * r_inv² / (r * nC)
```

推导链：`r_inv → r → mean_sq → x² → x`，其中：
- `∂r_inv/∂r = -1/(r+eps)² = -r_inv²`
- `∂r/∂(mean_sq) = 1/(2r)`
- `∂(mean_sq)/∂(x²) = 1/nC`
- `∂(x²)/∂x = 2x`

合起来：`∂r_inv/∂x = -x * r_inv² / (r * nC)`。

#### Step 8: 合并 grad_x

```
grad_x_flat = grad_x_matmul + grad_x_rms + grad_x_streams.reshape(M, nC)
grad_x = grad_x_flat.reshape(B, S, N, C).to(input_dtype)
```

三部分分别来自：MatMul 梯度的反向传播、RMS Norm 的梯度、加权聚合的 `grad_x_streams` 贡献。

### 3.3 AscendC Kernel 架构

```
Process()
├─ AIV (Vector 核)                        ├─ AIC (Cube 核)
│                                          │
│  GetHcScaleAndHcBase()                  │
│    加载 alpha, bias 到 UB               │
│    加载 perm_mats 到 UB（24×4×4）       │
│                                          │
│  for each tile (BS 分块):               │  for each tile:
│    ├─ ComputeGradPre()                  │    CrossCoreWaitFlag()
│    │   └─ Step 1:                       │    ├─ ProcessMatmul1()
│    │      grad_x_streams                │    │   MatMul: gradH2(M,32) @ phi(32,nC)
│    │      累加 grad_h_pre               │    │   → gradXCube(M, nC)
│    │      (Kahan 求和保证精度)          │    │
│    │                                     │    ├─ ProcessMatmul2()
│    ├─ ComputeGradHHat2()                │    │   MatMul: gradH2^T(32,M) @ xW(M,nC)
│    │   ├─ SoftmaxPermutationGrad() ★    │    │   → gradPhi(32, nC)
│    │   │   Step 4a: softmax 前向重算    │    CrossCoreSetFlag()
│    │   │   Step 4b: grad_res_coeff      │
│    │   │     = Σ grad_h_res * perm      │
│    │   │   Step 4c: softmax 反向梯度    │
│    │   ├─ Step 2-3: sigmoid 反向        │
│    │   ├─ Step 5: alpha/bias 链式       │
│    │   ├─ Step 7: RMS norm 反向         │
│    │   └─ 写 grad_H2 到 workspace GM    │
│    │                                     │
│    └─ ComputeGradX1()                   │
│        ├─ Step 6: grad_x_matmul 贡献    │
│        ├─ Step 7: grad_x_rms 贡献       │
│        └─ 写入 grad_x GM                │
│                                          │
│  ComputeScaleBias()                     │
│    ├─ Reduce grad_bias (32,)            │
│    ├─ Reduce grad_alpha (3,)            │
│    └─ 写入 GM (AtomicAdd/确定性)        │
│                                          │
│  (DETERMINISTIC=true 时)                │
│    ComputeDeterministic()                │
│      多核结果合并                        │
```

**AIV-AIC 流水线协作**：
- AIV 先计算 `grad_H(Ĥ₂)`（shape: `tileTaskCount × 32`），通过 workspace GM 传给 AIC
- AIC 用 Cube 单元执行两个 MatMul：
  - `MM1: grad_H @ phi → grad_x_cube`（给 AIV 后续 ComputeGradX1 使用）
  - `MM2: grad_H^T @ x_workspace → grad_phi`
- 通过 `CrossCoreSetFlag<0x2>` / `CrossCoreWaitFlag<0x2>` 做 barrier 同步

**Kahan 求和**：在 `ComputeGradPre` 和 `ComputeGradHHat2` 中使用 Kahan 补偿求和算法 `kahanCustom()`，减少浮点累加时的截断误差。

**确定性模式**：当 `DETERMINISTIC=true` 时，各 AIV 核的 `grad_bias`、`grad_alpha`、`grad_phi` 先写入 workspace，然后由 `ComputeDeterministic()` 多核归约合并，保证结果与核调度顺序无关。

### 3.4 与旧版 Sinkhorn 反向的差异

| | 旧版 (SinkhornGrad) | 新版 (SoftmaxPermutationGrad) |
|---|---|---|
| H_res 梯度计算 | Sinkhorn-Knopp 迭代反向传播 | 解析 softmax + permutation einsum 梯度 |
| 额外输入 | `sumOut`, `normOut`（迭代中间结果） | `perm_mats`（置换矩阵，Python 预计算） |
| 迭代次数 | 固定 20 次（Kernel 内循环） | 无需迭代 |
| 计算量 | O(iter × N²) = O(20 × N²) per row | O(n! × N²) = O(24 × 16) = 384 ops per row |
| 数值精度 | Sinkhorn 迭代累积误差 | 解析梯度，精度更高 |
| Kernel 复杂度 | 高（迭代循环 + 多组中间缓冲区） | 低（三层嵌套循环，直接 einsum） |
| 内存占用 | 高（需存储 40 步迭代中间结果） | 低（仅 nPerm × N × N 的置换矩阵） |

核心改进：用 **Birkhoff-von Neumann 定理** 将双随机矩阵表示为置换矩阵的凸组合，反向传播变为对 24 个 softmax 系数求解析梯度，避免了对 Sinkhorn 迭代过程的反向传播，大幅降低 kernel 复杂度和内存占用。

---

## 4. 两份实现对比

| | `model.py`（前向参考） | `model_new_ascendc.py`（AscendC 前向） | `backward_reference.py`（反向参考） | AscendC Backward Kernel |
|---|---|---|---|---|
| RMSNorm | `x.norm()` + `gamma` 权重 | `mean(x²)` → `1/sqrt`（无 gamma） | 同 forward | 同 forward |
| MatMul | Python `@`（fp32） | AscendC Cube 单元 | Python `@` | AscendC Cube 单元（MM1 + MM2） |
| 置换矩阵 einsum | Python `einsum` | Python `einsum`（kernel 输出 res_coeff 后） | Python `einsum` 反向 | AIV Vector 核：三重循环直接计算 |
| Softmax | `torch.softmax` | AscendC kernel 内实现 | `torch.softmax` | AIV 手动实现（max + exp + normalize） |
| 梯度计算 | `torch.autograd.grad` | 不适用 | 手动解析梯度 | AIV + AIC 混合（Vector + Cube） |
| 精度保证 | fp32 | bf16/fp16 输入，fp32 中间计算 | fp32 | Kahan 求和 + 确定性归约 |
| 输出数量 | 3（h_in, h_post, h_res） | 6（含反向中间结果） | 4（grad_x, grad_phi, grad_alpha, grad_bias） | 4 |
