# 性能评估结果

- **Operator**: mhc_pre_cmhc
- **Device**: npu:12 (source=auto)
- **Warmup**: 3
- **Repeats**: 1
- **Seed**: 0
- **Timing method**: msprof.quick.Task_Duration

## 性能对比

| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |
| ---- | ----- | ----- | ------------- | -------- | -------------- |
| 0 | [1, 128, 4, 3584] | bfloat16 | 1.18 | 1.18 | 0.999 |
| 1 | [2, 256, 4, 3584] | bfloat16 | 1.17 | 1.17 | 1.001 |
| 2 | [4, 512, 4, 3584] | bfloat16 | 62.11 | 27.83 | 0.448 |

## 全量汇总

| 指标 | 值 |
| ---- | -- |
| 用例数 | 3 |
| 平均加速比（>1 表示自定义算子更快） | 0.816 |
| 自定义算子更优（比值>1） | 1 |
| 标杆更优（比值<1） | 2 |

### 按数据类型汇总

| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |
| ----- | ------ | ------------------- | ------------- | -------- |
| bfloat16 | 3 | 0.816 | 1 | 2 |

## 简短分析

- 平均加速比 0.816 小于 1，标杆路径整体更优。
- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。

## 深度瓶颈分析

如需进一步分析性能瓶颈（各流水线利用率、核间负载均衡、主 Bound 判定），可运行：
```bash
python3 ${SKILL_PATH}/scripts/msprof_perf_summary.py ./PROF_GROUP_* mhc_pre_cmhc
```

或参考 `ops-profiling/references/optimization_quickref.md` 获取优化建议。
