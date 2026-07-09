# 性能评估结果

- **Operator**: cmhc_backward
- **Device**: npu:1 (source=env)
- **Warmup**: 3
- **Repeats**: 1
- **Seed**: 0
- **Timing method**: msprof.quick.Task_Duration

## 性能对比

| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |
| ---- | ----- | ----- | ------------- | -------- | -------------- |
| 0 | [1, 2, 64] | float32 | 1.75 | 5.38 | 3.076 |
| 1 | [2, 3, 128] | float16 | 2.71 | 6.01 | 2.221 |
| 2 | [1, 4, 256] | bfloat16 | 2.31 | 6.05 | 2.622 |

## 全量汇总

| 指标 | 值 |
| ---- | -- |
| 用例数 | 3 |
| 平均加速比（>1 表示自定义算子更快） | 2.640 |
| 自定义算子更优（比值>1） | 3 |
| 标杆更优（比值<1） | 0 |

### 按数据类型汇总

| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |
| ----- | ------ | ------------------- | ------------- | -------- |
| bfloat16 | 1 | 2.622 | 1 | 0 |
| float16 | 1 | 2.221 | 1 | 0 |
| float32 | 1 | 3.076 | 1 | 0 |

## 简短分析

- 平均加速比 2.640 大于 1，自定义算子整体有优势。
- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。

## 深度瓶颈分析

如需进一步分析性能瓶颈（各流水线利用率、核间负载均衡、主 Bound 判定），可运行：
```bash
python3 ${SKILL_PATH}/scripts/msprof_perf_summary.py ./PROF_GROUP_* cmhc_backward
```

或参考 `ops-profiling/references/optimization_quickref.md` 获取优化建议。
