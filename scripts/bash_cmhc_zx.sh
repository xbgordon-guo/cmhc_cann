#!/bin/bash
# === 用法 ===
#   bash bash_cmhc.sh <OPS_TRANSFORMER_DIR> <CMHC_CANN_DIR> <CANNBOT_SKILLS_DIR> [TORCH_LIB_DIR]
#
# 示例:
#   bash bash_cmhc.sh /path/ops-transformer /path/cmhc_cann /path/cannbot-skills/plugins-community/ascendc-ops-lab-developer

set -euo pipefail

# === 路径配置 ===
OPS_TRANSFORMER_DIR=/home/z00941044/0709/ops-transformer
CMHC_CANN_DIR=/home/z00941044/0709/cmhc_cann
CANNBOT_SKILLS_DIR=/home/z00941044/0709/cannbot-skills
TORCH_LIB_DIR=/usr/local/python3.11.14/lib/python3.11/site-packages/torch/lib

# === AscendC Kernel（改 op_kernel/ 或 op_host/ 下代码后） ===
cd "${OPS_TRANSFORMER_DIR}"
rm -rf build/binary/ascend910b/src/mhc_pre_cmhc
bash build.sh --opkernel --soc=ascend910b --ops=mhc_pre_cmhc

# === Python Binding（改 binding/ 下代码后） ===
cd "${CMHC_CANN_DIR}/cmhc/binding"
rm -rf build/ *.so *.o
python setup.py build_ext --inplace

# 验证

# Binding 加载验证
cd "${CMHC_CANN_DIR}/cmhc/binding"
LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
SO_NAME=$(ls mhc_pre_cmhc_torch_ops.cpython-311-*.so | head -1)
python -c "import torch; torch.ops.load_library('${SO_NAME}'); print('OK')"

# 精度验证
export ASCEND_RT_VISIBLE_DEVICES=1
python "${CANNBOT_SKILLS_DIR}/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/scripts/verification_ascendc.py" "${CMHC_CANN_DIR}/cmhc/"

# 性能验证
python "${CANNBOT_SKILLS_DIR}/ops/ops-profiling/scripts/msprof_perf_summary.py" --quick --output-dir="${CMHC_CANN_DIR}/cmhc/"
