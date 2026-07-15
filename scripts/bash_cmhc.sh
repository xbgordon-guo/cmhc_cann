#!/bin/bash
# === 用法 ===
#   bash bash_cmhc.sh <OPS_TRANSFORMER_DIR> <CMHC_CANN_DIR> <CANNBOT_SKILLS_DIR> [TORCH_LIB_DIR]
#
# 示例:
#   bash bash_cmhc.sh /path/ops-transformer /path/cmhc_cann /path/cannbot-skills/plugins-community/ascendc-ops-lab-developer

set -euo pipefail

# === 入参解析 ===
OPS_TRANSFORMER_DIR="${1:?请指定 ops-transformer 目录}"
CMHC_CANN_DIR="${2:?请指定 cmhc_cann 目录}"
CANNBOT_SKILLS_DIR="${3:?请指定 cannbot-skills/plugins-community/ascendc-ops-lab-developer 目录}"
TORCH_LIB_DIR="${4:-/usr/local/python3.11.10/lib/python3.11/site-packages/torch/lib}"

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
python -c "import torch; torch.ops.load_library('mhc_pre_cmhc_torch_ops.cpython-311-x86_64-linux-gnu.so'); print('OK')"

# 精度验证
export ASCEND_RT_VISIBLE_DEVICES=1
python "${CANNBOT_SKILLS_DIR}/.claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py" "${CMHC_CANN_DIR}/cmhc/"

# 性能验证
python "${CANNBOT_SKILLS_DIR}/.claude/skills/ops-profiling/scripts/msprof_perf_summary.py" --quick --output-dir="${CMHC_CANN_DIR}/cmhc/"
