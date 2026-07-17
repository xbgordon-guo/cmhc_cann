set -euo pipefail

# === 入参解析 ===
OPS_TRANSFORMER_DIR="${1:?请指定 ops-transformer 目录}"
 # === AscendC Kernel（改 op_kernel/ 或 op_host/ 下代码后） ===
cd "${OPS_TRANSFORMER_DIR}"
rm -rf build/*
# 1. 编译
bash build.sh --pkg --soc=ascend910b --ops=mhc_pre_cmhc -j16

# 2. 安装
./build_out/cann-ops-transformer-*linux*.run

# 3. 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}

# 4. 运行 eager 测试
bash build.sh --run_example mhc_pre_cmhc eager cust --vendor_name=custom --soc=ascend910b

# 5. 性能采集
bash build.sh --run_example mhc_pre_cmhc eager cust --vendor_name=custom

cd build/

msprof --application="./test_aclnn_mhc_pre_cmhc"