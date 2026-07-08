# === AscendC Kernel（改 op_kernel/ 或 op_host/ 下代码后） ===
cd /home/y00889327/ops-transformer
rm -rf build/binary/ascend910b/src/mhc_pre_cmhc
bash build.sh --opkernel --soc=ascend910b --ops=mhc_pre_cmhc

# === Python Binding（改 binding/ 下代码后） ===
cd /home/y00889327/cmhc_cann/cmhc/binding
rm -rf build/ *.so *.o
python setup.py build_ext --inplace

# 验证

# Binding 加载验证
cd /home/y00889327/cmhc_cann/cmhc/binding
LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH \
python -c "import torch; torch.ops.load_library('mhc_pre_cmhc_torch_ops.cpython-311-x86_64-linux-gnu.so'); print('OK')"

# 精度验证
export ASCEND_RT_VISIBLE_DEVICES=1
python /home/y00889327/cannbot-skills/plugins-community/ascendc-ops-lab-developer/.claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py /home/y00889327/cmhc_cann/cmhc/

# 性能验证