#!/usr/bin/env bash
# ============================================================================
# cmhc WHL 打包脚本
#
# 用法:
#   bash package.sh                    # 构建 WHL
#   bash package.sh --install          # 构建 WHL + pip install + 验证
#   bash package.sh --run              # 构建 run 包 (tar.gz)
#   bash package.sh --clean            # 清理打包产物
#
# 环境变量:
#   ASCEND_HOME_PATH    CANN 安装路径 (默认: /usr/local/Ascend/latest)
#   TORCH_LIB_DIR       PyTorch lib 路径 (默认自动探测)
#   SOC_VERSION         目标 NPU 架构 (默认: Ascend910B3)
# ============================================================================

set -euo pipefail

# ── 路径计算 ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"                         # cmhc 根目录
PACKAGE_NAME="mhc_pre_cmhc"
BUILD_DIR="${PROJECT_DIR}/_package_build"
DIST_DIR="${PROJECT_DIR}/dist"
VERSION="${VERSION:-1.0.0}"

# ── 环境变量默认值 ──────────────────────────────────────────────────────────
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/latest}"
SOC_VERSION="${SOC_VERSION:-Ascend910B3}"

# 自动探测 torch lib 路径
if [[ -z "${TORCH_LIB_DIR:-}" ]]; then
    TORCH_LIB_DIR="$(python -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")"
    if [[ -z "${TORCH_LIB_DIR}" ]]; then
        TORCH_LIB_DIR="$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null || echo "")/lib"
    fi
fi

# ── 颜色 ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "${CYAN}[STEP]${NC} $*"; }

# ── 帮助 ────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
用法: bash package.sh [MODE]

MODE:
  (默认)        构建 WHL 包到 dist/
  --install     构建 WHL + pip install + 验证
  --run         构建 run 包 (tar.gz，无需 pip)
  --clean       清理 _package_build/ 和 dist/
  --help        显示帮助

环境变量:
  ASCEND_HOME_PATH   CANN 路径 (默认 /usr/local/Ascend/latest)
  SOC_VERSION        目标架构 (默认 Ascend910B3)
  VERSION            包版本号 (默认 1.0.0)

示例:
  bash package.sh
  bash package.sh --install
  VERSION=2.0.0 bash package.sh --run
EOF
}

# ── 前置检查 ────────────────────────────────────────────────────────────────
preflight() {
    step "前置检查..."
    local errors=0

    # Python
    if ! command -v python &>/dev/null; then
        error "未找到 python"; ((errors++))
    fi

    # PyTorch
    if ! python -c "import torch" 2>/dev/null; then
        error "未找到 torch"; ((errors++))
    fi

    # CANN 环境 (source set_env.sh)
    if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
        source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true
    else
        warn "未找到 ${ASCEND_HOME_PATH}/set_env.sh，跳过 CANN 环境 source"
    fi

    # 源码文件
    for f in binding/setup.py binding/register.cpp binding/mhc_pre_cmhc_torch_host.cpp binding/ops.h; do
        if [[ ! -f "${PROJECT_DIR}/${f}" ]]; then
            error "缺少文件: ${f}"; ((errors++))
        fi
    done

    if (( errors > 0 )); then
        error "前置检查失败 (${errors} 项错误)，退出"; exit 1
    fi
    info "前置检查通过"
}

# ── 编译 binding .so ────────────────────────────────────────────────────────
build_binding() {
    step "编译 Python binding..."
    cd "${PROJECT_DIR}/binding"

    rm -rf build/ *.so 2>/dev/null || true

    # 使用 Torch CppExtension 编译
    python setup.py build_ext --inplace

    # 确认产出
    local so_file
    so_file="$(ls mhc_pre_cmhc_torch_ops*.so 2>/dev/null | head -1)"
    if [[ -z "${so_file}" ]]; then
        error "binding .so 编译失败"; exit 1
    fi
    info "编译完成: binding/${so_file}"

    cd "${PROJECT_DIR}"
}

# ── 构建 WHL 包 ─────────────────────────────────────────────────────────────
build_whl() {
    step "构建 WHL 包..."

    # 准备临时构建目录
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}/${PACKAGE_NAME}"

    # ── 收集文件 ──

    # 1. binding .so
    local so_file
    so_file="$(ls "${PROJECT_DIR}/binding"/mhc_pre_cmhc_torch_ops*.so 2>/dev/null | head -1)"
    if [[ -z "${so_file}" ]]; then
        error "未找到 binding .so，请先编译"; exit 1
    fi
    cp "${so_file}" "${BUILD_DIR}/${PACKAGE_NAME}/"

    # 2. Python 模块
    cp "${PROJECT_DIR}/model_new_ascendc.py" "${BUILD_DIR}/${PACKAGE_NAME}/"
    cp "${PROJECT_DIR}/model.py"             "${BUILD_DIR}/${PACKAGE_NAME}/"
    cp "${PROJECT_DIR}/compare_precision.py" "${BUILD_DIR}/${PACKAGE_NAME}/" 2>/dev/null || true

    # 3. 数据文件
    cp "${PROJECT_DIR}/mhc_pre_sinkhorn.json" "${BUILD_DIR}/${PACKAGE_NAME}/" 2>/dev/null || true

    # ── 生成 __init__.py ──
    cat > "${BUILD_DIR}/${PACKAGE_NAME}/__init__.py" <<'PYEOF'
"""mhc_pre_cmhc — AscendC kernel operator for MHC (Multi-Head Composition)."""

import os
from pathlib import Path
import torch

# 加载 torch custom op (.so)
_pkg_dir = Path(__file__).resolve().parent
_so_files = list(_pkg_dir.glob("mhc_pre_cmhc_torch_ops*.so"))
if _so_files:
    torch.ops.load_library(str(_so_files[0]))

from .model_new_ascendc import ModelNew
from .model import Model

__all__ = ["ModelNew", "Model"]
PYEOF

    # ── 生成 setup.py ──
    cat > "${BUILD_DIR}/setup.py" <<PYEOF
import os
from setuptools import setup, find_packages

_pkg_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="${PACKAGE_NAME}",
    version="${VERSION}",
    description="MHC (Multi-Head Composition) AscendC kernel operator",
    author="Huawei Technologies Co., Ltd.",
    python_requires=">=3.9",
    packages=["${PACKAGE_NAME}"],
    package_dir={"${PACKAGE_NAME}": "${PACKAGE_NAME}"},
    package_data={
        "${PACKAGE_NAME}": [
            "*.so",
            "*.json",
        ],
    },
    include_package_data=True,
    install_requires=[],
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
    ],
)
PYEOF

    # ── 生成 MANIFEST.in ──
    cat > "${BUILD_DIR}/MANIFEST.in" <<EOF
recursive-include ${PACKAGE_NAME} *.so
recursive-include ${PACKAGE_NAME} *.json
include ${PACKAGE_NAME}/*.py
EOF

    # ── 生成 pyproject.toml ──
    cat > "${BUILD_DIR}/pyproject.toml" <<EOF
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"
EOF

    # ── 构建 ──
    mkdir -p "${DIST_DIR}"
    cd "${BUILD_DIR}"
    python -m build --wheel --outdir "${DIST_DIR}" 2>&1 || {
        # 如果 python -m build 不可用，降级到 setuptools
        warn "python -m build 失败，尝试 setuptools 直接构建..."
        python setup.py bdist_wheel --dist-dir "${DIST_DIR}"
    }

    cd "${PROJECT_DIR}"

    # 确认产物
    local whl
    whl="$(ls "${DIST_DIR}/${PACKAGE_NAME}"*.whl 2>/dev/null | head -1)"
    if [[ -z "${whl}" ]]; then
        error "WHL 构建失败"; exit 1
    fi

    info "WHL 包已生成: ${whl}"
    echo "  $(du -h "${whl}" | cut -f1)"
}

# ── 安装 + 验证 ─────────────────────────────────────────────────────────────
install_and_verify() {
    local whl
    whl="$(ls "${DIST_DIR}/${PACKAGE_NAME}"*.whl 2>/dev/null | head -1)"
    if [[ -z "${whl}" ]]; then
        error "未找到 WHL 文件，请先构建"; exit 1
    fi

    step "安装 WHL..."
    pip install "${whl}" --force-reinstall

    step "验证导入..."
    python -c "
import ${PACKAGE_NAME}
from ${PACKAGE_NAME} import ModelNew, Model
print('import OK')
print(f'  ModelNew: {ModelNew}')
print(f'  Model:    {Model}')
" || {
        error "导入验证失败"; exit 1
    }

    step "验证 torch.ops 注册..."
    python -c "
import torch
import ${PACKAGE_NAME}
# 确认 torch.ops.npu.mhc_pre_cmhc 已注册
op = getattr(torch.ops.npu, 'mhc_pre_cmhc', None)
if op is None:
    raise RuntimeError('torch.ops.npu.mhc_pre_cmhc 未注册')
print(f'torch.ops.npu.mhc_pre_cmhc OK: {op}')
" || {
        error "torch.ops 注册验证失败"; exit 1
    }

    info "安装验证通过 ✓"
}

# ── 构建 Run 包 ─────────────────────────────────────────────────────────────
build_run() {
    step "构建 Run 包 (tar.gz)..."

    local run_dir="${BUILD_DIR}/run/${PACKAGE_NAME}"
    rm -rf "${run_dir}"
    mkdir -p "${run_dir}"

    # 1. 复制 Python 模块
    cp "${PROJECT_DIR}/model_new_ascendc.py" "${run_dir}/"
    cp "${PROJECT_DIR}/model.py"             "${run_dir}/"
    cp "${PROJECT_DIR}/compare_precision.py" "${run_dir}/" 2>/dev/null || true
    cp "${PROJECT_DIR}/mhc_pre_sinkhorn.json" "${run_dir}/" 2>/dev/null || true

    # 2. 复制 binding .so + setup.py 源码 (方便用户重新编译)
    mkdir -p "${run_dir}/binding"
    cp "${PROJECT_DIR}/binding"/*.so     "${run_dir}/binding/" 2>/dev/null || true
    cp "${PROJECT_DIR}/binding/setup.py"  "${run_dir}/binding/"
    cp "${PROJECT_DIR}/binding"/*.cpp     "${run_dir}/binding/"
    cp "${PROJECT_DIR}/binding"/*.h       "${run_dir}/binding/"
    mkdir -p "${run_dir}/binding/include"
    cp -r "${PROJECT_DIR}/binding/include/"* "${run_dir}/binding/include/" 2>/dev/null || true

    # 3. 生成 __init__.py
    cat > "${run_dir}/__init__.py" <<'PYEOF'
"""mhc_pre_cmhc run package — drop-in module (no pip install needed)."""
from pathlib import Path
import torch

_pkg_dir = Path(__file__).resolve().parent
_so_files = list(_pkg_dir.glob("binding/mhc_pre_cmhc_torch_ops*.so"))
if _so_files:
    torch.ops.load_library(str(_so_files[0]))

from .model_new_ascendc import ModelNew
from .model import Model

__all__ = ["ModelNew", "Model"]
PYEOF

    # 4. 生成 README
    cat > "${run_dir}/README.txt" <<EOF
mhc_pre_cmhc Run 包
====================

用法:
  export PYTHONPATH="\$(pwd):\${PYTHONPATH}"
  python -c "import mhc_pre_cmhc; print(mhc_pre_cmhc.ModelNew)"

如需重新编译 binding:
  cd binding && python setup.py build_ext --inplace && cd ..

依赖:
  - PyTorch >= 2.0 (with NPU support / torch_npu)
  - CANN toolkit (Ascend NPU driver)
EOF

    # 5. 打包
    mkdir -p "${DIST_DIR}"
    local tarball="${DIST_DIR}/${PACKAGE_NAME}-run-${VERSION}.tar.gz"
    cd "${BUILD_DIR}/run"
    tar czf "${tarball}" "${PACKAGE_NAME}/"
    cd "${PROJECT_DIR}"

    info "Run 包已生成: ${tarball}"
    echo "  $(du -h "${tarball}" | cut -f1)"
}

# ── 清理 ────────────────────────────────────────────────────────────────────
clean() {
    step "清理打包产物..."
    rm -rf "${BUILD_DIR}"
    rm -rf "${DIST_DIR}"
    rm -f "${PROJECT_DIR}/binding"/*.so
    rm -rf "${PROJECT_DIR}/binding/build"
    info "清理完成"
}

# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

MODE="${1:-whl}"

case "${MODE}" in
    --help|-h)
        usage; exit 0
        ;;
    --clean)
        clean; exit 0
        ;;
    --install)
        preflight
        build_binding
        build_whl
        install_and_verify
        ;;
    --run)
        preflight
        build_binding
        build_run
        ;;
    whl|--whl|"")
        preflight
        build_binding
        build_whl
        ;;
    *)
        error "未知模式: ${MODE}"; usage; exit 1
        ;;
esac

echo ""
info "===== 打包完成 ====="
ls -lh "${DIST_DIR}/" 2>/dev/null || true
