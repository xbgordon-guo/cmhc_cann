import os
from setuptools import setup, find_packages

_pkg_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="mhc_pre_cmhc",
    version="1.0.0",
    description="MHC (Multi-Head Composition) AscendC kernel operator",
    author="Huawei Technologies Co., Ltd.",
    python_requires=">=3.9",
    packages=["mhc_pre_cmhc"],
    package_dir={"mhc_pre_cmhc": "mhc_pre_cmhc"},
    package_data={
        "mhc_pre_cmhc": [
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
