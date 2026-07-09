"""
mhc_pre_cmhc_backward torch.ops plugin — PyTorch-native, no CANN dependencies.
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

LOCAL = os.path.dirname(os.path.abspath(__file__))

setup(
    name="mhc_pre_cmhc_backward_torch_ops",
    ext_modules=[
        CppExtension(
            name="mhc_pre_cmhc_backward_torch_ops",
            sources=["register.cpp", "mhc_pre_cmhc_backward_torch_host.cpp"],
            include_dirs=[LOCAL],
            extra_compile_args=["-std=c++17", "-O2", "-fPIC"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
