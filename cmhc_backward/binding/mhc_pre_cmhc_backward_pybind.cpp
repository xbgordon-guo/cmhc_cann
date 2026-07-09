/*
 * mhc_pre_cmhc_backward pybind11 wrapper — direct callable from Python (CPU or NPU).
 *
 * Permutation matrices are passed as input (pre-computed on Python side).
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "ops.h"

namespace py = pybind11;

std::vector<at::Tensor> mhc_pre_cmhc_backward_py(
    const at::Tensor &grad_h_in,
    const at::Tensor &grad_h_post,
    const at::Tensor &grad_h_res,
    const at::Tensor &x,
    const at::Tensor &phi,
    const at::Tensor &alpha,
    const at::Tensor &bias,
    const at::Tensor &h_pre,
    const at::Tensor &hc_before_norm,
    const at::Tensor &inv_rms,
    const at::Tensor &perm_mats,
    double hc_eps)
{
    return mhc_pre_cmhc_backward_binding::mhc_pre_cmhc_backward(
        grad_h_in, grad_h_post, grad_h_res,
        x, phi, alpha, bias,
        h_pre, hc_before_norm, inv_rms,
        perm_mats,
        hc_eps);
}

PYBIND11_MODULE(mhc_pre_cmhc_backward_pybind, m)
{
    m.def("mhc_pre_cmhc_backward", &mhc_pre_cmhc_backward_py, "cmhc backward pass");
}
