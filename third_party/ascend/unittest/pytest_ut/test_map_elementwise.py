# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch_npu
import triton
import triton.language as tl
import test_common


@triton.jit
def compare_scalar(x, y):
    if x < y:
        return -1
    elif x == y:
        return 0
    else:
        return 1


@triton.jit
def divmod_pack2_scalar(a0, a1, b0, b1):
    return a0 // b0, a1 // b1, a0 % b0, a1 % b1


@triton.jit
def map_elementwise_compare_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    out = tl.map_elementwise(compare_scalar, x, y)
    tl.store(out_ptr + offsets, out)


@triton.jit
def map_elementwise_divmod_pack2_kernel(a_ptr, b_ptr, q_ptr, r_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    q, r = tl.map_elementwise(divmod_pack2_scalar, a, b, pack=2)
    tl.store(q_ptr + offsets, q)
    tl.store(r_ptr + offsets, r)


def test_map_elementwise_compare_control_flow():
    x_cpu = torch.tensor([-3, -1, 0, 2, 5, -8, 9, 0], dtype=torch.int32)
    y_cpu = torch.tensor([-4, 1, 0, 1, 7, -8, 3, 2], dtype=torch.int32)
    minus_one = torch.full_like(x_cpu, -1)
    zero = torch.zeros_like(x_cpu)
    one = torch.ones_like(x_cpu)
    ref = torch.where(x_cpu < y_cpu, minus_one, torch.where(x_cpu == y_cpu, zero, one))

    x = x_cpu.npu()
    y = y_cpu.npu()
    out = torch.empty_like(x)
    map_elementwise_compare_kernel[1, 1, 1](x, y, out, x_cpu.numel())

    test_common.validate_cmp("int32", out, ref)


def test_map_elementwise_divmod_pack2():
    a_cpu = torch.tensor([7, -7, 7, -7, 8, -8, 1, -1], dtype=torch.int32)
    b_cpu = torch.tensor([3, 3, -3, -3, 3, 3, 2, 2], dtype=torch.int32)
    q_ref = torch.div(a_cpu, b_cpu, rounding_mode="trunc")
    r_ref = a_cpu - q_ref * b_cpu

    a = a_cpu.npu()
    b = b_cpu.npu()
    q = torch.empty_like(a)
    r = torch.empty_like(a)
    map_elementwise_divmod_pack2_kernel[1, 1, 1](a, b, q, r, a_cpu.numel())

    test_common.validate_cmp("int32", q, q_ref)
    test_common.validate_cmp("int32", r, r_ref)
