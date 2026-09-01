# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("deinterleave"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]


def vec_deinterleave(block_M, block_N, channel_nums, dtype="float16"):
    BLOCK_SIZE = 1
    block_N_half = block_N // channel_nums

    @T.prim_func
    def sliceDeinterleaveDev(
        A: T.Tensor((block_M, block_N), dtype),
        O0: T.Tensor((block_M, block_N_half), dtype),
        O1: T.Tensor((block_M, block_N_half), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            O0_VEC = T.alloc_shared((block_M, block_N_half), dtype)
            O1_VEC = T.alloc_shared((block_M, block_N_half), dtype)
            T.copy(A, A_VEC)
            T.npuir_deinterleave(
                A_VEC[:block_M, :block_N],
                O0_VEC[:block_M, :block_N_half],
                O1_VEC[:block_M, :block_N_half],
                channel_nums=channel_nums,
            )
            T.copy(O0_VEC, O0)
            T.copy(O1_VEC, O1)

    return sliceDeinterleaveDev


def vec_deinterleave_partial_dst(block_M, block_N, channel_nums, dtype="float16"):
    BLOCK_SIZE = 1
    block_N_half = block_N // channel_nums
    dst_offset_n = block_N // 4

    @T.prim_func
    def sliceDeinterleavePartialDstDev(
        A: T.Tensor((block_M, block_N), dtype),
        O0: T.Tensor((block_M, block_N), dtype),
        O1: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            O0_VEC = T.alloc_shared((block_M, block_N), dtype)
            O1_VEC = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A, A_VEC)
            T.copy(O0, O0_VEC)
            T.copy(O1, O1_VEC)
            T.npuir_deinterleave(
                A_VEC[:block_M, :block_N],
                O0_VEC[:block_M, dst_offset_n : dst_offset_n + block_N_half],
                O1_VEC[:block_M, dst_offset_n : dst_offset_n + block_N_half],
                channel_nums=channel_nums,
            )
            T.copy(O0_VEC, O0)
            T.copy(O1_VEC, O1)

    return sliceDeinterleavePartialDstDev


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_deinterleave_dev(dtype):
    M, N = 32, 32
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    O0 = gen_tensor((M, N // 2), dtype, kind="zeros")
    O1 = gen_tensor((M, N // 2), dtype, kind="zeros")
    ref_O0 = A.cpu()[:, 0::2]
    ref_O1 = A.cpu()[:, 1::2]

    func = vec_deinterleave(32, 32, channel_nums=2)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, O0, O1)

    assert_close(O0.cpu(), ref_O0, dtype=dtype, rtol=1e-2, atol=1e-2)
    assert_close(O1.cpu(), ref_O1, dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_deinterleave_partial_dst_dev(dtype):
    M, N = 32, 32
    offset_n = N // 4
    half_n = N // 2
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    O0 = gen_tensor((M, N), dtype, kind="zeros")
    O1 = gen_tensor((M, N), dtype, kind="zeros")
    ref_O0 = torch.zeros_like(O0.cpu())
    ref_O1 = torch.zeros_like(O1.cpu())
    ref_O0[:, offset_n : offset_n + half_n] = A.cpu()[:, 0::2]
    ref_O1[:, offset_n : offset_n + half_n] = A.cpu()[:, 1::2]

    func = vec_deinterleave_partial_dst(32, 32, channel_nums=2)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, O0, O1)

    assert_close(O0.cpu(), ref_O0, dtype=dtype, rtol=1e-2, atol=1e-2)
    assert_close(O1.cpu(), ref_O1, dtype=dtype, rtol=1e-2, atol=1e-2)
