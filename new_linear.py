import numpy as np
import time
import torch
import triton
import triton.language as tl

import torch

import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                                      num_stages=num_stages, num_warps=num_warps))
                    # split_k
                    for split_k in [2, 4, 8, 16]:
                        configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                                                     num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ] + get_configs_io_bound(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
            ACC_TYPE: tl.constexpr
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    scale = tl.zeros((1,), dtype=tl.float16)
    scale += 16.
    for k in range(K, 0, -BLOCK_K * SPLIT_K):
        if EVEN_K:
            a = tl.load(A).to(tl.float8e4, bitcast=True)
            b = tl.load(B).to(tl.float8e4, bitcast=True)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.).to(tl.float8e4, bitcast=True)
            b = tl.load(B, mask=rk[:, None] < k, other=0.).to(tl.float8e4, bitcast=True)

        ascaled = (a * scale)
        amax = tl.atomic_max(tl.math.abs(ascaled))
        bscaled = (b * scale)
        result = tl.dot(ascaled , bscaled, out_dtype=tl.float16)
        result = (result / scale / scale).to(ACC_TYPE)

        acc += result
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(tl.float8e4).to(C.dtype.element_ty, bitcast=True)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    _locks = dict()

    @staticmethod
    def _call(a, b):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        # accumulator types
        ACC_TYPE = tl.float16 #if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        # grid = (1, 1, 1)
        knl = _kernel[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1),
                      b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1),
                      GROUP_M=8, ACC_TYPE=ACC_TYPE)
        for k, v in knl.asm.items():
            with open(f'kernel.asm{k}', 'w') as f:
                f.write(str(v) if isinstance(v, bytes) else v)
        return c

    @staticmethod
    def forward(ctx, a, b):
        return _matmul._call(a, b)


matmul = _matmul.apply


@triton.jit
def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = input
    tl.store(output_ptr + offsets, output, mask=mask)

def fp16_2_fp8(f16_tensor):
    n_elements = f16_tensor.numel()
    f8_output_tensor = torch.empty_like(f16_tensor, dtype=torch.int8)
    f8_output = triton.reinterpret(f8_output_tensor, tl.float8e4)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f16_tensor, f8_output, n_elements, BLOCK_SIZE=1024)
    return f8_output_tensor

def fp8_2_fp16(f8_tensor):
    n_elements = f8_tensor.numel()
    f8_triton = triton.reinterpret(f8_tensor, tl.float8e4)

    f16_output_tensor = torch.empty_like(f8_tensor, dtype=torch.float16)
    f16_output = triton.reinterpret(f16_output_tensor, tl.float16)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](f8_triton, f16_output, n_elements, BLOCK_SIZE=1024)
    return f16_output_tensor

def test_matmul():
    torch.manual_seed(12)
    M = 512*16
    N = 768*16
    K = 128
    AT = False
    BT = True
    DTYPE=torch.float16

    a = 0.01 * torch.ones((K, M) if AT else (M, K), device="cuda", dtype=torch.float16)
    b = 0.05 * torch.ones((N, K) if BT else (K, N), device="cuda", dtype=torch.float16)
    a_fp8 = fp16_2_fp8(a)
    b_fp8 = fp16_2_fp8(b)

    a = a.t() if AT else a
    b = b.t() if BT else b
    a_fp8 = a_fp8.t() if AT else a_fp8
    b_fp8 = b_fp8.t() if BT else b_fp8


    a_int8 = torch.ones((K, M) if AT else (M, K), device="cuda", dtype=torch.int8)
    b_int8 = torch.ones((N, K) if BT else (K, N), device="cuda", dtype=torch.int8)
    a_int8 = a_int8.t() if AT else a_int8
    b_int8 = b_int8.t() if BT else b_int8

    th_c = torch.matmul(a, b)
    print(f"pytorch matmul fp16: {th_c}")

    # matmul = triton.ops.matmul
    c_fp8 = matmul(a_fp8, b_fp8)
    c_fp16 = fp8_2_fp16(c_fp8)
    print(f"trition matmul fp8: {c_fp16}")

    ################### 1. triton matmul fp8 ###################
    for _ in range(8):
        c_fp8 = matmul(a_fp8, b_fp8)

    triton_matmul_fp8_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_matmul_fp8_graph):
        c_fp8 = matmul(a_fp8, b_fp8)

    triton_matmul_fp8_graph.replay()

    repeat = 16

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_matmul_fp8_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"triton fp8 matmul: {(end - start) / repeat * 1000:.3f} ms")

    ################### 2. triton matmul int8 ###################
    for _ in range(8):
        c_int8 = triton.ops.matmul(a_int8, b_int8)

    triton_matmul_int8_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_matmul_int8_graph):
        c_int8 = triton.ops.matmul(a_int8, b_int8)

    triton_matmul_int8_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_matmul_int8_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"triton int8 matmul: {(end - start) / repeat * 1000:.3f} ms")

    # ################### 3. triton matmul fp16 ###################
    for _ in range(8):
        c_fp16 = triton.ops.matmul(a, b)

    triton_matmul_fp16_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_matmul_fp16_graph):
        c_fp16 = triton.ops.matmul(a, b)

    triton_matmul_fp16_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"triton fp16 matmul: {(end - start) / repeat * 1000:.3f} ms")

    # ################### 4. torch matmul fp16 ###################
    for _ in range(8):
        th_c = torch.matmul(a, b)

    torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_matmul_fp16_graph):
        th_c = torch.matmul(a, b)

    torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"torch fp16 matmul: {(end - start) / repeat * 1000:.3f} ms")


test_matmul()
