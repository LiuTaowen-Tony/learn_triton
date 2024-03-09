import math
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

# This is a matmul kernel based on triton.ops.matmul
# It is modified to support rowwise quantized input and columnwise quantized weight
# It's purpose is fused matmul then dequantize
# It does support bias.

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


print("here")
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
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        *get_configs_io_bound(),
    ],
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
def _int8_matmul_block_rowwise_dequantize(A, B, C, bias, state_x_ptr, state_w_ptr, M, N, K, divfactor, has_bias : tl.constexpr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            stride_state: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
            ACC_TYPE: tl.constexpr
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
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

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    state_x_ptr = state_x_ptr + rm * stride_state
    state_w_ptr = state_w_ptr + rn * stride_state

    x_factor = tl.load(state_x_ptr)
    w_factor = tl.load(state_w_ptr)
    # acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for i in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - i * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.)
        result = tl.dot(a, b)
        
        # acc += result
        acc += (x_factor[:, None] * w_factor[None, :] * result)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    # acc = (w_factor * (x_factor * (acc * divfactor)))
    acc = acc.to(C.dtype.element_ty)

    if has_bias:
        bias = tl.load(bias + rn).to(C.dtype.element_ty)
        acc = acc + bias[None, :]

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)
print("there")


def int8_matmul_block_rowwise_dequantize(a, b, state_x, state_w, bias=None):
    divfactor = 1. / (127. * 127.)

    has_bias = 0 if bias is None else 1

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
    c = torch.empty((M, N), device=device, dtype=torch.float16)
    # accumulator types
    ACC_TYPE = tl.float32 #if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
    # launch int8_matmul_rowwise_dequantize kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    _int8_matmul_block_rowwise_dequantize[grid](a, b, c, bias, state_x, state_w, M, N, K, divfactor, has_bias,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    stride_state=ceil_div(K, 32),
                    GROUP_M=8, ACC_TYPE=ACC_TYPE)
    return c

import triton
import triton.language as tl

# TODO: autotune this better.
@triton.autotune(
        configs=[
            triton.Config({}, num_stages=1, num_warps=8),
            triton.Config({}, num_stages=2, num_warps=8),
            triton.Config({}, num_stages=4, num_warps=8),
            triton.Config({}, num_stages=8, num_warps=8),
            triton.Config({}, num_stages=1),
            triton.Config({}, num_stages=2),
            triton.Config({}, num_stages=4),
            triton.Config({}, num_stages=8),
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=['n_elements']
)
@triton.jit
def _quantize_block_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    K: tl.constexpr,
    n_elements
):
    pid = tl.program_id(axis=0)
    block_start = pid * K
    offsets = block_start + tl.arange(0, 32)
    output_maxs_offset = pid * tl.cdiv(K, 32)
    
    for _ in range(0, tl.cdiv(K, 32)):
        row_mask = offsets < block_start + K
        x = tl.load(x_ptr + offsets, mask=row_mask)
        abs_x = tl.abs(x)
        max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
        output = tl.math.llrint(127. * (x / max_val))
        tl.store(output_ptr + offsets, output, mask=row_mask)
        tl.store(output_maxs + output_maxs_offset, max_val)
        offsets += 32
        output_maxs_offset += 1

def ceil_div(n, d):
    return -(n // -d)

def quantize_block_rowwise(x: torch.Tensor):
    m, k = x.shape

    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_maxs = torch.empty(m, ceil_div(k, 32) , device=x.device, dtype=torch.float16)

    assert x.is_cuda and output.is_cuda
    grid = lambda meta: (x.shape[0],)
    n_elements = output.numel()
    _quantize_block_rowwise[grid](x, output, output_maxs,K=k, n_elements=n_elements)
    return output, output_maxs


# def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
#     output = torch.empty(*x.shape, device=x.device, dtype=torch.float16)

#     P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

#     assert x.is_cuda and output.is_cuda
#     n_elements = output.numel()
#     grid = lambda meta: (x.shape[0],)
#     _dequantize_rowwise[grid](x, state_x, output, 1./127, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
#     return output

# TODO: autotune this better.
@triton.autotune(
        configs=[
            triton.Config({}, num_stages=1, num_warps=8),
            triton.Config({}, num_stages=2, num_warps=8),
            triton.Config({}, num_stages=4, num_warps=8),
            triton.Config({}, num_stages=8, num_warps=8),
            triton.Config({}, num_stages=1),
            triton.Config({}, num_stages=2),
            triton.Config({}, num_stages=4),
            triton.Config({}, num_stages=8),
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=['n_elements']
)
@triton.jit
def _dequantize_block_rowwise(
    x_ptr,
    state_x,
    output_ptr,
    inv_127,
    K: tl.constexpr,
    n_elements,
):
    pid = tl.program_id(axis=0)
    block_start = pid * K
    offsets = block_start + tl.arange(0, 32)
    output_maxs_offset = pid * tl.cdiv(K, 32)
    
    for i in range(0, tl.cdiv(K, 32)):
        row_mask = offsets < block_start + K
        x = tl.load(x_ptr + offsets, mask=row_mask)
        max_val = tl.load(state_x + output_maxs_offset)
        output = max_val * x * inv_127
        tl.store(output_ptr + offsets, output, mask=row_mask)
        offsets += 32
        output_maxs_offset += 1

def dequantize_block_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.float16)
    assert x.is_cuda and output.is_cuda
    grid = lambda meta: (x.shape[0],)
    n_elements = output.numel()
    _dequantize_block_rowwise[grid](x, state_x, output, 1./127, K=x.shape[1], n_elements=n_elements)
    return output

def block_quantize_dequantize_is_close():
    w = 253
    h = 258
    a = torch.randn((w, h), device="cuda", dtype=torch.float16)
    y, state_x = quantize_block_rowwise(a)
    x = dequantize_block_rowwise(y, state_x)
    print(a)
    print(x)
    return torch.allclose(a, x)

# print(block_quantize_dequantize_is_close())



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[512 * i for i in range(2, 10)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    print("here")
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        state_x = None
        state_w = None
        a, state_x = quantize_block_rowwise(a)
        b, state_w = quantize_block_rowwise(b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: int8_matmul_block_rowwise_dequantize(a, b.t(), state_x.to(torch.int32), state_w.t().to(torch.int32)), quantiles=quantiles) 
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
# benchmark.run(show_plots=True, print_data=True)
