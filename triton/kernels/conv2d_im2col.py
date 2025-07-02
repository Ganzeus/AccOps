import torch
import triton
import triton.language as tl
from typing import Optional
device = 'cuda:0'
torch.cuda.set_device(device)  # 确保默认设备是device
torch.cuda.empty_cache()  # 清空未使用的 GPU 内存

@triton.jit
def gelu(input):
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input

dtype=torch.float16

@triton.autotune(
    configs=[
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 64, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 32, 'bsk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
        triton.Config({'bsy': 32, 'bsx': 64, 'bsk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 128, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 256, 'bsx': 128, 'bsk': 128, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 256, 'bsx': 64, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 64, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64, 'group_sz': 4}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 32, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=5, num_warps=2),
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 128, 'group_sz': 4}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 256, 'bsx': 128, 'bsk': 128, 'group_sz': 4}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 128, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, O_ptr,
    A_stride_height, A_stride_width,
    B_stride_batch,
    B_stride_height, B_stride_width,
    O_stride_batch,
    O_stride_height, O_stride_width,
    M, N, K,
    bias_ptr,
    add_bias: tl.constexpr,
    apply_activation: tl.constexpr,
    activation: tl.constexpr,
    bsx: tl.constexpr, bsy: tl.constexpr, bsk: tl.constexpr, group_sz: tl.constexpr
):
    """
    Grouped Matmul
    O = A x B
    """
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)

    num_row_programs = tl.num_programs(1)
    num_col_programs = tl.num_programs(2)

    row_idxnew, col_idxnew = tl.swizzle2d(row_idx, col_idx, num_row_programs, num_col_programs, group_sz)

    offset_batch = batch_idx * B_stride_batch

    acc = tl.zeros((bsy, bsx), dtype=tl.float32)    # accumulator

    for offset in range(0, K, bsk):
        offset_k = offset + tl.arange(0, bsk)

        # Read offsets from A_ptr
        offset_a = row_idxnew * bsy + tl.arange(0, bsy)
        offset_a = offset_a[:, None]*A_stride_height + offset_k[None, :]*A_stride_width  # bsy * bsk
        mask_a = row_idxnew * bsy + tl.arange(0, bsy)
        mask_a = (mask_a[:, None] < M) & (offset_k[None, :] < K)
        a = tl.load(A_ptr + offset_a, mask_a)

        # Read offset from B_ptr
        offset_b = col_idxnew * bsx + tl.arange(0, bsx)
        offset_b = offset_k[:, None]*B_stride_height + offset_b[None, :]*B_stride_width  # bsk * bsx
        mask_b = col_idxnew * bsx + tl.arange(0, bsx)
        mask_b = (offset_k[:, None] < K) & (mask_b[None, :] < N)
        b = tl.load(B_ptr + offset_batch + offset_b, mask_b)

        # acc += tl.dot(a, b, allow_tf32=True)  # triton old version
        acc = tl.dot(a, b, acc, allow_tf32=True)
        
        

    offset_batch_out = batch_idx * O_stride_batch
    offset_or = row_idxnew * bsy + tl.arange(0, bsy)
    offset_oc = col_idxnew * bsx + tl.arange(0, bsx)
    offset_o = offset_or[:, None]*O_stride_height+ offset_oc[None, :]*O_stride_width  # bsy * bsx
    mask_o = (offset_or[:, None] < M) & (offset_oc[None, :] < N)

    if add_bias:
        bias = tl.load(bias_ptr + offset_oc, offset_oc < N)
        acc += bias[None, :]

    if apply_activation:
        if activation == 'gelu':
            acc = gelu(acc)

    tl.store(O_ptr + offset_batch_out + offset_o, acc, mask_o)


def matmul_triton(A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor] = None, activation: Optional[str] = None) -> torch.Tensor:
    """
    Implements matrix multiplication between input matrix A and B
    
    Args:
        - A {torch.Tensor}: (M, K)
        - B {torch.Tensor}: (B, K, N)
        - bias {torch.Tensor}: Optionally add a bias to the ouput, shape (1, N)
        - activation {str}: Optionally apply activation to the ouput

    Returns:
        - {torch.Tensor}: Output tensor with (B, M, N)
    """
    assert len(B.shape) == 3, f"Second input matrix needs to have 3 dimensions (B, K, N) but {B.shape}"
    assert A.device == B.device and A.is_cuda, "Both matrix should be on GPU"

    if bias is not None:
        assert bias.is_cuda, "Bias is not on GPU"
        bias = bias.unsqueeze(0)
        assert bias.shape[1] == B.shape[2], "Bias shape does not match output feature dimension shape"

    if activation:
        assert activation in ["gelu"], f"Only GELU activation supported as of now! Provided: {activation}"

    M, K = A.shape
    batch_size, K, N = B.shape

    grid = lambda meta: (batch_size, triton.cdiv(M, meta["bsy"]), triton.cdiv(N, meta["bsx"]))

    O = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)

    matmul_kernel[grid](
        A, B, O,
        A_stride_height=A.stride(0), A_stride_width=A.stride(1),
        B_stride_batch=B.stride(0),
        B_stride_height=B.stride(1), B_stride_width=B.stride(2),
        O_stride_batch=O.stride(0),
        O_stride_height=O.stride(1), O_stride_width=O.stride(2),
        M=M, N=N, K=K,
        bias_ptr=bias,
        add_bias=True if bias is not None else False,
        activation=activation,
        apply_activation=True if activation else False
    )

    return O

def unroll(X, K, stride=1):
    B, C, H, W = X.shape
    H_out, W_out = (H - K) // stride + 1, (W - K) // stride + 1
    X_unroll = torch.zeros(B, C*K*K, H_out*W_out, dtype = X.dtype, device=X.device)
    for c in range(C):
        col_idx = 0
        h = 0
        for _ in range(H_out):
            w = 0
            for _ in range(W_out):
                # 将X(h, w)开始的K*K个元素存入X_unroll
                for i in range(K):
                    for j in range(K):
                        row_idx = i*K+j + c*K*K         # 当前部分的第几行+当前部分的起始行
                        X_unroll[:, row_idx, col_idx] = X[:, c, h+i, w+j]
                col_idx += 1            # 进入X_unroll的下一列
                w += stride
            h += stride
    return X_unroll

def conv2d_im2col(
    input: torch.Tensor,		# (batch_size, Cin, h, w)
    kernel: torch.Tensor,		# (Cout, Cin, K, K)
    stride: int=1,
    bias: Optional[torch.Tensor] = None, 		# (Cout, 1)
) -> torch.Tensor:
    assert input.is_cuda and kernel.is_cuda, 'Input or kernel is not on GPU'
    assert len(input.shape) == 4, f'Input needs to be 4 dimensional, provided: {input.shape}'
    assert len(kernel.shape) == 4, f'Kernel size needs to be 4 dimensional, provided: {kernel.shape}'
    assert kernel.shape[2] == kernel.shape[3], f'kernel height and width should be equal'

    if bias is not None:
        assert bias.shape[0] == kernel.shape[0], f'Bias dimension should be same as the kernel 1st dimension'
        
    batch_size, Cin, H, W = input.shape
    num_kernels, kernel_depth, K, K = kernel.shape

    assert (H - K) % stride  == 0
    assert (W - K) % stride == 0
    assert Cin == kernel_depth, f"Kernel channel depth ({kernel_depth}) and input channel depth ({Cin}) should be the same"

    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    # output = torch.empty((batch_size, num_kernels, H//kernel_height, W//kernel_width), device=device, dtype=dtype)			# kernel步长为kernel边长(即每次计算不会重叠)
    
    input_unroll = unroll(input, K, stride=stride)           # (batch_size, Cin*K*K, H_out*W_out)
    kernel_unroll = kernel.view(num_kernels, -1)        # (Cout, Cin*K*K)

    # M = num_kernels
    # N = H_out * W_out
    # output_unroll = torch.empty(batch_size, M, N)   # (batch_size, Cout, H_out*W_out)

    # bs = 16
    # grid = (batch_size, triton.cdiv(M, bs), triton.cdiv(N, bs))

    # matmul_kernel(kernel_unroll, input_unroll, output_unroll,
    #               kernel_unroll.stride(0), kernel_unroll.stride(1), kernel_unroll.stride(2),
    #               input_unroll.stride(0), input_unroll.stride(1), input_unroll.stride(2),
    #               output_unroll.stride(0), output_unroll.stride(1), output_unroll.stride(2),
    #               M, N, Cin*K*K,
    #               bias_ptr=bias,
    #               add_bias=True if bias is not None else False,
    #               activation=None,
    #               apply_activation=False)

    output_unroll = matmul_triton(kernel_unroll, input_unroll)
    output = output_unroll.view(batch_size, num_kernels, H_out, W_out)
    
    return output

if __name__ == "__main__":
    import torch.nn.functional as F
    X1 = torch.tensor([
        [[1, 2, 0],
        [1, 1, 3],
        [0, 2, 2]],
        [[0, 2, 1],
        [0, 3, 2],
        [1, 1, 0]],
        [[1, 2, 1],
        [0, 1, 3],
        [3, 3, 2]]
    ], dtype=torch.float16, device=device)
    X2 = torch.tensor([
        [[0, 2, 1],
        [0, 3, 2],
        [1, 1, 0]],
        [[1, 2, 0],
        [1, 1, 3],
        [0, 2, 2]],
        [[1, 2, 1],
        [0, 1, 3],
        [3, 3, 2]]
    ], dtype=torch.float16, device=device)
    inputs = torch.stack((X1, X2), dim=0)
    kernel = torch.tensor([1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 2, 1, 2, 1, 1, 2, 2, 0], dtype = torch.float16, device=device).view(2, 3, 2, 2)
    out_torch = F.conv2d(inputs, kernel)
    print(f"output_torch:\n{out_torch}")
    out_triton = conv2d_im2col(inputs, kernel,stride=1, bias=None)
    print(f"output_triton:\n{out_triton}")