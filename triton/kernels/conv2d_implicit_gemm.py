import torch
import triton
import triton.language as tl
import pytest
import math


def get_triton_configs():
    """返回卷积操作的Triton配置"""
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, 
                      num_stages=3, num_warps=8),
    ]


@triton.autotune(
    configs=get_triton_configs(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'stride_h', 'stride_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def conv2d_forward_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr, 
    N, C, H, W, K, P, Q, R, S, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):             
    """卷积前向传播的Triton核心函数"""
    # 计算程序ID和相关索引
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算GEMM索引和输出索引
    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    # 提取输出特征图位置信息
    n = gemm_i // (P * Q)
    npq_residual = gemm_i % (P * Q)
    p = npq_residual // Q
    q = npq_residual % Q
    k = gemm_j

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 遍历并计算卷积
    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        # 计算输入和滤波器索引
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        c = gemm_k // (R * S)
        crs_residual = gemm_k % (R * S)
        r = crs_residual // S
        s = crs_residual % S
        
        # 计算输入特征图位置
        h = p[:, None] * stride_h + r[None, :] * dil_h - pad_h
        w = q[:, None] * stride_w + s[None, :] * dil_w - pad_w

        # 创建掩码确保索引有效
        mask_input = (h >= 0) & (h < H) & (w >= 0) & (w < W)
        mask_weight = (r[:, None] < R) & (s[:, None] < S) & (c[:, None] < C)

        # 计算内存偏移量
        offs_input = n[:, None] * C * H * W + c[None, :] * H * W + h * W + w
        offs_weight = k[None, :] * C * R * S + c[:, None] * R * S + r[:, None] * S + s[:, None]

        # 加载数据
        input_ptrs = input_ptr + offs_input
        weight_ptrs = weight_ptr + offs_weight
        
        input_data = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_data = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

        # 执行矩阵乘法
        acc = tl.dot(input_data, weight_data, acc)

    # 添加偏置（如果存在）
    if bias_ptr is not None:
        offs_bias = k[None, :]
        bias_ptrs = bias_ptr + offs_bias
        bias_data = tl.load(bias_ptrs)
        acc = acc + bias_data

    # 转换为FP16
    acc = acc.to(tl.float16)

    # 存储结果为 [N, K, P, Q] 格式
    offs_nkpq = n[:, None] * K * P * Q + k[None, :] * P * Q + p[:, None] * Q + q[:, None]
    mask_nkpq = (n[:, None] < N) & (k[None, :] < K) & (p[:, None] < P) & (q[:, None] < Q)

    output_ptrs = output_ptr + offs_nkpq
    tl.store(output_ptrs, acc, mask=mask_nkpq)
        

def conv2d_forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                   stride, padding, dilation):
    """卷积前向传播函数"""
    # 获取形状参数
    N, C, H, W = input.shape
    K, C, R, S = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # 计算输出特征图尺寸
    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // stride_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // stride_w + 1

    # 定义GEMM参数
    GEMM_M = N * P * Q  # 输出元素总数
    GEMM_N = K          # 输出通道数
    GEMM_K = C * R * S  # 每个输出元素的卷积操作数
    
    # 创建输出张量
    output = torch.zeros((N, K, P, Q), dtype=input.dtype, device=input.device)
    
    # 定义计算网格
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * 
                         triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    
    # 调用kernel
    conv2d_forward_kernel[grid](
        output, input, weight, bias, 
        N, C, H, W, K, P, Q, R, S, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
    )

    return output


@triton.autotune(
    configs=get_triton_configs(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'stride_h', 'stride_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def conv2d_input_grad_kernel(
    dinput_ptr, doutput_ptr, weight_ptr, 
    N, C, H, W, K, P, Q, R, S, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):             
    """输入梯度反向传播核心函数"""
    # 计算程序ID和索引
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算GEMM索引和输入梯度索引
    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    n = gemm_i // (H * W)
    nhw_residual = gemm_i % (H * W)
    h = nhw_residual // W
    w = nhw_residual % W
    c = gemm_j

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 遍历并计算输入梯度
    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        k = gemm_k // (R * S)
        krs_residual = gemm_k % (R * S)
        r = krs_residual // S
        s = krs_residual % S

        # 计算输出特征图位置
        h_tmp = (h[:, None] + pad_h - r[None, :] * dil_h)
        p = h_tmp // stride_h
        mask_p = (h_tmp % stride_h == 0) & (p >= 0) & (p < P)
        
        w_tmp = (w[:, None] + pad_w - s[None, :] * dil_w)
        q = w_tmp // stride_w
        mask_q = (w_tmp % stride_w == 0) & (q >= 0) & (q < Q)
        
        # 创建掩码
        mask_doutput = (n[:, None] < N) & (k[None, :] < K) & mask_p & mask_q
        mask_weight = (k[:, None] < K) & (c[None, :] < C) & (r[:, None] < R) & (s[:, None] < S)

        # 计算偏移量
        offs_doutput = n[:, None] * K * P * Q + k[None, :] * P * Q + p * Q + q
        offs_weight = k[:, None] * C * R * S + c[None, :] * R * S + r[:, None] * S + s[:, None]
        
        # 加载数据
        doutput_ptrs = doutput_ptr + offs_doutput
        weight_ptrs = weight_ptr + offs_weight

        doutput_data = tl.load(doutput_ptrs, mask=mask_doutput, other=0.0)
        weight_data = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

        # 执行矩阵乘法
        acc = tl.dot(doutput_data, weight_data, acc)

    # 转换为FP16
    acc = acc.to(tl.float16)

    # 存储结果为 [N, C, H, W] 格式
    offs_nchw = n[:, None] * C * H * W + c[None, :] * H * W + h[:, None] * W + w[:, None]
    mask_nchw = (n[:, None] < N) & (c[None, :] < C) & (h[:, None] < H) & (w[:, None] < W)
    dinput_ptrs = dinput_ptr + offs_nchw
    tl.store(dinput_ptrs, acc, mask=mask_nchw)


def conv2d_input_backward(doutput, weight, N, C, H, W, K, R, S, stride, padding, dilation):
    """计算卷积的输入梯度"""
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # 计算输出特征图尺寸
    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // stride_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // stride_w + 1

    # 定义GEMM参数
    GEMM_M = N * H * W  # 输入元素总数
    GEMM_N = C          # 输入通道数
    GEMM_K = K * R * S  # 每个输入元素的梯度操作数

    # 创建输入梯度张量
    dinput = torch.zeros((N, C, H, W), dtype=doutput.dtype, device=doutput.device)
    
    # 定义计算网格
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * 
                         triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    
    # 调用kernel
    conv2d_input_grad_kernel[grid](
        dinput, doutput, weight, 
        N, C, H, W, K, P, Q, R, S, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
    )

    return dinput


@triton.autotune(
    configs=get_triton_configs(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'stride_h', 'stride_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def conv2d_weight_grad_kernel(
    dweight_ptr, doutput_ptr, input_ptr, 
    N, C, H, W, K, P, Q, R, S, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):             
    """权重梯度反向传播核心函数"""
    # 计算程序ID和索引
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算GEMM索引和权重梯度索引
    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    k = gemm_i
    c = gemm_j // (R * S)
    crs_residual = gemm_j % (R * S)
    r = crs_residual // S
    s = crs_residual % S

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 遍历并计算权重梯度
    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        n = gemm_k // (P * Q)
        npq_residual = gemm_k % (P * Q)
        p = npq_residual // Q
        q = npq_residual % Q

        # 计算偏移量和掩码
        offs_doutput = n[None, :] * K * P * Q + k[:, None] * P * Q + p[None, :] * Q + q[None, :]
        
        # 计算输入特征图位置
        h = p[:, None] * stride_h + r[None, :] * dil_h - pad_h
        w = q[:, None] * stride_w + s[None, :] * dil_w - pad_w
        
        # 计算偏移量
        offs_input = n[:, None] * C * H * W + c[None, :] * H * W + h * W + w

        # 创建掩码
        mask_doutput = (n[None, :] < N) & (k[:, None] < K) & (p[None, :] < P) & (q[None, :] < Q)
        mask_input = (n[:, None] < N) & (c[None, :] < C) & (h < H) & (w < W) & (h >= 0) & (w >= 0)

        # 加载数据
        doutput_ptrs = doutput_ptr + offs_doutput
        input_ptrs = input_ptr + offs_input

        doutput_data = tl.load(doutput_ptrs, mask=mask_doutput, other=0.0)
        input_data = tl.load(input_ptrs, mask=mask_input, other=0.0)

        # 执行矩阵乘法
        acc = tl.dot(doutput_data, input_data, acc)

    # 转换为FP16
    acc = acc.to(tl.float16)
    
    # 存储结果
    offs_weight = gemm_i[:, None] * GEMM_N + gemm_j[None, :]
    mask_weight = (gemm_i[:, None] < GEMM_M) & (gemm_j[None, :] < GEMM_N)
    dweight_ptrs = dweight_ptr + offs_weight
    tl.store(dweight_ptrs, acc, mask=mask_weight)


def conv2d_weight_backward(doutput, input, N, C, H, W, K, R, S, stride, padding, dilation):
    """计算卷积的权重梯度"""
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # 计算输出特征图尺寸
    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // stride_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // stride_w + 1

    # 定义GEMM参数
    GEMM_M = K           # 输出通道数
    GEMM_N = C * R * S   # 每个输出通道的权重元素数
    GEMM_K = N * P * Q   # 每个权重元素的梯度操作数

    # 创建权重梯度张量
    dweight = torch.zeros((K, C, R, S), dtype=doutput.dtype, device=doutput.device)
    
    # 定义计算网格
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * 
                         triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    
    # 调用kernel
    conv2d_weight_grad_kernel[grid](
        dweight, doutput, input, 
        N, C, H, W, K, P, Q, R, S, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
    )
    
    return dweight


@triton.jit
def conv2d_bias_grad_kernel(dbias, doutput_ptr, N, K, P, Q, BLOCK_SIZE: tl.constexpr):
    """偏置梯度反向传播核心函数"""
    # 获取输出通道索引
    k = tl.program_id(0)

    # 计算P*Q的偏移量和掩码
    offs_pq = tl.arange(0, BLOCK_SIZE)
    mask_pq = offs_pq < P * Q

    # 偏置梯度的偏移量
    offs_k = k + tl.arange(0, 1)

    # 初始化累加器
    acc = tl.zeros((1, ), dtype=tl.float32)

    # 遍历批次维度
    for idx_n in range(0, N):
        offs_nkpq = idx_n * K * P * Q + k * P * Q + offs_pq
        doutput_ptrs = doutput_ptr + offs_nkpq
        doutput_data = tl.load(doutput_ptrs, mask=mask_pq, other=0.0)
        acc = acc + tl.sum(doutput_data)

    # 转换为FP16
    acc = acc.to(tl.float16)

    # 存储结果
    dbias_ptrs = dbias + offs_k
    tl.store(dbias_ptrs, acc)


def conv2d_bias_backward(doutput):
    """计算卷积的偏置梯度"""
    N, K, P, Q = doutput.shape
    BLOCK_SIZE = triton.next_power_of_2(P * Q)
    dbias = torch.zeros((K), dtype=doutput.dtype, device=doutput.device)
    
    # 调用kernel
    conv2d_bias_grad_kernel[K, ](dbias, doutput, N, K, P, Q, BLOCK_SIZE)
    
    return dbias


class ConvTritonFunc(torch.autograd.Function):
    """使用Triton实现的卷积操作的自动微分函数"""
    
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        """卷积前向传播"""
        output = conv2d_forward(input, weight, bias, stride, padding, dilation)

        # 保存用于反向传播的上下文
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bias_requires_grad = bias is not None and bias.requires_grad

        return output
    
    @staticmethod
    def backward(ctx, doutput):
        """卷积反向传播"""
        input, weight = ctx.saved_tensors
        N, C, H, W = input.shape
        K, C, R, S = weight.shape
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        # 计算输入梯度
        dinput = None
        if input.requires_grad:
            dinput = conv2d_input_backward(doutput, weight, N, C, H, W, K, R, S, 
                                         stride, padding, dilation)

        # 计算权重梯度
        dweight = None
        if weight.requires_grad:
            dweight = conv2d_weight_backward(doutput, input, N, C, H, W, K, R, S, 
                                           stride, padding, dilation)

        # 计算偏置梯度
        dbias = None
        if ctx.bias_requires_grad:
            dbias = conv2d_bias_backward(doutput)
            
        return dinput, dweight, dbias, None, None, None


class Conv2d_triton(torch.nn.Module):
    """使用Triton实现的2D卷积层，接口与nn.Conv2d兼容"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None):
        """初始化卷积层"""
        super(Conv2d_triton, self).__init__()
        
        # 检查不支持的参数
        if groups != 1:
            raise ValueError("Conv2d_triton currently only supports groups=1")
        if padding_mode != 'zeros':
            raise ValueError("Conv2d_triton currently only supports padding_mode='zeros'")
        
        # 标准化参数格式
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        # 保存参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        
        # 创建权重和偏置参数
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(
            torch.empty((out_channels, in_channels, kernel_size[0], kernel_size[1]), **factory_kwargs)
        )
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化权重和偏置参数"""
        # 使用与nn.Conv2d相同的初始化方法
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        """前向传播"""
        return ConvTritonFunc.apply(
            input, self.weight, self.bias, self.stride, self.padding, self.dilation
        )
    
    def extra_repr(self):
        """返回模块的字符串表示"""
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.stride != (1, 1):
            s += ', stride={stride}'
        if self.padding != (0, 0):
            s += ', padding={padding}'
        if self.dilation != (1, 1):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


# 便捷函数，直接调用自动微分函数
triton_conv2d = ConvTritonFunc.apply


# benchmark

import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from tqdm import tqdm


def benchmark_conv2d(batch_size, in_channels, height, width, out_channels, kernel_size, 
                    stride=1, padding=0, dilation=1, num_warmup=10, num_runs=50, 
                    check_correctness=True, device='cuda'):
    """
    对比Conv2d_triton和nn.Conv2d的推理性能
    """
    # 标准化参数格式
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # 设定数据类型为float16
    dtype = torch.float16
    
    # 准备输入数据
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=dtype)
    
    # 创建卷积层实例，确保指定正确的dtype
    torch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
                                stride, padding, dilation, bias=True,
                                device=device, dtype=dtype)  # 明确指定dtype
    
    triton_conv = Conv2d_triton(in_channels, out_channels, kernel_size, 
                               stride, padding, dilation, bias=True,
                               device=device, dtype=dtype)  # 明确指定dtype
    
    # 确保两个模型使用相同的权重和偏置
    triton_conv.weight.data.copy_(torch_conv.weight.data)
    triton_conv.bias.data.copy_(torch_conv.bias.data)
    
    # 设置为评估模式
    torch_conv.eval()
    triton_conv.eval()
    
    # 预热运行
    print(f"Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            torch_out = torch_conv(x)
            triton_out = triton_conv(x)
    
    # 同步GPU
    torch.cuda.synchronize()
    
    # 测量PyTorch原生实现
    print(f"Benchmarking PyTorch...")
    torch_times = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            torch_out = torch_conv(x)
            end.record()
            
            torch.cuda.synchronize()
            torch_times.append(start.elapsed_time(end))
    
    # 测量Triton实现
    print(f"Benchmarking Triton...")
    triton_times = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            triton_out = triton_conv(x)
            end.record()
            
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))
    
    # 计算性能指标
    torch_time_mean = np.mean(torch_times)
    torch_time_std = np.std(torch_times)
    triton_time_mean = np.mean(triton_times)
    triton_time_std = np.std(triton_times)
    
    speedup = torch_time_mean / triton_time_mean
    
    # 检查结果正确性
    if check_correctness:
        max_diff = torch.max(torch.abs(torch_out - triton_out)).item()
        rel_diff = torch.norm(torch_out - triton_out) / torch.norm(torch_out)
        is_correct = torch.allclose(torch_out, triton_out, rtol=1e-2, atol=1e-2)
    else:
        max_diff = None
        rel_diff = None
        is_correct = None
    
    # 计算每次推理的浮点运算次数 (FLOPs)
    output_height = ((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
    output_width = ((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
    # 卷积操作的FLOPs = 2 * 输入通道 * 核大小 * 输出大小 * 输出通道 * 批次大小
    flops = 2 * in_channels * kernel_size[0] * kernel_size[1] * output_height * output_width * out_channels * batch_size
    
    # 计算TFLOPS (每秒万亿次浮点运算)
    torch_tflops = flops / (torch_time_mean * 1e-3) / 1e12
    triton_tflops = flops / (triton_time_mean * 1e-3) / 1e12
    
    # 收集结果
    results = {
        'config': {
            'batch_size': batch_size,
            'in_channels': in_channels,
            'height': height,
            'width': width,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'output_size': (output_height, output_width)
        },
        'pytorch': {
            'time_ms_mean': torch_time_mean,
            'time_ms_std': torch_time_std,
            'tflops': torch_tflops
        },
        'triton': {
            'time_ms_mean': triton_time_mean,
            'time_ms_std': triton_time_std,
            'tflops': triton_tflops
        },
        'comparison': {
            'speedup': speedup,
            'flops': flops
        },
        'correctness': {
            'max_diff': max_diff,
            'rel_diff': rel_diff,
            'is_correct': is_correct
        }
    }
    
    return results


def run_conv2d_benchmarks(configs=None, **kwargs):
    """
    运行一系列卷积配置的性能测试并可视化结果
    """
    if configs is None:
        # 默认测试配置
        configs = [
            # 图像分类模型中常见的卷积配置
            {'batch_size': 32, 'in_channels': 3, 'height': 224, 'width': 224, 'out_channels': 64, 'kernel_size': 7, 'stride': 2, 'padding': 3},
            {'batch_size': 32, 'in_channels': 64, 'height': 56, 'width': 56, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'batch_size': 32, 'in_channels': 128, 'height': 28, 'width': 28, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'batch_size': 32, 'in_channels': 256, 'height': 14, 'width': 14, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            
            # 不同批次大小
            {'batch_size': 1, 'in_channels': 64, 'height': 56, 'width': 56, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'batch_size': 8, 'in_channels': 64, 'height': 56, 'width': 56, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'batch_size': 64, 'in_channels': 64, 'height': 56, 'width': 56, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            
            # 特殊卷积类型
            {'batch_size': 32, 'in_channels': 256, 'height': 28, 'width': 28, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding': 0},  # 1x1 卷积
            {'batch_size': 32, 'in_channels': 64, 'height': 56, 'width': 56, 'out_channels': 64, 'kernel_size': 5, 'stride': 1, 'padding': 2},  # 5x5 卷积
        ]
    
    results = []
    
    for idx, config in enumerate(configs):
        print(f"\n\nRunning benchmark {idx+1}/{len(configs)}:")
        print(f"Config: {config}")
        
        try:
            result = benchmark_conv2d(**config, **kwargs)
            results.append(result)
            
            # 打印单个结果
            print(f"\nResults for benchmark {idx+1}:")
            print(f"PyTorch: {result['pytorch']['time_ms_mean']:.3f} ± {result['pytorch']['time_ms_std']:.3f} ms ({result['pytorch']['tflops']:.2f} TFLOPS)")
            print(f"Triton:  {result['triton']['time_ms_mean']:.3f} ± {result['triton']['time_ms_std']:.3f} ms ({result['triton']['tflops']:.2f} TFLOPS)")
            print(f"Speedup: {result['comparison']['speedup']:.2f}x")
            if result['correctness']['is_correct'] is not None:
                print(f"Correct: {'Yes' if result['correctness']['is_correct'] else 'No'}, Max diff: {result['correctness']['max_diff']:.6e}")
            
        except Exception as e:
            print(f"Error with config {config}: {e}")
    
    # 创建结果表格
    if results:
        table_data = []
        for result in results:
            config = result['config']
            row = [
                f"{config['batch_size']}",
                f"{config['in_channels']}",
                f"{config['height']}x{config['width']}",
                f"{config['out_channels']}",
                f"{config['kernel_size'][0]}x{config['kernel_size'][1]}",
                f"{config['stride'][0]}",
                f"{config['padding'][0]}",
                f"{result['pytorch']['time_ms_mean']:.2f}±{result['pytorch']['time_ms_std']:.2f}",
                f"{result['triton']['time_ms_mean']:.2f}±{result['triton']['time_ms_std']:.2f}",
                f"{result['comparison']['speedup']:.2f}x",
                f"{result['pytorch']['tflops']:.2f}",
                f"{result['triton']['tflops']:.2f}",
                "✓" if result['correctness']['is_correct'] else f"✗ ({result['correctness']['max_diff']:.1e})"
            ]
            table_data.append(row)
        
        headers = ["Batch", "In Ch", "Input Size", "Out Ch", "Kernel", "Stride", "Pad", 
                   "PyTorch (ms)", "Triton (ms)", "Speedup", "PT TFLOPS", "Triton TFLOPS", "Correct"]
        
        print("\n\nConv2D Benchmark Summary:")
        print(tabulate(table_data, headers=headers, tablefmt="pipe"))
        
        # 创建条形图比较
        plot_benchmark_results(results)
    else:
        print("\nNo successful benchmark results to display.")
    
    return results


def plot_benchmark_results(results):
    """绘制基准测试结果的可视化图表"""
    if not results:
        return
        
    plt.figure(figsize=(14, 10))
    
    # 提取数据
    configs = []
    pytorch_times = []
    triton_times = []
    speedups = []
    
    for result in results:
        config = result['config']
        config_str = f"B{config['batch_size']}_C{config['in_channels']}to{config['out_channels']}_K{config['kernel_size'][0]}"
        configs.append(config_str)
        pytorch_times.append(result['pytorch']['time_ms_mean'])
        triton_times.append(result['triton']['time_ms_mean'])
        speedups.append(result['comparison']['speedup'])
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. 执行时间对比图
    x = np.arange(len(configs))
    width = 0.35
    
    ax1.bar(x - width/2, pytorch_times, width, label='PyTorch')
    ax1.bar(x + width/2, triton_times, width, label='Triton')
    
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Conv2D Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 加速比图
    ax2.bar(x, speedups, color='green')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Triton Conv2D Speedup over PyTorch')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig('conv2d_benchmark_results.png')
    plt.show()


def main():
    """运行卷积基准测试的主函数"""
    print("Conv2D Performance Benchmark: PyTorch vs Triton")
    print("="*50)
    
    # 为了减少运行时间，可以减少测试配置的数量
    selected_configs = [
        # 选择几个有代表性的配置
        {'batch_size': 32, 'in_channels': 3, 'height': 224, 'width': 224, 'out_channels': 64, 'kernel_size': 7, 'stride': 2, 'padding': 3},
        {'batch_size': 32, 'in_channels': 64, 'height': 56, 'width': 56, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'batch_size': 32, 'in_channels': 256, 'height': 28, 'width': 28, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding': 0},  # 1x1 卷积
    ]
    
    # 减少预热和运行次数以加快测试
    results = run_conv2d_benchmarks(configs=selected_configs, num_warmup=3, num_runs=5, check_correctness=True)


if __name__ == "__main__":
    main()