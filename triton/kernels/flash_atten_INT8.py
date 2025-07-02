import pytest
import torch

import triton
import triton.language as tl

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [32, 64, 128, 256]\
    for BN in [32, 64, 128, 256]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]

# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
#     for BM in [32]\
#     for BN in [32]\
#     for s in [3]\
#     for w in [4]\
# ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True

@triton.jit
def int8_attention_inner_kernel(
    acc,                # 累积的输出值
    norm_factor,        # softmax的归一化因子
    max_val,            # 数值稳定性的最大值
    query,              # 查询张量
    key_ptr,            # 键张量的块指针
    value_ptr,          # 值张量的块指针
    query_scale,        # 查询量化的缩放因子
    key_scale_ptr,      # 键缩放因子的块指针
    value_scale,        # 值量化的缩放因子
    start_block,        # 起始块索引
    qk_scale,           # 查询-键点积的缩放因子
    BLOCK_M: tl.constexpr,  # M维度的块大小
    HEAD_DIM: tl.constexpr, # 注意力头维度
    BLOCK_N: tl.constexpr,  # N维度的块大小
    STAGE: tl.constexpr,    # 处理阶段
    offs_m: tl.constexpr,   # M维度的偏移量
    offs_n: tl.constexpr,   # N维度的偏移量
    N_CTX: tl.constexpr,    # 上下文长度
    fp8_v: tl.constexpr     # 值是否为fp8格式
):
    """
    int8注意力计算的内核函数。
    
    根据STAGE参数处理注意力计算的不同部分:
    - STAGE 1: 处理当前块之前的token (因果注意力)
    - STAGE 2: 处理当前块中的token (带因果掩码)
    - STAGE 3: 处理所有token (非因果注意力)
    """
    # 根据STAGE确定处理范围
    if STAGE == 1:
        # 处理当前块之前的token
        lo, hi = 0, start_block * BLOCK_M
    elif STAGE == 2:
        # 处理当前块中的token (带因果掩码)
        lo, hi = start_block * BLOCK_M, (start_block + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)  # 确保正确对齐
    else:
        # 处理所有token (全注意力)
        lo, hi = 0, N_CTX
    
    # 将指针移动到起始位置
    key_ptr = tl.advance(key_ptr, (0, lo))
    key_scale_ptr = tl.advance(key_scale_ptr, (lo,))
    value_ptr = tl.advance(value_ptr, (lo, 0))
    
    # 处理键值对的块
    for block_start in range(lo, hi, BLOCK_N):
        block_start = tl.multiple_of(block_start, BLOCK_N)
        
        # 加载键块及其缩放因子
        key_block = tl.load(key_ptr)
        key_block_scale = tl.load(key_scale_ptr)
        
        # 计算查询-键注意力分数
        qk_scores = tl.dot(query, key_block).to(tl.float32)
        qk_scores = qk_scores * query_scale[:, None]  # 应用查询缩放
        qk_scores = qk_scores * key_block_scale       # 应用键缩放
        
        # 对带因果掩码的块应用掩码
        if STAGE == 2:
            # 创建因果掩码: 查询位置 >= 键位置
            mask = offs_m[:, None] >= (block_start + offs_n[None, :])
            # 应用掩码和缩放
            qk_scores = qk_scores * qk_scale + tl.where(mask, 0, -1.0e6)
            # 更新最大值以保持数值稳定性
            new_max = tl.maximum(max_val, tl.max(qk_scores, 1))
            # 用新的最大值归一化分数
            qk_scores -= new_max[:, None]
        else:
            # 非因果注意力或非当前块处理
            new_max = tl.maximum(max_val, tl.max(qk_scores, 1) * qk_scale)
            qk_scores = qk_scores * qk_scale - new_max[:, None]
        
        # 计算注意力概率
        probs = tl.math.exp2(qk_scores)
        block_norm = tl.sum(probs, 1)
        
        # 将概率量化为int8以高效处理
        probs = probs.to(tl.float16)
        probs = probs * 127  # 缩放到int8范围 [-127, 127]
        probs = (probs + 0.5).to(tl.int8)  # 四舍五入并转换为int8
        
        # 更新累积器和归一化因子
        alpha = tl.math.exp2(max_val - new_max)
        norm_factor = norm_factor * alpha + block_norm
        acc = acc * alpha[:, None]
        
        # 加载值块并计算加权和
        value_block = tl.load(value_ptr)
        output_block = tl.dot(probs, value_block)
        output_block = output_block.to(tl.float32)
        # 从int8量化恢复
        output_block = output_block * value_scale / 127
        acc = acc + output_block
        
        # 更新下一轮迭代的最大值
        max_val = new_max
        
        # 将指针前进到下一个块
        value_ptr = tl.advance(value_ptr, (BLOCK_N, 0))
        key_ptr = tl.advance(key_ptr, (0, BLOCK_N))
        key_scale_ptr = tl.advance(key_scale_ptr, (BLOCK_N,))
    
    return acc, norm_factor

@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def int8_attention_kernel(
    Q, K, V,                     # 查询、键、值张量 (int8格式)
    Q_scale, K_scale, V_scale,   # Q、K、V的缩放因子
    sm_scale,                    # Softmax缩放因子
    Out,                         # 输出张量
    
    # 张量的步长信息
    stride_qz, stride_qh, stride_qm, stride_qk,  # Q的步长
    stride_kz, stride_kh, stride_kn, stride_kk,  # K的步长
    stride_vz, stride_vh, stride_vk, stride_vn,  # V的步长
    stride_oz, stride_oh, stride_om, stride_on,  # 输出的步长
    
    # 缩放因子的步长信息
    stride_s1, stride_s2, stride_s3,  # Q/K缩放步长
    stride_v1, stride_v2,             # V缩放步长
    
    # 维度
    Z, H, N_CTX,                 # 批次大小、注意力头数、上下文长度
    
    # 常量
    HEAD_DIM: tl.constexpr,      # 每个头的维度
    BLOCK_M: tl.constexpr,       # M维度的块大小
    BLOCK_N: tl.constexpr,       # N维度的块大小
    STAGE: tl.constexpr          # 处理阶段 (1=非因果, 3=因果)
):
    """
    int8量化注意力的主要内核。
    
    使用int8量化高效计算注意力:
    attention(Q, K, V) = softmax(Q·K^T/sqrt(d_k))·V
    
    该实现以块为单位处理查询，支持因果和非因果注意力。
    """
    # 确保块大小约束
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    # 获取程序并行索引
    block_idx_m = tl.program_id(0)  # 序列维度的块索引
    batch_head_idx = tl.program_id(1)  # 批次和头的组合索引
    
    # 分解批次和头索引
    batch_idx = batch_head_idx // H  # 批次索引
    head_idx = batch_head_idx % H    # 头索引
    
    # 计算张量的基础偏移量
    qkv_offset = batch_idx.to(tl.int64) * stride_qz + head_idx.to(tl.int64) * stride_qh
    scale_offset = batch_idx.to(tl.int64) * stride_s1 + head_idx.to(tl.int64) * stride_s2
    v_scale_offset = batch_idx.to(tl.int64) * stride_v1 + head_idx.to(tl.int64) * stride_v2

    # 创建高效内存访问的块指针
    # 查询块指针
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(block_idx_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    # 值块指针
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    
    # 键块指针
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    
    # 输出块指针
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(block_idx_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    # 缩放向量指针
    Q_scale_ptr = tl.make_block_ptr(
        base=Q_scale + scale_offset,
        shape=(N_CTX,),
        strides=(stride_s3,),
        offsets=(block_idx_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    
    K_scale_ptr = tl.make_block_ptr(
        base=K_scale + scale_offset,
        shape=(N_CTX,),
        strides=(stride_s3,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    
    # 初始化位置跟踪的偏移量
    m_offsets = block_idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    
    # 初始化跟踪变量
    max_score = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    norm_factor = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    output_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # 准备softmax缩放因子 (转换为以2为底，用于exp2)
    softmax_scale = sm_scale * 1.44269504  # 1/log(2)
    
    # 加载查询和缩放因子 - 这些在整个计算过程中保留在SRAM中
    query = tl.load(Q_block_ptr)
    query_scale = tl.load(Q_scale_ptr)
    value_scale = tl.load(V_scale + v_scale_offset)
    
    # 阶段1: 处理非当前块部分
    # 对于causal=True, STAGE=3，内核使用STAGE=1
    # 对于causal=False, STAGE=1，内核使用STAGE=3
    if STAGE & 1:
        output_acc, norm_factor = int8_attention_inner_kernel(
            output_acc, norm_factor, max_score, query, 
            K_block_ptr, V_block_ptr, query_scale, K_scale_ptr, value_scale,
            block_idx_m, softmax_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE, m_offsets, n_offsets, N_CTX, V.dtype.element_ty == tl.float8e5
        )
    
    # 阶段2: 处理当前块部分 (如需要应用因果掩码)
    if STAGE & 2:
        output_acc, norm_factor = int8_attention_inner_kernel(
            output_acc, norm_factor, max_score, query,
            K_block_ptr, V_block_ptr, query_scale, K_scale_ptr, value_scale,
            block_idx_m, softmax_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2, m_offsets, n_offsets, N_CTX, V.dtype.element_ty == tl.float8e5
        )
    
    # 归一化累积的输出
    output_acc = output_acc / norm_factor[:, None]
    
    # 存储结果
    tl.store(O_block_ptr, output_acc.to(Out.type.element_ty))

class Int8Attention(torch.autograd.Function):
    """
    高效int8量化注意力实现。
    这是一个仅支持前向传播的自定义PyTorch autograd函数。
    """
    
    @staticmethod
    def forward(ctx, query, key, value, query_scale, key_scale, value_scale, causal, softmax_scale):
        """
        量化注意力的前向传播。
        
        参数:
            query (Tensor): int8格式的查询张量 [batch, heads, seq_len, head_dim]
            key (Tensor): int8格式的键张量 [batch, heads, seq_len, head_dim]
            value (Tensor): int8格式的值张量 [batch, heads, seq_len, head_dim]
            query_scale (Tensor): 查询的缩放因子
            key_scale (Tensor): 键的缩放因子
            value_scale (Tensor): 值的缩放因子
            causal (bool): 是否使用因果注意力掩码
            softmax_scale (float): softmax的缩放因子 (通常为1/sqrt(head_dim))
            
        返回:
            Tensor: float16格式的输出张量
        """
        # 验证形状约束
        head_dim_q, head_dim_k = query.shape[-1], key.shape[-1]
        head_dim_v = value.shape[-1]
        
        assert head_dim_q == head_dim_k and head_dim_k == head_dim_v, "头维度必须匹配"
        assert head_dim_k in {16, 32, 64, 128, 256}, "头维度必须是 {16, 32, 64, 128, 256} 之一"
        
        # 初始化输出张量
        output = torch.empty_like(query).to(torch.float16)
        
        # 根据因果性设置处理阶段
        stage = 3 if causal else 1
        
        # 为AMD目标配置额外的内核参数
        extra_args = {}
        if is_hip():
            waves_per_eu = 3 if head_dim_k <= 64 else 2
            extra_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        # 定义并行执行的网格
        def grid(args):
            return (
                triton.cdiv(query.shape[2], args["BLOCK_M"]),  # 序列维度的块数
                query.shape[0] * query.shape[1],                # 批次 * 头数
                1
            )
        
        # 启动内核
        int8_attention_kernel[grid](
            query, key, value,                      # 输入张量
            query_scale, key_scale, value_scale,    # 缩放因子
            softmax_scale, output,                  # Softmax缩放和输出
            
            # 步长信息
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            
            # 缩放因子步长信息
            query_scale.stride(0), query_scale.stride(1), query_scale.stride(2),
            value_scale.stride(0), value_scale.stride(1),
            
            # 维度
            query.shape[0], query.shape[1],         # 批次大小，头数
            N_CTX=query.shape[2],                   # 序列长度
            
            # 常量
            HEAD_DIM=head_dim_k,
            STAGE=stage,
            
            # AMD目标的额外参数
            **extra_args
        )

        # 保存上下文用于反向传播(如需要)
        ctx.sm_scale = softmax_scale
        ctx.HEAD_DIM = head_dim_k
        ctx.causal = causal
        
        return output

# 应用int8注意力的便捷函数
int8_attention = Int8Attention.apply