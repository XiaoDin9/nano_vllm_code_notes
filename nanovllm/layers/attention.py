import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# slot_mapping 告诉我们每个 token 的 kv 应该写到物理缓存的哪个位置，写操作用 trition 实现
# 每个 GPU 线程（program instance）负责 1 个 token，把该 token 的 K/V 向量写入 KV Cache 指定槽位
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    '''
    等价的写法:
    for idx in range(N):
        slot = slot_mapping[idx]
        if slot == -1:
            continue

        key = key_tensor[idx]
        value = value_tensor[idx]
        k_cache[slot] = key
        v_cache[slot] = value
    '''
    idx = tl.program_id(0)                      # 当前 GPU worker 编号
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:                              # pad token 不需要写 kv cache
        return

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)                    # 从 GPU 内存加载数据
    value = tl.load(value_ptr + value_offsets)

    cache_offsets = slot * D + tl.arange(0, D)

    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

# k_cache.shape: [num_blocks, block_size, num_heads, D], stride() 的值: [SxHxD, HxD, D, 1]
# k.shape: [N, num_kv_heads, D], stride() 的值: [HxD, D, 1]
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # 确保内存布局正确
    # tensor.stride(), 沿某个维度移动 1 个元素，在底层内存里需要跳过多少个元素
    # shape: (2, 3, 4), stride 就是:(3x4, 4, 1)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # 理解成 N 个 CUDA block/thread group
    store_kvcache_kernel[(N,)](key, key.stride(0),
                              value, value.stride(0), 
                              k_cache, 
                              v_cache, 
                              slot_mapping, 
                              D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 写入 kv cache , numel() 表示张量中的元素总数
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 执行 Attention
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache

            o = flash_attn_varlen_func(q,                                           # [total_q, num_heads, head_dim]
                                       k,                                           # [total_k, num_kv_heads, head_dim]
                                       v,                                           # [total_k, num_kv_heads, head_dim]
                                       max_seqlen_q=context.max_seqlen_q,           # int
                                       cu_seqlens_q=context.cu_seqlens_q,           # [batch_size + 1]
                                       max_seqlen_k=context.max_seqlen_k,           # int
                                       cu_seqlens_k=context.cu_seqlens_k,           # [batch_size + 1]
                                       softmax_scale=self.scale, 
                                       causal=True, 
                                       block_table=context.block_tables)            # [batch_size, max_blocks] 或 None, 它不为 None 时，flash attention 会从分页的 kv cache 中读取 k/v, 而不是传入的 k/v 参数
        else:    # decode
            # 每个序列只有一个 query, 但要和完整的历史 kv 做 attention
            # q.unsqueeze(1) 增加一个 seqlen 维度
            o = flash_attn_with_kvcache(q.unsqueeze(1),                             # [batch_size, 1, num_heads, head_dim]
                                        k_cache,                                    # [num_blocks, block_size, num_kv_heads, head_dim]
                                        v_cache,                                    # [num_blocks, block_size, num_kv_heads, head_dim]
                                        cache_seqlens=context.context_lens,         # [batch_size]， 告诉 Flash Attention 每个序列当前的长度，避免读取 padding 位置的无效数据
                                        block_table=context.block_tables,           # [batch_size, max_blocks]
                                        softmax_scale=self.scale,   
                                        causal=True)
        return o
