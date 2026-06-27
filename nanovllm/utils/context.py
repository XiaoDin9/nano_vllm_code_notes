from dataclasses import dataclass
import torch


# prefill 和 decode 两个阶段需要的元信息不同，用同一个 context 数据结构来统一管理
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None            # Query 累计长度
    cu_seqlens_k: torch.Tensor | None = None            # key 累计长度
    max_seqlen_q: int = 0                               # 最长 Query 长度
    max_seqlen_k: int = 0                               # 最长 key 长度
    slot_mapping: torch.Tensor | None = None            # kv 写入的物理槽位
    context_lens: torch.Tensor | None = None            # 每个序列的上下文长度 （Decode 用）
    block_tables: torch.Tensor | None = None            # 物理块表

# context 是一个全局单例，因为：context 需要被模型的各个层访问（主要是 Attention 层）
# 用全局变量可以在不改变模型接口的情况下传递额外的信息
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
