import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context

# 以输入 token_id 作为词表的行索引，取出对于的一行数据： [1, embedding_dim]
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,                                # 词表: [num_embeddings, embedding_dim]， 即：embedding_dim == hidden_size
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0

        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size                         # 按照此表的行进行切分，每一份大小: [self.num_embeddings // self.tp_size, embedding_dim]
        
        # 每个 gpu 的词表范围: []self.vocab_start_idx ~ self.vocab_end_idx， embedding_dim]
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))       # 获取 weight 的分片数据，大小为:  [self.num_embeddings // self.tp_size, embedding_dim]
        self.weight.weight_loader = self.weight_loader

    # 每个 rank 只持有词表的一部分
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size

        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # 检查输入 token_id 是否在当前 rank 的范围内
    # 在范围内的正常查表，不在范围内的置零
    # all_reduce 聚合所有 rank 的结果
    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 只处理属于当前 rank 的词表范围
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)

        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # 不在范围内的位置置零,然后 all_reduce 聚合
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()

        if context.is_prefill:
            # prefill 时只取每个序列的最后一个位置
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            # rank 0 gather 所有分片，拼接成完整词表
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
            
        return logits
