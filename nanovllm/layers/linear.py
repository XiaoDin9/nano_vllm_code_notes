import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0         # 保证能整除
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,          # 切分维度
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader     # 自定义加载函数，用于从完整权重中提取当前 rank 的分片

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)         # 0 - cols

    # checkpoint 权重（磁盘中的）  ---> loaded_weight
    #    模型中的参数             --->     param
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)           # 每块的大小: param_data.size // self.tp_dim
        start_idx = self.tp_rank * shard_size               # rank0(shared_size数据) -> ... -> rankN(shard_size数据), 当前块（id 是 tp_rank) 的开始索引
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)        # 从完整权重中切出对应的分片
        param_data.copy_(loaded_weight)                     # param 是模型注册的参数，直接复制会改变引用关系，即：拷贝数据，不改对象

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# pytorch 的线形层权重shape 是: [output_size, input_size]
''' 合并后的内存布局， 按 output dim 进行拼接
weight=[22016, 4096]
┌──────────────────────┐
│                      │
│ gate_proj.weight    │
│                      │
│ [11008,4096]        │
│                      │
├──────────────────────┤
│                      │
│ up_proj.weight      │
│                      │
│ [11008,4096]        │
│                      │
└──────────────────────┘
假设 tp_size = 2, 每张卡（gpu) 保持一半 output (gate && up 各自一半），即 gpu0 -> [11008, 4096], gpu1 -> [11008, 4096]
'''
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],            # [gate 的投影输出维度，up 的投影输出维度]
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # eg: input_size == 4096, output_sizes == [11008, 11008]，即: gate_proj = 11008, up_proj = 11008
        super().__init__(input_size, sum(output_sizes), bias)    # 生成的最终 weight shape: [22016, 4096]

    # 表示当前加载的是第几个子模块的权重，loaded_shard_id, 0 表示 gate, 1 表示 up
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data

        '''
        每张 GPU 的内存布局：
        ┌─────────────────────┐
        │ gate  [5504,4096]  │
        ├─────────────────────┤
        │ up    [5504,4096]  │
        └─────────────────────┘
        '''
        # 找到当前 shard 在 merged_weight 中的位置， loaded_shard_id=0, shard_offset = 0, loaded_shard_id = 1, shard_offset = 11008 // 2 = 5504 (每张 GPU 保持一半)
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# QKV 合并后的内存布局是: [Q | K | V], 每个部分按 head 数切分，loaded_shared_id 用字符串 "q" / "k" / "v" 来区分
# X: [B, S, H x D], Wq: [num_heads x head_dim, hidden_size], Wk/Wv: [num_kv_heads x head_dim, hidden_size]
# Q = matmul(X, Wq^T), K = matmul(X, Wk^T), V = matmul(X, Wv^T) => merged_weight = [Wq ; Wk ; Wv]，output = X × merged_weight^T
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]

        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)         # 1 - rows

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data

        # 针对这类算子：LayerNorm / RMSNorm / bias
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)          # bias 只在 rank0 加，避免重复
        if self.tp_size > 1:        # 聚合各 rank 的部分结果
            dist.all_reduce(y)
        return y
