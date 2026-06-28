你这里看到的是 LLM（例如 vLLM、Llama）里 MLP 层 gate_proj 和 up_proj 合并加载（Merged Linear） 的典型实现。

先说结论：

这个 MergedColumnParallelLinear 的作用是：

把 MLP 中两个独立线性层：

gate_proj
up_proj

在内存中合并成一个大矩阵：

[ gate_proj ]
[ up_proj   ]

然后做 Tensor Parallel 切分加载

（严格说 PyTorch weight layout 是按 output 维拼接，下面详细画图）

1. 先理解 Transformer 中 MLP 结构

以 Llama 为例：

x
│
├── gate_proj (Linear)
│
├── up_proj   (Linear)
│
├── SiLU(gate)
│
│
└── element-wise multiply

      SiLU(gate) * up

│
▼

down_proj
│
▼
output

公式：

FFN(x)=down_proj(SiLU(gate_proj(x))⊙up_proj(x))

shape（Llama-7B）：

hidden_size = 4096
intermediate_size = 11008

两个矩阵：

gate_proj.weight = [11008,4096]

up_proj.weight   = [11008,4096]
2. 为什么合并 gate 和 up？

正常：

x
│
├─ matmul(gate_proj)
│
└─ matmul(up_proj)

要执行：

2次 GEMM

优化后：

把两个 weight 合并：

merged_weight = [22016,4096]

只做：

一次 GEMM

即：

x × merged_weight^T

得到：

[gate_output , up_output]

再 split：

gate = output[:11008]

up = output[11008:]

好处：

减少 kernel launch
提高 GPU 利用率
减少 memory read
3. MergedColumnParallelLinear 初始化

代码：

class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

假设：

input_size=4096

output_sizes=[11008,11008]

即：

gate_proj = 11008
up_proj   = 11008

调用：

super().__init__(
    4096,
    22016
)

最终 weight：

weight=[22016,4096]
4. 合并后的内存布局
weight=[22016,4096]

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

按 output dim 拼接：

dim=0

即：

torch.cat([gate, up], dim=0)
5. Tensor Parallel（Column Parallel）

假设：

tp_size = 2

两张卡：

GPU0
GPU1

每张卡只保存一半 output。

完整：

weight=[22016,4096]

切分：

GPU0 → [11008,4096]

GPU1 → [11008,4096]
6. weight_loader 分析

代码：

def weight_loader(
        self,
        param,
        loaded_weight,
        loaded_shard_id
):

其中：

loaded_shard_id=0 → gate_proj

loaded_shard_id=1 → up_proj

checkpoint 里：

gate_proj.weight

up_proj.weight

分两次调用。

7. shard_offset

代码：

shard_offset = sum(
        self.output_sizes[:loaded_shard_id]
    ) // self.tp_size

作用：

找到当前 shard 在 merged_weight 中的位置

假设：

output_sizes=[11008,11008]

tp_size=2
第一次加载 gate
loaded_shard_id=0

计算：

sum([])=0

所以：

offset=0
第二次加载 up
loaded_shard_id=1

计算：

sum([11008])=11008

除：

11008/2=5504

即：

offset=5504

因为：

每张 GPU 只保存一半。

8. shard_size

代码：

shard_size = self.output_sizes[
    loaded_shard_id
] // self.tp_size

即：

每张卡负责多少。

例如：

11008 /2 =5504

所以：

gate_size=5504

up_size=5504
9. narrow()

代码：

param_data = param_data.narrow(
        self.tp_dim,
        shard_offset,
        shard_size
)

假设：

tp_dim=0

即：

从 merged_weight 中取一部分。

完整：

param=[11008,4096]

GPU0 local merged weight

（每张卡本地保存 11008 rows）

布局：

┌─────────────────────┐
│ gate  [5504,4096]  │
├─────────────────────┤
│ up    [5504,4096]  │
└─────────────────────┘
10. loaded_weight chunk

代码：

loaded_weight = loaded_weight.chunk(
                    self.tp_size,
                    self.tp_dim
               )[self.tp_rank]

checkpoint：

例如：

gate_proj=[11008,4096]

切：

chunk(2,dim=0)

得到：

GPU0 → [5504,4096]

GPU1 → [5504,4096]

假设：

tp_rank=0

取：

第一块
11. copy_

代码：

param_data.copy_(loaded_weight)

复制。

整个加载流程图（最核心）

假设：

hidden=4096

intermediate=11008

tp=2

checkpoint：

gate_proj.weight=[11008,4096]

up_proj.weight=[11008,4096]

GPU0 local param：

merged_weight=[11008,4096]

布局：

GPU0 param

┌────────────────────────────┐
│ gate part                 │
│ [5504,4096]              │ ← offset=0
├────────────────────────────┤
│ up part                   │
│ [5504,4096]              │ ← offset=5504
└────────────────────────────┘

加载过程：

load gate_proj

checkpoint gate
[11008,4096]

        │ chunk(2)
        ▼

GPU0 gate
[5504,4096]

        │
        ▼

copy to offset=0


--------------------------------


load up_proj

checkpoint up
[11008,4096]

        │ chunk(2)
        ▼

GPU0 up
[5504,4096]

        │
        ▼

copy to offset=5504

最终：

GPU0 merged_weight

┌─────────────────────┐
│ gate shard          │
│ [5504,4096]        │
├─────────────────────┤
│ up shard           │
│ [5504,4096]        │
└─────────────────────┘
推理时一次 GEMM

原来：

x

 ├── gate_proj(x)
 └── up_proj(x)

现在：

x

  │

  ▼

merged_linear(x)

  │

  ▼

output=[gate,up]

  │ split

  ▼

gate = output[:5504]

up   = output[5504:]

一次：

GEMM

替代：

2次 GEMM

可以把它记成一句话：

MergedColumnParallelLinear

= 把多个 ColumnParallelLinear 的 weight
先 concat（按输出维）

再 tensor parallel 切分

最后一次 matmul 计算

如果你在看 nano-vLLM 源码，后面通常还会看到 QKVParallelLinear（Q、K、V 三个矩阵合并），原理和这个几乎一样，只是从 2 个矩阵 → 3 个矩阵。