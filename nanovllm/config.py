import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str                              # 模型的路径
    max_num_batched_tokens: int = 16384     # 单步最多的 token 数
    max_num_seqs: int = 512                 # 并发的请求数
    max_model_len: int = 4096               # 每个请求最多的 token 数
    gpu_memory_utilization: float = 0.9     # gpu 内存利用率
    tensor_parallel_size: int = 1           # 切分 tp 并行
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256           # 每个 kv cache 块的 token 数
    num_kvcache_blocks: int = -1            # kv cache 总的块数

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
