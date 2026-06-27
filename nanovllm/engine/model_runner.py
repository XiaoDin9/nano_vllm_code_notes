import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

# scheduler: 选出本步要执行的请求数
# BlockManager: 分配 kv cache 的物理块
# ModelRuner: 输入转换成模型的格式 -> 执行前向传播 -> 采样得到下一个 token
class ModelRunner:

    # 初始化顺序: NCLL init -> 构建模型 -> 加载权重 -> warmup -> 分配 kv cache -> 捕获 Graph -> IPC 设置
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # NCCL 初始化
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # 设置默认 dtype 和 device
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        # 构建模型
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # 预热和资源分配
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 还原默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多卡 IPC 设置
        if self.world_size > 1:
            if rank == 0:           # 主进程创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()         # 非 rank0 进入等待循环

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
                
        if not self.enforce_eager:
            del self.graphs, self.graph_pool

        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # 稳定显存峰值，用 "最坏情况" 的输入跑一次，让 pytorch 完成所有内存分配和编译
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]

        for seq in seqs:
            seq.num_scheduled_tokens = seq_len

        self.run(seqs, True)
        torch.cuda.empty_cache()

    # 这个计算考虑了模型权重，激活值峰值等其它显存占用，只把剩余部分分给 kv cache
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # tp 拆分，分到每个 gpu 的头数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim",  hf_config.hidden_size // hf_config.num_attention_heads)

        # 每 block 占用的字节数, [2, L, 1, S, H, D], S: block_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        # 可用显存 = 总量 x 利用率 - 已用 - (峰值 - 当前)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # kv cache shape: [2, L, N, S, H, D], S: block_size, N: num_kvcache_blocks
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]                     # 第 layer_id 层的 k_cache 
                module.v_cache = self.kv_cache[1, layer_id]                     # 第 layer_id 层的 v_cache
                layer_id += 1

    # block_tables 是一个 2D 数组，每一行是一个序列的物理块列表，需要 pad 到相同的长度
    def prepare_block_tables(self, seqs: list[Sequence]):
        # eg: seqs = [Sequence([1, 2]), Sequence([3]), Sequence([4, 5, 6])], max_len = 4
        # =》 [[1, 2, -1, -1], [3, -1, -1, -1], [4, 5, 6, -1]]
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # 把多个 request 组织成一次 FlashAttention/PagedAttention kernel 输入
    '''
    ① 找出本轮真正要计算的 token（Query）
    ② 告诉 attention 要读取全部历史 KV（Key）
    ③ 告诉 GPU 新生成 KV 写入哪些物理 slot
    ④ 在 Prefix Cache 命中时，把旧 KV 和新 KV 拼接起来做 attention
    '''
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []

        # 每个 sequence 的 Query 起始 offset, eg: seq0 q_len=300, seq1 q_len=100, cu_seqlens_q = [0, 300, 400]
        # seq1 query = [0:300], seq2 query = [300:400]
        cu_seqlens_q = [0]    
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            # 只处理未缓存的部分 seq[start:end]
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end

            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)        # Query 长度（未缓存的部分）
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)        # Key 长度（完整的序列）
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:    # warmup 时，没有 block_table
                continue

            # 先算逻辑 block 范围: [start_block, end_block], 逻辑块是连续的，物理块可能不是连续的
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
    
            # 本轮算出的 KV，写到 KV Cache 哪里
            # 构造 slot_mapping: 只包含要写入的槽位
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size   # 物理的 slot_start 位置
                if i == start_block:
                    slot_start += start % self.block_size           # 物理槽位 = block_id x block_size + offset_in_block

                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size

                slot_mapping.extend(range(slot_start, slot_end))

        # 当 prefix cache 命中时，序列的一部分 token 已经有 kv cache 了，不需要重新计算
        # 所以 Query 只包含未缓存的部分，Key 包含完整的序列（缓存的部分从 cache 读取）
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache, 构造 block_tables, 让 Attention 知道取哪里读取缓存的 kv
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # Decode 阶段每个序列只有一个 token 输入
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 每个序列只有一个元素，就是新的 token 的 kv 要写入的位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    '''
    Eager 模式 = 每次请求动态调度、动态 launch CUDA kernel。
    Graph 模式 = 预先捕获固定执行图，后续直接 replay，减少 CPU launch 开销。
    '''
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # prefill 的 shape 变化大，不适合 graph 或 batch 太大，graph 收益不明显
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:   # decode 的 shape 固定，graph 能显著降低 kernel launch 开销
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 准备输入
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 执行模型
        logits = self.run_model(input_ids, positions, is_prefill)

        # 采样（只在 rank 0)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        reset_context()
        return token_ids

    # Q.???
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config

        max_bs = min(self.config.max_num_seqs, 512)                                         # 最大并行度（seq 数）
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size    # 每个序列最多包含的 block num
        
        # 预分配 staging 张量
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)

        # block_tables 是一个 2D 数组，每一行是一个序列的物理块列表，需要 pad 到相同的长度
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 要捕获的 batch size 列表，运行时选择第一个 >= 实际 bs 的图
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 逆序捕获：从大到小，这样大图先分配显存，小图复用, graph_pool 让多个图共享同一块显存
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 捕获前先跑一次，让 pytorch 完成编译和显存分配
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])        # warmup
            
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
                
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph

            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
