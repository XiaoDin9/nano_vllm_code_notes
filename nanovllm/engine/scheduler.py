from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs                                 # 最大并发请求数
        self.max_num_batched_tokens = config.max_num_batched_tokens             # 单步最多 token 数
        self.eos = config.eos
        self.block_size = config.kvcache_block_size                             # 每个 kv cache 块的 token 数
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()     # WAITTING 状态的请求
        self.running: deque[Sequence] = deque()     # RUNNING 状态的请求

    # 两个队列都为空，表示已完成
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill 优先, 会尝试 admit 尽可能多的新请求
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:      # 请求 seq 在这轮没法加入了
                break

            '''
            seq.block_table 为空
                = 这个 sequence 还没有分配过 KV Cache block
                = 这是 Prefill 阶段（新请求刚进入）
            因此需要判断:
            ① 这个 prompt 前缀有没有 cache 命中
            ② 需要新申请多少 block
            ③ free block 够不够
            '''
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:     # 没有足够的 kv cache 块时， num_cached_blocks == -1
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens                    # prefix cache 命中的部分不需要重新计算，只计入实际要处理的 token 数

            # eg: 假设 self.max_num_batched_tokens == 8k， seq0.num_tokens == 4k, 当前序列 seq1.num_tokens == 4.1k
            # scheduled_seqs[seq0], num_batched_tokens = 4k （seq0 的tokens), remainning 只剩下 4k, 不够保存当前 seq1 的所有 token 了, 因此本 batch 不包含 seq1
            # only allow chunked prefill for the first seq
            if remaining < num_tokens and scheduled_seqs:  
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)                       # chunked prefill for the first seq
            num_batched_tokens += seq.num_scheduled_tokens

            # chunked prefill for the first seq
            # eg0: self.max_num_batched_tokens == 4k， seq0.num_tokens == 4.1k, seq0.num_scheduled_tokens == 4k, seq0 还剩下 0.1k token 没有处理
            # 此时 seq0 还不能从 self.waiting 队列中弹出，还一轮还要继续执行 seq0 的 0.1k token
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode，没有新的请求需要 prefill ，就进入 Decode 阶段，处理 running 队列中的请求
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            # 如果当前没有足够的块给这个请求追加新 token，就需要释放一些资源
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())            # 抢占队尾 （最后加入的请求）
                else:
                    self.preempt(seq)                           # 自抢占
                    break
            else: # 有足够的块资源
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs                                   # Decode 阶段必须至少有一个请求被调度
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # 把这个请求踢回 waiting 队列，释放它的块
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)                            # 放入是队首，保证了公平性，不会让一个请求被反复抢占而饿死

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)

            # 检查终止条件， 遇到 EOS token 或 达到最大生成长度
            if (not seq.ignore_eos and token_id == self.eos) or \
                seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

