from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

# 每个 block 是一个固定大小的存储单元，可以存放 block_size 个 token 的 kv 数据
class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0                      # 引用计数（支持块共享，多个请求如果有相同的 prefix，可以共享同一个块）
        self.hash = -1                          # 链式哈希值, -1 表示未定稿（用于 prefix cache, 满块才计算哈希，未满的块（开放块）hash 为 -1）
        self.token_ids = []                     # 块内的 token （用于校验），用于hash 碰撞时的内容校验

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()                          # 哈希 -> 块ID 映射
        self.free_block_ids: deque[int] = deque(range(num_blocks))              # 空闲块队列
        self.used_block_ids: set[int] = set()                                   # 已用块集合，跟 free_block_ids 互斥的

    # 链式哈希: 计算当前的哈希时，把前一个块的哈希值也加进去
    # 相同的块内容但不同的 prefix, 会得到不同的哈希值，只有prefix也相同时，哈希值才相同
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 分配新的物理内存块 block
    # (1) 重置 self.hash_to_block_id 表; (2) 重置 block.reset; (3) 从 free 移动到 used
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0

        # 如果这个空闲块是之前被使用过的 block(即：回收的块，需要清空下 hash_to_block_id 表的记录)
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        block.reset()                               # block 重置
        self.used_block_ids.add(block_id)
        return block_id

    # 块回收, notes: (1) 这里没有清空 hash_to_block_id; (2) 回收也仅仅是把 block_id 从 used 移动 free 队列，实际 block 中的数据都还在
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0     # 只有当前 block 的引用计数==0，才可以被回收
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1                                  # 前一个 block 的 hash
        num_cached_blocks = 0                   # 命中的 prefix block 数量
        num_new_blocks = seq.num_blocks         # 默认所有 block 都要新申请

        # 最后一个 block 不参与 cache 判断，因为：最后一个 block 可能是不完整 block（decode 会继续append token, 不共享）
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)                        # 取当前 block token
            h = self.compute_hash(token_ids, h)             # 计算当前 block 的链式哈希值
            block_id = self.hash_to_block_id.get(h, -1)

            # 判断 block i 的是否 cache 命中，命中的条件: bokck_id 不为 -1， 并且相同的块内容也相同（哈希冲突）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            
            # block(i) cache 命中
            num_cached_blocks += 1

            # 如果被复用的 block_id 正在被使用，可以直接共享，不需要重新申请
            if block_id in self.used_block_ids:
                num_new_blocks -= 1 

        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    # eg: seq 包含 256 x 2 + 70 个token, 即 3 个 num_blocks, 
    # 逻辑内存排布: block0(t0 ~ t255) -> block1(t255 ~ 510) -> block2(510 ~ 580)，假设其中 block0 是缓存命中的块，即: num_cached_blocks = 1
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table          # 确保这个 sequence 还没有分配过 KV Cache block
        h = -1

        # 处理哈希命中 + 内容一致的块
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)                    # 取当前 block token
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]         # 当前 seq 的block i 复用的块id 为: block_id
            block = self.blocks[block_id]

            if block_id in self.used_block_ids:         # 如果复用的块 block 正在被使用, 块的引用计数 + 1
                block.ref_count += 1
            else:                                       # 分配（从 free 移到 used)
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)

            seq.block_table.append(block_id)            # seq.block_table 保存，分配的物理块列表

        # 哈希未命中 或 内容不一致的块，分配新块
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())

        seq.num_cached_tokens = num_cached_blocks * self.block_size     # 缓存命中的 token 总数

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):      # 逆序遍历 block_table
            block = self.blocks[block_id]
            block.ref_count -= 1                        # 减少引用计数

            if block.ref_count == 0:                    # 只有引用计数 == 0，才进行真正的释放
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # 如果下一个 token 是新块的第一个，需要有空闲的块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:              # 新块的第一个 token，需要分配新块
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size         # 未考虑最后一个块

        if start == end:  # 可能缓存命中的 token + scheduled 的token 数还不足一个 block_size 大小
            return

        # 取前一个 block 的哈希值
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1

        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)                        # 取当前 block token

            h = self.compute_hash(token_ids, h)             # 计算当前 block 的链式哈希值
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id       # 登记到映射
